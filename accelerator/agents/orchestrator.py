#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import difflib
import json
import math
import os
import pathlib
import re
import shlex
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from typing import Any, Dict, List

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
RUNS_DIR = SCRIPT_DIR / "runs"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
DAILY_BUDGET_LEDGER_PATH = SCRIPT_DIR / "budget_ledger.json"

# Source: OpenAI model pricing pages (developers.openai.com), verified 2026-03-22.
# Rates are USD per 1M text tokens.
MODEL_PRICING_PER_1M: Dict[str, Dict[str, float]] = {
    "gpt-5.3-codex": {
        "input": 1.75,
        "cached_input": 0.175,
        "output": 14.0,
    },
    "gpt-5.4": {
        "input": 2.50,
        "cached_input": 0.25,
        "output": 15.0,
    },
}

# GPT-5.4 long-context multiplier applies when input exceeds 272K tokens.
MODEL_LONG_CONTEXT_MULTIPLIER: Dict[str, Dict[str, float]] = {
    "gpt-5.4": {
        "threshold_input_tokens": 272_000.0,
        "input_mult": 2.0,
        "output_mult": 1.5,
    }
}


SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
]


def redact_text(value: str, extra_secrets: List[str] | None = None) -> str:
    out = str(value)
    if extra_secrets:
        for secret in extra_secrets:
            if not isinstance(secret, str):
                continue
            token = secret.strip()
            if len(token) < 8:
                continue
            out = out.replace(token, "[REDACTED_SECRET]")
    for pat in SECRET_PATTERNS:
        out = pat.sub("[REDACTED_API_KEY]", out)
    return out


def sanitize_for_log(value: Any, extra_secrets: List[str] | None = None) -> Any:
    if isinstance(value, dict):
        return {str(k): sanitize_for_log(v, extra_secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [sanitize_for_log(v, extra_secrets) for v in value]
    if isinstance(value, tuple):
        return [sanitize_for_log(v, extra_secrets) for v in value]
    if isinstance(value, str):
        return redact_text(value, extra_secrets)
    return value


AGENT_A_SYSTEM = """You are Agent A (Planner) for the Accelera codebase.
Produce a concrete implementation plan with exact target files and tests.
Return STRICT JSON only (no markdown, no prose outside JSON).

Schema:
{
  "objective": "string",
  "assumptions": ["string"],
  "target_files": [{"path": "relative/path", "reason": "string"}],
  "edits": [{"path": "relative/path", "intent": "string"}],
  "tests": [{"command": "string", "purpose": "string"}],
  "risks": ["string"],
  "acceptance_criteria": ["string"]
}
"""


AGENT_B_PLAN_SYSTEM = """You are Agent B (Plan Critic/Reviewer) for Accelera.
Critique the proposed plan and return a revised plan focused on correctness and reliability.
Return STRICT JSON only.

Schema:
{
  "approve_plan": true,
  "critical_issues": ["string"],
  "revised_plan": {
    "objective": "string",
    "assumptions": ["string"],
    "target_files": [{"path": "relative/path", "reason": "string"}],
    "edits": [{"path": "relative/path", "intent": "string"}],
    "tests": [{"command": "string", "purpose": "string"}],
    "risks": ["string"],
    "acceptance_criteria": ["string"]
  },
  "review_focus": ["string"]
}
"""


AGENT_C_SYSTEM = """You are Agent C (Implementer) for Accelera.
Apply the approved plan using ONLY search-and-replace operations and/or explicit new file content.
Do not output diffs.
Return STRICT JSON only.

Schema:
{
  "summary": "string",
  "edits": [
    {
      "path": "relative/path",
      "create_if_missing": false,
      "new_file_content": null,
      "replacements": [
        {
          "search": "exact old text",
          "replace": "new text",
          "replace_all": false
        }
      ]
    }
  ],
  "notes": ["string"]
}

Rules:
- Use exact search strings from provided file context.
- Keep replacements minimal and deterministic.
- For a new file, set create_if_missing=true and set new_file_content to full file text.
- Do not include markdown fences.
"""


AGENT_B_REVIEW_SYSTEM = """You are Agent B (Implementation Reviewer) for Accelera.
Review implementation quality, correctness, and test evidence.
Return STRICT JSON only.

Schema:
{
  "approve": true,
  "blocking_issues": ["string"],
  "recommended_fixes": ["string"],
  "confidence": "low|medium|high"
}
"""


DEFAULT_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "your",
    "have", "will", "would", "should", "could", "about", "what", "when", "where",
    "which", "while", "using", "make", "run", "runs", "loop", "agent", "agents",
    "code", "file", "files", "task", "plan", "review", "implement", "implementation",
    "accelera", "python", "json", "output", "input", "api", "call", "calls", "test",
}


class OrchestratorError(RuntimeError):
    pass


class OpenAIAPIError(OrchestratorError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        body: str = "",
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = bool(retryable)
        self.body = body


def is_retryable_status(status_code: int | None) -> bool:
    return status_code in {408, 409, 429, 500, 502, 503, 504}


def to_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def get_model_pricing(model: str) -> Dict[str, float]:
    rates = MODEL_PRICING_PER_1M.get(model)
    if not rates:
        supported = ", ".join(sorted(MODEL_PRICING_PER_1M.keys()))
        raise OrchestratorError(
            f"No pricing table configured for model '{model}'. Supported: {supported}"
        )
    return rates


def estimate_tokens_from_text(text: str) -> int:
    if not text:
        return 1
    # Conservative estimate for budget guardrail: over-estimate to avoid overspend.
    return max(1, math.ceil(len(text) / 3.2))


def get_long_context_multipliers(model: str, input_tokens: int) -> tuple[float, float]:
    spec = MODEL_LONG_CONTEXT_MULTIPLIER.get(model)
    if not spec:
        return 1.0, 1.0
    threshold = float(spec.get("threshold_input_tokens", 0.0))
    if float(input_tokens) > threshold:
        return float(spec.get("input_mult", 1.0)), float(spec.get("output_mult", 1.0))
    return 1.0, 1.0


def extract_response_output_text(api_response: Dict[str, Any]) -> str:
    parts: List[str] = []
    output = api_response.get("output", [])
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "output_text":
                txt = str(item.get("text", ""))
                if txt:
                    parts.append(txt)
            content = item.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "output_text":
                        txt = str(block.get("text", ""))
                        if txt:
                            parts.append(txt)

    if not parts:
        output_text = api_response.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        if isinstance(output_text, list):
            for item in output_text:
                if isinstance(item, str) and item.strip():
                    parts.append(item.strip())

    return "\n".join(parts).strip()


def extract_response_usage(api_response: Dict[str, Any]) -> Dict[str, int]:
    usage = api_response.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    input_tokens = max(0, to_int(usage.get("input_tokens", 0)))
    output_tokens = max(0, to_int(usage.get("output_tokens", 0)))
    details = usage.get("input_tokens_details", {})
    if not isinstance(details, dict):
        details = {}
    cached_input_tokens = max(0, to_int(details.get("cached_tokens", 0)))
    return {
        "input_tokens": int(input_tokens),
        "output_tokens": int(output_tokens),
        "cached_input_tokens": int(cached_input_tokens),
    }


def compute_call_cost_usd(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int = 0,
) -> float:
    rates = get_model_pricing(model)
    input_rate = float(rates["input"])
    cached_rate = float(rates["cached_input"])
    output_rate = float(rates["output"])

    input_mult, output_mult = get_long_context_multipliers(model, input_tokens)
    input_rate *= input_mult
    output_rate *= output_mult

    cached = max(0, int(cached_input_tokens))
    total_input = max(0, int(input_tokens))
    billable_input = max(0, total_input - cached)
    out = max(0, int(output_tokens))

    return max(
        0.0,
        ((billable_input * input_rate) + (cached * cached_rate) + (out * output_rate)) / 1_000_000.0,
    )


class DailyBudgetGuardrail:
    def __init__(
        self,
        *,
        ledger_path: pathlib.Path,
        daily_budget_usd: float,
        enabled: bool = True,
        safety_buffer_usd: float = 0.05,
    ):
        self.ledger_path = ledger_path.resolve()
        self.daily_budget_usd = float(daily_budget_usd)
        self.enabled = bool(enabled)
        self.safety_buffer_usd = max(0.0, float(safety_buffer_usd))
        if self.daily_budget_usd <= 0.0:
            raise OrchestratorError("daily_budget_usd must be > 0.")

    def _today_key(self) -> str:
        return dt.datetime.now(dt.UTC).date().isoformat()

    def _load_ledger(self) -> Dict[str, Any]:
        if not self.ledger_path.exists():
            return {"version": 1, "updated_at": utc_now_iso(), "days": {}}
        try:
            data = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("ledger root is not object")
            if "days" not in data or not isinstance(data.get("days"), dict):
                data["days"] = {}
            return data
        except Exception as exc:
            raise OrchestratorError(f"Failed to read budget ledger at {self.ledger_path}: {exc}") from exc

    def _save_ledger(self, data: Dict[str, Any]) -> None:
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_suffix(self.ledger_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, self.ledger_path)

    def _ensure_day(self, data: Dict[str, Any], day_key: str) -> Dict[str, Any]:
        days = data.setdefault("days", {})
        day = days.get(day_key)
        if not isinstance(day, dict):
            day = {}
        day.setdefault("budget_usd", float(self.daily_budget_usd))
        day.setdefault("spent_usd", 0.0)
        day.setdefault("calls", [])
        day["budget_usd"] = float(self.daily_budget_usd)
        if not isinstance(day.get("calls"), list):
            day["calls"] = []
        days[day_key] = day
        return day

    def remaining_usd(self) -> float:
        data = self._load_ledger()
        day = self._ensure_day(data, self._today_key())
        return float(day.get("budget_usd", self.daily_budget_usd)) - float(day.get("spent_usd", 0.0))

    def clamp_max_output_tokens(
        self,
        *,
        model: str,
        requested_max_output_tokens: int,
        estimated_input_tokens: int,
    ) -> Dict[str, Any]:
        requested = max(1, int(requested_max_output_tokens))
        est_input = max(1, int(estimated_input_tokens))
        data = self._load_ledger()
        day_key = self._today_key()
        day = self._ensure_day(data, day_key)
        budget = float(day.get("budget_usd", self.daily_budget_usd))
        spent = float(day.get("spent_usd", 0.0))
        remaining = budget - spent

        rates = get_model_pricing(model)
        input_rate = float(rates["input"])
        output_rate = float(rates["output"])
        input_mult, output_mult = get_long_context_multipliers(model, est_input)
        input_rate *= input_mult
        output_rate *= output_mult

        est_input_cost = (est_input * input_rate) / 1_000_000.0
        min_reserve = self.safety_buffer_usd if self.enabled else 0.0
        available_for_output = remaining - est_input_cost - min_reserve

        if self.enabled and remaining <= 0.0:
            raise OrchestratorError(
                f"Daily budget exhausted for {day_key}: spent=${spent:.4f} / budget=${budget:.2f}"
            )

        if self.enabled and available_for_output <= 0.0:
            raise OrchestratorError(
                "Daily budget remaining is too low for another call after accounting for "
                f"estimated input tokens ({est_input}). Remaining=${remaining:.4f}, "
                f"estimated_input_cost=${est_input_cost:.4f}, reserve=${min_reserve:.4f}"
            )

        if self.enabled:
            max_by_budget = math.floor((available_for_output * 1_000_000.0) / output_rate)
            clamped = max(1, min(requested, int(max_by_budget)))
            if clamped < 64:
                raise OrchestratorError(
                    "Daily budget guardrail would allow fewer than 64 output tokens; "
                    f"remaining=${remaining:.4f}. Stopping to avoid low-signal spend."
                )
        else:
            clamped = requested

        est_total_cost = est_input_cost + ((clamped * output_rate) / 1_000_000.0)
        return {
            "day": day_key,
            "budget_usd": budget,
            "spent_usd": spent,
            "remaining_usd": remaining,
            "requested_max_output_tokens": requested,
            "allowed_max_output_tokens": int(clamped),
            "estimated_input_tokens": est_input,
            "estimated_call_cost_usd": est_total_cost,
        }

    def record_call(
        self,
        *,
        run_id: str,
        label: str,
        model: str,
        request_max_output_tokens: int,
        usage: Dict[str, int],
        response_id: str,
    ) -> Dict[str, Any]:
        input_tokens = max(0, int(usage.get("input_tokens", 0)))
        output_tokens = max(0, int(usage.get("output_tokens", 0)))
        cached_input_tokens = max(0, int(usage.get("cached_input_tokens", 0)))
        cost_usd = compute_call_cost_usd(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_input_tokens=cached_input_tokens,
        )

        data = self._load_ledger()
        day_key = self._today_key()
        day = self._ensure_day(data, day_key)
        before_spent = float(day.get("spent_usd", 0.0))
        after_spent = before_spent + cost_usd
        budget = float(day.get("budget_usd", self.daily_budget_usd))

        call_entry = {
            "timestamp": utc_now_iso(),
            "run_id": run_id,
            "label": label,
            "model": model,
            "request_max_output_tokens": int(request_max_output_tokens),
            "usage": {
                "input_tokens": input_tokens,
                "cached_input_tokens": cached_input_tokens,
                "output_tokens": output_tokens,
            },
            "response_id": response_id,
            "cost_usd": cost_usd,
        }
        day.setdefault("calls", []).append(call_entry)
        day["spent_usd"] = round(after_spent, 8)
        day["budget_usd"] = budget
        data["updated_at"] = utc_now_iso()
        self._save_ledger(data)

        return {
            "day": day_key,
            "budget_usd": budget,
            "spent_before_usd": before_spent,
            "spent_after_usd": after_spent,
            "remaining_after_usd": budget - after_spent,
            "call_cost_usd": cost_usd,
            "guardrail_enabled": bool(self.enabled),
        }


class RunLogger:
    def __init__(self, run_dir: pathlib.Path, secret_values: List[str] | None = None):
        self.run_dir = run_dir
        self.trace_dir = run_dir / "trace"
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(self.trace_dir.glob("*.json"))
        self.step = len(existing)
        self.secret_values = [s for s in (secret_values or []) if isinstance(s, str) and s.strip()]

    def _next_path(self, label: str) -> pathlib.Path:
        self.step += 1
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", label)
        return self.trace_dir / f"{self.step:03d}_{safe}.json"

    def log_call(
        self,
        *,
        label: str,
        request_payload: Dict[str, Any],
        response_text: str,
        parsed_response: Dict[str, Any] | List[Any] | None,
        api_response: Dict[str, Any] | None,
    ) -> pathlib.Path:
        payload = {
            "timestamp": utc_now_iso(),
            "label": label,
            "request_payload": request_payload,
            "response_text": response_text,
            "parsed_response": parsed_response,
            "api_response": api_response,
        }
        out = self._next_path(label)
        write_json(out, sanitize_for_log(payload, self.secret_values))
        return out

    def log_event(self, label: str, payload: Dict[str, Any]) -> pathlib.Path:
        out = self._next_path(label)
        write_json(
            out,
            sanitize_for_log(
                {
                    "timestamp": utc_now_iso(),
                    "label": label,
                    "event": payload,
                },
                self.secret_values,
            ),
        )
        return out


class OpenAIClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        fallback_model: str,
        reasoning_effort: str = "high",
        timeout_sec: int = 180,
    ):
        if not api_key.strip():
            raise OrchestratorError("OpenAI API key is empty.")
        self.api_key = api_key
        self.model = model.strip()
        self.fallback_model = fallback_model.strip()
        effort = reasoning_effort.strip().lower()
        if effort not in {"low", "medium", "high", "xhigh"}:
            raise OrchestratorError(
                f"Unsupported reasoning_effort '{reasoning_effort}'. Use low|medium|high|xhigh."
            )
        self.reasoning_effort = effort
        self.timeout_sec = max(10, int(timeout_sec))

    def _build_payload(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
    ) -> Dict[str, Any]:
        return {
            "model": model,
            "instructions": system_prompt,
            "input": user_prompt,
            "reasoning": {"effort": self.reasoning_effort},
            "max_output_tokens": int(max_output_tokens),
        }

    def _request(self, req_payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(req_payload).encode("utf-8")
        req = urllib.request.Request(
            OPENAI_RESPONSES_URL,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise OpenAIAPIError(
                f"OpenAI API HTTPError {exc.code}: {detail[:1200]}",
                status_code=int(exc.code),
                retryable=is_retryable_status(int(exc.code)),
                body=detail,
            ) from exc
        except urllib.error.URLError as exc:
            raise OpenAIAPIError(
                f"OpenAI API URLError: {exc}",
                retryable=True,
            ) from exc

        try:
            return json.loads(raw)
        except Exception as exc:
            raise OpenAIAPIError(
                f"OpenAI API returned non-JSON body: {raw[:1000]}",
                retryable=False,
            ) from exc

    def message(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int,
        model_override: str = "",
    ) -> Dict[str, Any]:
        model = model_override.strip() or self.model
        req_payload = self._build_payload(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_output_tokens=max_output_tokens,
        )
        response = self._request(req_payload)
        text = extract_response_output_text(response)
        return {
            "request": req_payload,
            "response": response,
            "text": text,
            "model_used": model,
        }


def utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: pathlib.Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_text_limited(path: pathlib.Path, max_chars: int) -> str:
    try:
        txt = path.read_text(encoding="utf-8")
    except Exception:
        return ""
    if max_chars > 0 and len(txt) > max_chars:
        return txt[:max_chars] + "\n...<truncated>...\n"
    return txt


def slugify(text: str, max_len: int = 64) -> str:
    out = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    if not out:
        out = "task"
    return out[:max_len].strip("-")


def extract_keywords(task: str, max_keywords: int = 10) -> List[str]:
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_\-]{2,}", task.lower())
    ranked: List[str] = []
    for w in words:
        if w in DEFAULT_STOPWORDS:
            continue
        if w not in ranked:
            ranked.append(w)
    ranked.sort(key=len, reverse=True)
    return ranked[:max_keywords]


def run_cmd(cmd: List[str], cwd: pathlib.Path, timeout_sec: int = 1200) -> Dict[str, Any]:
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        timed_out = False
        rc = int(proc.returncode)
        out = proc.stdout
        err = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        rc = 124
        out = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        err = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
    elapsed = round(time.time() - t0, 3)
    return {
        "command": " ".join(shlex.quote(x) for x in cmd),
        "returncode": rc,
        "timed_out": bool(timed_out),
        "elapsed_sec": elapsed,
        "stdout_tail": out[-20000:],
        "stderr_tail": err[-20000:],
    }


def discover_relevant_files(task: str, repo_root: pathlib.Path, max_files: int = 12) -> List[str]:
    keys = extract_keywords(task)
    selected: List[str] = []
    baseline = ["README.md", "IMPROVEMENT_LOG.md"]
    for p in baseline:
        if (repo_root / p).exists():
            selected.append(p)

    matches: List[str] = []
    if keys:
        cmd = ["rg", "-l", "-i"]
        for k in keys:
            cmd.extend(["-e", k])
        cmd.extend(
            [
                "--glob", "!data/**",
                "--glob", "!obj_dir/**",
                "--glob", "!OpenLane/**",
                "--glob", "!build/**",
                ".",
            ]
        )
        res = run_cmd(cmd, cwd=repo_root, timeout_sec=30)
        if res["returncode"] in (0, 1):
            for line in res["stdout_tail"].splitlines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("./"):
                    line = line[2:]
                if line not in matches:
                    matches.append(line)

    allowed_suffix = {
        ".py", ".md", ".json", ".txt", ".sv", ".h", ".c", ".yml", ".yaml", ".tcl"
    }
    scored: List[tuple[int, str]] = []
    for p in matches:
        suffix = pathlib.Path(p).suffix.lower()
        if suffix and suffix not in allowed_suffix:
            continue
        score = 0
        lp = p.lower()
        for k in keys:
            if k in lp:
                score += 2
        if p.startswith("auto_tuner/"):
            score += 1
        if p.startswith("compiler/"):
            score += 1
        if p.startswith("tests/"):
            score += 1
        scored.append((score, p))
    scored.sort(key=lambda x: (-x[0], x[1]))

    for _, p in scored:
        if p not in selected:
            selected.append(p)
        if len(selected) >= max_files:
            break

    return selected[:max_files]


def load_file_context(repo_root: pathlib.Path, rel_paths: List[str], max_chars: int = 8000) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for rel in rel_paths:
        p = (repo_root / rel).resolve()
        if not p.exists() or not p.is_file():
            continue
        txt = read_text_limited(p, max_chars=max_chars)
        out.append({"path": rel, "content": txt})
    return out


def gather_test_hints(repo_root: pathlib.Path, target_files: List[str], max_hits_per_file: int = 20) -> Dict[str, List[str]]:
    hints: Dict[str, List[str]] = {}
    for rel in target_files:
        base = pathlib.Path(rel).stem
        if not base:
            continue
        cmd = ["rg", "-n", "--no-heading", "-S", base, "tests", "test_model_blob.py", "test_sparse.py"]
        res = run_cmd(cmd, cwd=repo_root, timeout_sec=20)
        hits: List[str] = []
        if res["returncode"] in (0, 1):
            for line in res["stdout_tail"].splitlines():
                line = line.strip()
                if line:
                    hits.append(line)
                if len(hits) >= max_hits_per_file:
                    break
        hints[rel] = hits
    return hints


def extract_json_from_text(raw: str) -> Dict[str, Any] | List[Any]:
    raw = (raw or "").strip()
    if not raw:
        raise OrchestratorError("Model returned empty response text.")

    # Direct parse first.
    try:
        return json.loads(raw)
    except Exception:
        pass

    fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", raw, flags=re.S)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass

    # Fallback: first balanced JSON object.
    start = None
    depth = 0
    for i, ch in enumerate(raw):
        if ch == "{":
            if start is None:
                start = i
            depth += 1
        elif ch == "}":
            if start is not None:
                depth -= 1
                if depth == 0:
                    snippet = raw[start : i + 1]
                    try:
                        return json.loads(snippet)
                    except Exception:
                        start = None
                        depth = 0
                        continue
    raise OrchestratorError(f"Could not parse JSON response: {raw[:400]}")


def call_agent(
    *,
    client: OpenAIClient,
    budget_guardrail: DailyBudgetGuardrail,
    run_id: str,
    logger: RunLogger,
    label: str,
    system_prompt: str,
    input_payload: Dict[str, Any],
    max_tokens: int = 4096,
) -> Dict[str, Any]:
    user_prompt = (
        "Return STRICT JSON only. Do not include markdown code fences.\n"
        "INPUT:\n"
        + json.dumps(input_payload, indent=2, sort_keys=True)
    )
    model_candidates: List[str] = [client.model]
    if client.fallback_model and client.fallback_model not in model_candidates:
        model_candidates.append(client.fallback_model)

    estimated_input_tokens = estimate_tokens_from_text(system_prompt + "\n" + user_prompt)
    last_api_error: OpenAIAPIError | None = None

    for attempt_idx, model_name in enumerate(model_candidates, start=1):
        clamp_info = budget_guardrail.clamp_max_output_tokens(
            model=model_name,
            requested_max_output_tokens=max_tokens,
            estimated_input_tokens=estimated_input_tokens,
        )
        allowed_max_tokens = int(clamp_info["allowed_max_output_tokens"])
        try:
            call = client.message(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_output_tokens=allowed_max_tokens,
                model_override=model_name,
            )
        except OpenAIAPIError as exc:
            logger.log_event(
                f"{label}_api_error_attempt_{attempt_idx}",
                {
                    "attempt": attempt_idx,
                    "label": label,
                    "model": model_name,
                    "retryable": bool(exc.retryable),
                    "status_code": exc.status_code,
                    "error": str(exc),
                    "budget_clamp": clamp_info,
                },
            )
            last_api_error = exc
            if bool(exc.retryable) and attempt_idx < len(model_candidates):
                continue
            raise

        usage = extract_response_usage(call["response"])
        billing = budget_guardrail.record_call(
            run_id=run_id,
            label=label,
            model=model_name,
            request_max_output_tokens=allowed_max_tokens,
            usage=usage,
            response_id=str(call["response"].get("id", "")),
        )

        parsed_for_log: Dict[str, Any] | List[Any] | None = None
        parse_error = ""
        try:
            parsed = extract_json_from_text(call["text"])
            parsed_for_log = parsed if isinstance(parsed, dict) else {"_list": parsed}
        except Exception as exc:
            parsed = None
            parse_error = str(exc)
            parsed_for_log = {"_parse_error": parse_error}

        logger.log_call(
            label=label,
            request_payload={
                "model_candidates": model_candidates,
                "attempt": attempt_idx,
                "model_used": model_name,
                "reasoning_effort": client.reasoning_effort,
                "system_prompt": system_prompt,
                "input_payload": input_payload,
                "requested_max_tokens": int(max_tokens),
                "budget_clamp": clamp_info,
                "billing": billing,
            },
            response_text=call["text"],
            parsed_response=parsed_for_log,
            api_response=call["response"],
        )

        if parse_error:
            raise OrchestratorError(
                f"{label} produced non-JSON content (model={model_name}, attempt={attempt_idx}): {parse_error}"
            )
        if not isinstance(parsed, dict):
            raise OrchestratorError(f"{label} returned non-object JSON.")
        return parsed

    if last_api_error is not None:
        raise last_api_error
    raise OrchestratorError(f"{label} failed before receiving a model response.")


def extract_target_files_from_plan(plan: Dict[str, Any]) -> List[str]:
    out: List[str] = []
    for key in ("target_files", "edits"):
        val = plan.get(key, [])
        if not isinstance(val, list):
            continue
        for item in val:
            if isinstance(item, dict):
                p = str(item.get("path", "")).strip()
                if p and p not in out:
                    out.append(p)
            elif isinstance(item, str):
                p = item.strip()
                if p and p not in out:
                    out.append(p)
    return out


def ensure_repo_relative(path: str, repo_root: pathlib.Path) -> pathlib.Path:
    rel = pathlib.Path(path)
    resolved = (repo_root / rel).resolve()
    repo_resolved = repo_root.resolve()
    if not str(resolved).startswith(str(repo_resolved)):
        raise OrchestratorError(f"Path escapes repository root: {path}")
    return resolved


def apply_search_replace_edits(
    *,
    repo_root: pathlib.Path,
    edits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    repo_base = repo_root.resolve()
    changed_paths: List[str] = []
    details: List[Dict[str, Any]] = []
    errors: List[str] = []

    for idx, edit in enumerate(edits):
        if not isinstance(edit, dict):
            errors.append(f"edit[{idx}] is not an object")
            continue

        rel_path = str(edit.get("path", "")).strip()
        if not rel_path:
            errors.append(f"edit[{idx}] missing path")
            continue

        abs_path = ensure_repo_relative(rel_path, repo_root)
        create_if_missing = bool(edit.get("create_if_missing", False))
        new_file_content = edit.get("new_file_content", None)
        replacements = edit.get("replacements", [])

        existed = abs_path.exists()
        if existed:
            before = abs_path.read_text(encoding="utf-8")
        else:
            before = ""

        if (not existed) and (new_file_content is None) and (not create_if_missing):
            errors.append(f"{rel_path}: file does not exist and create_if_missing is false")
            continue

        text = before
        if new_file_content is not None:
            text = str(new_file_content)

        if replacements is None:
            replacements = []
        if not isinstance(replacements, list):
            errors.append(f"{rel_path}: replacements must be a list")
            continue

        local_changes = 0
        for ridx, rep in enumerate(replacements):
            if not isinstance(rep, dict):
                errors.append(f"{rel_path}: replacement[{ridx}] is not an object")
                continue
            search = str(rep.get("search", ""))
            replace = str(rep.get("replace", ""))
            replace_all = bool(rep.get("replace_all", False))
            if not search:
                errors.append(f"{rel_path}: replacement[{ridx}] has empty search")
                continue

            occurrences = text.count(search)
            if occurrences <= 0:
                errors.append(f"{rel_path}: replacement[{ridx}] search text not found")
                continue

            if replace_all:
                text = text.replace(search, replace)
                local_changes += occurrences
            else:
                text = text.replace(search, replace, 1)
                local_changes += 1

        if text != before or ((not existed) and (new_file_content is not None)):
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(text, encoding="utf-8")
            rel_saved = str(abs_path.relative_to(repo_base))
            if rel_saved not in changed_paths:
                changed_paths.append(rel_saved)
            details.append(
                {
                    "path": rel_saved,
                    "existed": bool(existed),
                    "bytes_before": len(before.encode("utf-8")),
                    "bytes_after": len(text.encode("utf-8")),
                    "replacement_events": int(local_changes),
                }
            )

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "changed_files": changed_paths,
        "details": details,
    }


def capture_file_snapshots(
    *,
    repo_root: pathlib.Path,
    before_map: Dict[str, str],
    changed_files: List[str],
    max_chars: int = 8000,
) -> List[Dict[str, Any]]:
    snapshots: List[Dict[str, Any]] = []
    for rel in changed_files:
        abs_path = ensure_repo_relative(rel, repo_root)
        after = read_text_limited(abs_path, max_chars=max_chars)
        before = before_map.get(rel, "")
        diff_lines = list(
            difflib.unified_diff(
                before.splitlines(),
                after.splitlines(),
                fromfile=f"a/{rel}",
                tofile=f"b/{rel}",
                lineterm="",
            )
        )
        diff_text = "\n".join(diff_lines[:400])
        snapshots.append(
            {
                "path": rel,
                "before_excerpt": before[-max_chars:],
                "after_excerpt": after[-max_chars:],
                "diff_excerpt": diff_text,
            }
        )
    return snapshots


def run_validation_suite(
    *,
    repo_root: pathlib.Path,
    python_bin: str,
    changed_files: List[str],
    run_make_verify: bool,
) -> Dict[str, Any]:
    commands: List[Dict[str, Any]] = []

    py_files = [p for p in changed_files if p.endswith(".py")]
    if py_files:
        cmd = [python_bin, "-m", "py_compile", *py_files]
        res = run_cmd(cmd, cwd=repo_root, timeout_sec=600)
        res["name"] = "py_compile"
        commands.append(res)
    else:
        commands.append(
            {
                "name": "py_compile",
                "command": "(skipped: no changed .py files)",
                "returncode": 0,
                "timed_out": False,
                "elapsed_sec": 0.0,
                "stdout_tail": "",
                "stderr_tail": "",
                "skipped": True,
            }
        )

    ut = run_cmd(
        [python_bin, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"],
        cwd=repo_root,
        timeout_sec=3600,
    )
    ut["name"] = "unittest_discover"
    commands.append(ut)

    if run_make_verify:
        mv = run_cmd(["make", "verify", f"PYTHON={python_bin}"], cwd=repo_root, timeout_sec=7200)
        mv["name"] = "make_verify"
        commands.append(mv)

    passed = True
    for c in commands:
        if c.get("skipped"):
            continue
        if int(c.get("returncode", 1)) != 0:
            passed = False
            break

    return {"passed": bool(passed), "commands": commands}


def summarize_feedback(
    *,
    apply_result: Dict[str, Any],
    validation: Dict[str, Any],
    review: Dict[str, Any],
) -> str:
    parts: List[str] = []
    if not apply_result.get("ok", False):
        parts.append("apply_edits_failed: " + "; ".join(apply_result.get("errors", [])[:6]))
    if not validation.get("passed", False):
        failing = [c for c in validation.get("commands", []) if not c.get("skipped") and int(c.get("returncode", 1)) != 0]
        for c in failing[:3]:
            parts.append(f"{c.get('name')}: rc={c.get('returncode')} stderr_tail={str(c.get('stderr_tail',''))[-500:]}")
    if not bool(review.get("approve", False)):
        issues = review.get("blocking_issues", [])
        if isinstance(issues, list) and issues:
            parts.append("review_blockers: " + " | ".join(str(x) for x in issues[:6]))
    if not parts:
        return "No actionable feedback captured."
    return "\n".join(parts)


def build_planner_payload(
    *,
    task: str,
    repo_root: pathlib.Path,
    relevant_files: List[str],
    max_file_chars: int,
) -> Dict[str, Any]:
    docs = {
        "README.md": read_text_limited(repo_root / "README.md", max_chars=max_file_chars),
        "IMPROVEMENT_LOG.md": read_text_limited(repo_root / "IMPROVEMENT_LOG.md", max_chars=max_file_chars),
    }
    file_context = load_file_context(repo_root, relevant_files, max_chars=max_file_chars)
    return {
        "task": task,
        "repository": {
            "root": str(repo_root),
            "docs": docs,
        },
        "relevant_files": file_context,
        "constraints": {
            "edit_style": "search_and_replace_pairs",
            "loop": "A_plan -> B_revise -> C_implement -> B_review -> validation",
            "max_rounds": 3,
        },
    }


def build_plan_review_payload(
    *,
    task: str,
    plan_a: Dict[str, Any],
    repo_root: pathlib.Path,
    max_file_chars: int,
) -> Dict[str, Any]:
    plan_files = extract_target_files_from_plan(plan_a)
    file_context = load_file_context(repo_root, plan_files, max_chars=max_file_chars)
    coverage = gather_test_hints(repo_root, plan_files)
    return {
        "task": task,
        "plan_a": plan_a,
        "target_file_context": file_context,
        "test_coverage_hints": coverage,
        "review_goal": "Critique and revise plan for correctness/reliability with concrete tests.",
    }


def build_implementation_payload(
    *,
    task: str,
    plan: Dict[str, Any],
    round_idx: int,
    max_rounds: int,
    feedback: str,
    repo_root: pathlib.Path,
    max_file_chars: int,
) -> Dict[str, Any]:
    target_files = extract_target_files_from_plan(plan)
    file_context = load_file_context(repo_root, target_files, max_chars=max_file_chars)
    return {
        "task": task,
        "round": int(round_idx),
        "max_rounds": int(max_rounds),
        "revised_plan": plan,
        "feedback_from_previous_round": feedback,
        "target_file_context": file_context,
        "implementation_rules": {
            "use_search_replace_pairs": True,
            "no_unrequested_refactors": True,
            "only_touch_needed_files": True,
        },
    }


def build_review_payload(
    *,
    task: str,
    plan: Dict[str, Any],
    round_idx: int,
    implementation: Dict[str, Any],
    apply_result: Dict[str, Any],
    validation: Dict[str, Any],
    snapshots: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "task": task,
        "round": int(round_idx),
        "revised_plan": plan,
        "implementation_response": implementation,
        "apply_result": apply_result,
        "validation": validation,
        "changed_file_snapshots": snapshots,
        "review_goal": "Decide approve=true only if implementation and tests are good.",
    }


def save_state(run_dir: pathlib.Path, state: Dict[str, Any]) -> None:
    write_json(run_dir / "state.json", state)


def load_state(run_dir: pathlib.Path) -> Dict[str, Any]:
    state_path = run_dir / "state.json"
    if not state_path.exists():
        raise OrchestratorError(f"Missing state file at {state_path}")
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise OrchestratorError(f"Failed to parse state file: {state_path}") from exc


def orchestrate(args: argparse.Namespace) -> int:
    repo_root = pathlib.Path(args.repo_root).resolve()
    if not repo_root.exists():
        raise OrchestratorError(f"Repository root does not exist: {repo_root}")
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise OrchestratorError(f"Environment variable {args.api_key_env} is not set.")

    if args.resume_run_dir:
        run_dir = pathlib.Path(args.resume_run_dir).resolve()
        if not run_dir.exists():
            raise OrchestratorError(f"Resume run directory does not exist: {run_dir}")
        state = load_state(run_dir)
        task = str(state.get("task", "")).strip()
        if not task:
            raise OrchestratorError("Resume state missing task.")
        print(f"[orchestrator] Resuming run: {run_dir}")
    else:
        if not args.task.strip():
            raise OrchestratorError("--task is required when not resuming.")
        task = args.task.strip()
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
        run_id = f"{stamp}_{slugify(task, max_len=48)}"
        run_dir = RUNS_DIR / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        state = {
            "run_id": run_id,
            "task": task,
            "status": "initializing",
            "created_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "model": args.model,
            "max_rounds": int(args.max_rounds),
            "next_round": 1,
            "feedback": "",
            "plan_a": {},
            "plan_b": {},
            "active_plan": {},
            "rounds": [],
            "run_make_verify": bool(args.make_verify),
            "python_bin": args.python_bin,
        }
        save_state(run_dir, state)
        print(f"[orchestrator] New run created: {run_dir}")

    run_id = str(state.get("run_id", run_dir.name)).strip() or run_dir.name
    budget_ledger_path = pathlib.Path(args.budget_ledger_path).resolve()
    state["run_id"] = run_id
    state["model"] = args.model
    state["fallback_model"] = args.fallback_model
    state["reasoning_effort"] = args.reasoning_effort
    state["daily_budget_usd"] = float(args.daily_budget_usd)
    state["budget_ledger_path"] = str(budget_ledger_path)
    state["budget_guardrail_enabled"] = bool(not args.disable_budget_guardrail)
    state["updated_at"] = utc_now_iso()
    save_state(run_dir, state)

    logger = RunLogger(run_dir, secret_values=[api_key])
    client = OpenAIClient(
        api_key=api_key,
        model=args.model,
        fallback_model=args.fallback_model,
        reasoning_effort=args.reasoning_effort,
        timeout_sec=args.api_timeout_sec,
    )
    budget_guardrail = DailyBudgetGuardrail(
        ledger_path=budget_ledger_path,
        daily_budget_usd=float(args.daily_budget_usd),
        enabled=bool(not args.disable_budget_guardrail),
        safety_buffer_usd=float(args.budget_safety_buffer_usd),
    )
    logger.log_event(
        "budget_guardrail_config",
        {
            "ledger_path": str(budget_ledger_path),
            "daily_budget_usd": float(args.daily_budget_usd),
            "enabled": bool(not args.disable_budget_guardrail),
            "safety_buffer_usd": float(args.budget_safety_buffer_usd),
            "primary_model": args.model,
            "fallback_model": args.fallback_model,
            "reasoning_effort": args.reasoning_effort,
        },
    )

    # Planning stage only if not already completed in state.
    active_plan = state.get("active_plan") or {}
    if not active_plan:
        state["status"] = "planning"
        state["updated_at"] = utc_now_iso()
        save_state(run_dir, state)

        relevant = discover_relevant_files(task, repo_root, max_files=args.max_context_files)
        logger.log_event("context_relevant_files", {"files": relevant})

        payload_a = build_planner_payload(
            task=task,
            repo_root=repo_root,
            relevant_files=relevant,
            max_file_chars=args.max_file_chars,
        )
        plan_a = call_agent(
            client=client,
            budget_guardrail=budget_guardrail,
            run_id=run_id,
            logger=logger,
            label="agent_a_plan",
            system_prompt=AGENT_A_SYSTEM,
            input_payload=payload_a,
            max_tokens=args.max_output_tokens,
        )

        payload_b_plan = build_plan_review_payload(
            task=task,
            plan_a=plan_a,
            repo_root=repo_root,
            max_file_chars=args.max_file_chars,
        )
        plan_b = call_agent(
            client=client,
            budget_guardrail=budget_guardrail,
            run_id=run_id,
            logger=logger,
            label="agent_b_revise_plan",
            system_prompt=AGENT_B_PLAN_SYSTEM,
            input_payload=payload_b_plan,
            max_tokens=args.max_output_tokens,
        )

        revised_plan = plan_b.get("revised_plan")
        if isinstance(revised_plan, dict) and revised_plan:
            active_plan = revised_plan
        else:
            active_plan = plan_a

        state["plan_a"] = plan_a
        state["plan_b"] = plan_b
        state["active_plan"] = active_plan
        state["status"] = "ready_for_round"
        state["updated_at"] = utc_now_iso()
        save_state(run_dir, state)

    max_rounds = int(state.get("max_rounds", args.max_rounds))
    next_round = int(state.get("next_round", 1))
    feedback = str(state.get("feedback", ""))

    if state.get("status") == "completed":
        print(f"[orchestrator] Run already completed: {run_dir}")
        return 0

    if next_round > max_rounds:
        print(f"[orchestrator] No remaining rounds for run: {run_dir}")
        return 1

    state["status"] = "running"
    state["updated_at"] = utc_now_iso()
    save_state(run_dir, state)

    success = False
    final_message = ""

    for round_idx in range(next_round, max_rounds + 1):
        print(f"[orchestrator] Round {round_idx}/{max_rounds}")

        impl_payload = build_implementation_payload(
            task=task,
            plan=active_plan,
            round_idx=round_idx,
            max_rounds=max_rounds,
            feedback=feedback,
            repo_root=repo_root,
            max_file_chars=args.max_file_chars,
        )

        impl = call_agent(
            client=client,
            budget_guardrail=budget_guardrail,
            run_id=run_id,
            logger=logger,
            label=f"agent_c_implement_round_{round_idx}",
            system_prompt=AGENT_C_SYSTEM,
            input_payload=impl_payload,
            max_tokens=args.max_output_tokens,
        )

        edits = impl.get("edits", [])
        if not isinstance(edits, list):
            edits = []

        # Capture before-state for changed/target files.
        before_map: Dict[str, str] = {}
        for p in extract_target_files_from_plan(active_plan):
            abs_path = ensure_repo_relative(p, repo_root)
            if abs_path.exists() and abs_path.is_file():
                before_map[p] = read_text_limited(abs_path, max_chars=args.max_file_chars)

        apply_result = apply_search_replace_edits(repo_root=repo_root, edits=edits)

        for rel in apply_result.get("changed_files", []):
            if rel not in before_map:
                abs_path = ensure_repo_relative(rel, repo_root)
                if abs_path.exists():
                    # For brand-new files, before excerpt is empty.
                    before_map[rel] = ""

        validation = run_validation_suite(
            repo_root=repo_root,
            python_bin=args.python_bin,
            changed_files=apply_result.get("changed_files", []),
            run_make_verify=bool(args.make_verify),
        )

        snapshots = capture_file_snapshots(
            repo_root=repo_root,
            before_map=before_map,
            changed_files=apply_result.get("changed_files", []),
            max_chars=args.max_file_chars,
        )

        review_payload = build_review_payload(
            task=task,
            plan=active_plan,
            round_idx=round_idx,
            implementation=impl,
            apply_result=apply_result,
            validation=validation,
            snapshots=snapshots,
        )

        review = call_agent(
            client=client,
            budget_guardrail=budget_guardrail,
            run_id=run_id,
            logger=logger,
            label=f"agent_b_review_round_{round_idx}",
            system_prompt=AGENT_B_REVIEW_SYSTEM,
            input_payload=review_payload,
            max_tokens=args.max_output_tokens,
        )

        approved = bool(review.get("approve", False))
        round_pass = bool(apply_result.get("ok", False)) and bool(validation.get("passed", False)) and approved

        round_record = {
            "round": int(round_idx),
            "implementation": impl,
            "apply_result": apply_result,
            "validation": validation,
            "review": review,
            "round_pass": bool(round_pass),
            "timestamp": utc_now_iso(),
        }
        state.setdefault("rounds", []).append(round_record)

        if round_pass:
            success = True
            final_message = f"Round {round_idx} passed (tests green and review approved)."
            state["status"] = "completed"
            state["next_round"] = int(round_idx + 1)
            state["feedback"] = ""
            state["updated_at"] = utc_now_iso()
            save_state(run_dir, state)
            break

        feedback = summarize_feedback(apply_result=apply_result, validation=validation, review=review)
        state["feedback"] = feedback
        state["next_round"] = int(round_idx + 1)
        state["status"] = "ready_for_round"
        state["updated_at"] = utc_now_iso()
        save_state(run_dir, state)

        print(f"[orchestrator] Round {round_idx} did not converge. Feedback prepared for next round.")

    if not success:
        state["status"] = "failed"
        state["updated_at"] = utc_now_iso()
        save_state(run_dir, state)
        final_message = (
            f"Did not converge in {max_rounds} rounds. "
            f"Inspect trace and state in: {run_dir}"
        )

    summary = {
        "run_id": state.get("run_id"),
        "task": task,
        "status": state.get("status"),
        "success": bool(success),
        "final_message": final_message,
        "rounds_attempted": len(state.get("rounds", [])),
        "updated_at": utc_now_iso(),
        "run_dir": str(run_dir),
    }
    write_json(run_dir / "summary.json", summary)

    print(f"[orchestrator] {final_message}")
    print(f"[orchestrator] run_dir={run_dir}")
    return 0 if success else 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Accelera deterministic A->B->C API orchestrator.")
    ap.add_argument("--task", type=str, default="", help="Task description to execute.")
    ap.add_argument("--resume-run-dir", type=str, default="", help="Resume from an existing agents/runs/<id> directory.")
    ap.add_argument("--model", type=str, default="gpt-5.3-codex", help="Primary OpenAI model.")
    ap.add_argument("--fallback-model", type=str, default="gpt-5.4", help="Fallback model used on retryable API errors.")
    ap.add_argument(
        "--reasoning-effort",
        type=str,
        default="high",
        choices=["low", "medium", "high", "xhigh"],
        help="Reasoning effort for OpenAI Responses calls.",
    )
    ap.add_argument("--api-key-env", type=str, default="OPENAI_API_KEY", help="Environment variable that stores OpenAI API key.")
    ap.add_argument("--api-timeout-sec", type=int, default=180, help="OpenAI HTTP timeout per call.")
    ap.add_argument("--max-output-tokens", type=int, default=4096, help="Max tokens per agent API response.")
    ap.add_argument("--daily-budget-usd", type=float, default=6.0, help="Daily spend cap in USD for orchestrator API calls.")
    ap.add_argument(
        "--budget-ledger-path",
        type=str,
        default=str(DAILY_BUDGET_LEDGER_PATH),
        help="JSON ledger path used to track daily spend and enforce budget cap.",
    )
    ap.add_argument(
        "--disable-budget-guardrail",
        action="store_true",
        help="Disable hard budget enforcement (usage is still recorded to the ledger).",
    )
    ap.add_argument(
        "--budget-safety-buffer-usd",
        type=float,
        default=0.05,
        help="Additional reserve per call to prevent budget overrun from token-estimate error.",
    )
    ap.add_argument("--max-rounds", type=int, default=3, help="Max implementation rounds.")
    ap.add_argument("--repo-root", type=str, default=str(REPO_ROOT), help="Repository root path.")
    ap.add_argument("--python-bin", type=str, default=sys.executable, help="Python binary for py_compile/unittest.")
    ap.add_argument("--make-verify", action="store_true", help="Also run `make verify` during validation.")
    ap.add_argument("--max-context-files", type=int, default=12, help="Max files included in planning context.")
    ap.add_argument("--max-file-chars", type=int, default=8000, help="Max chars loaded per file for model context.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if not args.task.strip() and not args.resume_run_dir:
        print("Either --task or --resume-run-dir is required.", file=sys.stderr)
        return 2

    try:
        return orchestrate(args)
    except KeyboardInterrupt:
        print("\n[orchestrator] KeyboardInterrupt. Current run state has been preserved where possible.")
        return 130
    except Exception as exc:
        print(f"[orchestrator] ERROR: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
