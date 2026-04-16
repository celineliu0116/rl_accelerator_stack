#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shlex
import subprocess
import threading
import time
from collections import Counter
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from mcp.server.fastmcp import FastMCP
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: mcp. Install with: python3 -m pip install -r mcp/requirements.txt"
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"
TRACE_PATH = DATA_DIR / "tuning_trace.jsonl"
JOBS_DIR = DATA_DIR / "mcp_jobs"

mcp = FastMCP("accelera")

_JOB_LOCK = threading.Lock()
_JOB_META: Dict[str, Dict[str, Any]] = {}
_JOB_PROCS: Dict[str, subprocess.Popen[str]] = {}
_JOB_LOGS: Dict[str, Any] = {}


def _clip_output(text: str, max_chars: int = 6000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _python_bin() -> str:
    """
    Generic Python for lightweight/export tasks.
    Override with ACCELERA_PYTHON.
    """
    return os.environ.get("ACCELERA_PYTHON", "python3").strip() or "python3"


def _rl_python_bin() -> str:
    """
    Python for RL/surrogate flows.
    Override with ACCELERA_RL_PYTHON, else fall back to ACCELERA_PYTHON, then python3.
    """
    return (
        os.environ.get("ACCELERA_RL_PYTHON")
        or os.environ.get("ACCELERA_PYTHON")
        or "python3"
    ).strip() or "python3"


def _run_cmd(argv: List[str], timeout_sec: int = 900) -> Dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            argv,
            cwd=str(REPO_ROOT),
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_sec)),
            check=False,
        )
        elapsed = time.time() - started
        return {
            "ok": proc.returncode == 0,
            "returncode": int(proc.returncode),
            "elapsed_sec": round(elapsed, 3),
            "command": argv,
            "stdout_tail": _clip_output(proc.stdout or ""),
            "stderr_tail": _clip_output(proc.stderr or ""),
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.time() - started
        return {
            "ok": False,
            "returncode": -1,
            "elapsed_sec": round(elapsed, 3),
            "command": argv,
            "error": f"timeout after {timeout_sec}s",
            "stdout_tail": _clip_output((exc.stdout or "") if isinstance(exc.stdout, str) else ""),
            "stderr_tail": _clip_output((exc.stderr or "") if isinstance(exc.stderr, str) else ""),
        }


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _new_job_id(kind: str) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    return f"{kind}_{stamp}_{os.getpid()}_{int(time.time() * 1000) % 1000:03d}"


def _read_json(path: Path) -> Dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _find_run_manifest(run_id: str) -> Dict[str, Any]:
    rid = str(run_id).strip()
    if not rid:
        return {"exists": False, "reason": "run_id is required"}
    if rid.endswith(".json"):
        rid = rid[:-5]

    if not RUNS_DIR.exists():
        return {"exists": False, "reason": "data/runs directory not found"}

    candidate = RUNS_DIR / f"{rid}.json"
    if candidate.exists():
        payload = _read_json(candidate)
        if payload is None:
            return {"exists": True, "path": str(candidate), "error": "invalid json"}
        return {"exists": True, "path": str(candidate), "manifest": payload}

    for p in RUNS_DIR.glob("run_*.json"):
        payload = _read_json(p)
        if isinstance(payload, dict) and str(payload.get("run_id", "")) == rid:
            return {"exists": True, "path": str(p), "manifest": payload}
    return {"exists": False, "reason": f"run manifest not found for run_id={rid}"}


def _latest_run_manifest() -> Dict[str, Any]:
    if not RUNS_DIR.exists():
        return {"exists": False, "reason": "data/runs directory not found"}

    run_files = sorted(RUNS_DIR.glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not run_files:
        return {"exists": False, "reason": "no run manifests present"}

    latest = run_files[0]
    try:
        payload = json.loads(latest.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"exists": True, "path": str(latest), "error": "invalid json"}

    return {"exists": True, "path": str(latest), "manifest": payload}


def _tail_trace(limit: int = 20, run_id: str = "") -> Dict[str, Any]:
    if not TRACE_PATH.exists():
        return {"exists": False, "reason": "trace file not found", "path": str(TRACE_PATH)}

    buf: deque[Dict[str, Any]] = deque(maxlen=max(1, int(limit)))
    with TRACE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                continue
            if run_id and str(row.get("run_id", "")) != run_id:
                continue
            buf.append(row)

    return {
        "exists": True,
        "path": str(TRACE_PATH),
        "count": len(buf),
        "rows": list(buf),
    }


def _inference_make_cmd(
    inference_src: str,
    enable_dual_issue: bool,
    generic_cflags: str,
    use_systolic_accel: bool,
) -> List[str]:
    src = str(inference_src).strip()
    # Keep INFERENCE_SRC constrained to firmware C file names to avoid accidental misuse.
    if "/" in src or "\\" in src or not src.endswith(".c"):
        raise ValueError("inference_src must be a C filename (for example, inference_generic.c)")
    cmd = [
        "make",
        "run",
        f"INFERENCE_SRC={src}",
        f"ENABLE_DUAL_ISSUE={1 if enable_dual_issue else 0}",
        f"GENERIC_CFLAGS={generic_cflags}",
    ]
    if use_systolic_accel:
        cmd.append("EXTRA_VFLAGS=-DUSE_SYSTOLIC_ACCEL")
    return cmd


def _ensure_jobs_dir() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)


def _finalize_job_locked(job_id: str, status: str, returncode: int, error: str = "") -> None:
    meta = _JOB_META.get(job_id)
    if not isinstance(meta, dict):
        return
    if meta.get("status") != "running":
        return
    meta["status"] = status
    meta["returncode"] = int(returncode)
    meta["error"] = error
    meta["ended_at"] = _utc_now_iso()
    started = float(meta.get("started_ts", time.time()))
    meta["elapsed_sec"] = round(max(0.0, time.time() - started), 3)

    # Attach additional structured outputs for known job types.
    if meta.get("job_type") == "eval_matrix":
        before_csv = set(meta.get("csv_before", []))
        after_csv = {p.name for p in DATA_DIR.glob("rl_eval_matrix_*.csv")} if DATA_DIR.exists() else set()
        meta["new_csv_files"] = [str(DATA_DIR / n) for n in sorted(after_csv - before_csv)]
    if meta.get("job_type") == "rl_daemon":
        meta["latest_run"] = _latest_run_manifest()

    proc = _JOB_PROCS.pop(job_id, None)
    _ = proc
    logf = _JOB_LOGS.pop(job_id, None)
    if logf is not None:
        try:
            logf.flush()
            logf.close()
        except Exception:
            pass


def _refresh_job_locked(job_id: str) -> None:
    meta = _JOB_META.get(job_id)
    if not isinstance(meta, dict):
        return
    if meta.get("status") != "running":
        return
    proc = _JOB_PROCS.get(job_id)
    if proc is None:
        _finalize_job_locked(job_id, status="failed", returncode=-1, error="process handle missing")
        return

    timeout_sec = int(meta.get("timeout_sec", 0))
    started = float(meta.get("started_ts", time.time()))
    if timeout_sec > 0 and (time.time() - started) > timeout_sec:
        try:
            proc.kill()
        except Exception:
            pass
        rc = proc.poll()
        if rc is None:
            rc = -1
        _finalize_job_locked(job_id, status="timeout", returncode=rc, error=f"timeout after {timeout_sec}s")
        return

    rc = proc.poll()
    if rc is None:
        return
    status = "completed" if int(rc) == 0 else "failed"
    _finalize_job_locked(job_id, status=status, returncode=int(rc))


def _refresh_all_jobs() -> None:
    with _JOB_LOCK:
        for job_id in list(_JOB_META.keys()):
            _refresh_job_locked(job_id)


def _log_tail(path: Path, max_chars: int = 6000) -> str:
    if not path.exists():
        return ""
    try:
        txt = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return _clip_output(txt, max_chars=max_chars)


def _start_job(job_type: str, command: List[str], timeout_sec: int) -> Dict[str, Any]:
    _ensure_jobs_dir()
    job_id = _new_job_id(job_type)
    log_path = JOBS_DIR / f"{job_id}.log"
    started_ts = time.time()
    started_at = _utc_now_iso()

    csv_before: List[str] = []
    if job_type == "eval_matrix" and DATA_DIR.exists():
        csv_before = sorted(p.name for p in DATA_DIR.glob("rl_eval_matrix_*.csv"))

    try:
        logf = log_path.open("w", encoding="utf-8")
        logf.write(f"[{started_at}] start job={job_id} type={job_type}\n")
        logf.write("command: " + " ".join(command) + "\n\n")
        logf.flush()
        proc = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            text=True,
            stdout=logf,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
    except Exception as exc:
        return {"ok": False, "error": f"failed to start job: {exc}", "command": command}

    with _JOB_LOCK:
        _JOB_PROCS[job_id] = proc
        _JOB_LOGS[job_id] = logf
        _JOB_META[job_id] = {
            "job_id": job_id,
            "job_type": job_type,
            "status": "running",
            "command": command,
            "pid": int(proc.pid),
            "log_path": str(log_path),
            "started_at": started_at,
            "started_ts": started_ts,
            "timeout_sec": int(timeout_sec),
            "returncode": None,
            "error": "",
            "csv_before": csv_before,
        }
        _refresh_job_locked(job_id)
        meta = dict(_JOB_META[job_id])

    return {"ok": True, "job": meta}


def _iter_trace_rows(run_id: str = "", last_n: int = 0) -> Iterable[Dict[str, Any]]:
    if not TRACE_PATH.exists():
        return []
    rid = str(run_id).strip()
    if last_n > 0:
        dq: deque[Dict[str, Any]] = deque(maxlen=int(last_n))
        with TRACE_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                try:
                    row = json.loads(s)
                except json.JSONDecodeError:
                    continue
                if rid and str(row.get("run_id", "")) != rid:
                    continue
                dq.append(row)
        return list(dq)

    rows: List[Dict[str, Any]] = []
    with TRACE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                row = json.loads(s)
            except json.JSONDecodeError:
                continue
            if rid and str(row.get("run_id", "")) != rid:
                continue
            rows.append(row)
    return rows


def _to_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _to_int(x: Any) -> int | None:
    try:
        return int(x)
    except Exception:
        return None


def _cycles_delta_pct(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    if baseline <= 0:
        return None
    return round(((candidate - baseline) / baseline) * 100.0, 4)


def _summarize_trace_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    valid = 0
    rejected = 0
    constraint_rejected = 0
    pruned = 0
    best_cycles = None
    sum_valid_cycles = 0.0
    sum_reward_all = 0.0
    mode_counts: Counter[str] = Counter()
    top_valid: List[Dict[str, Any]] = []

    for row in rows:
        total += 1
        reward = _to_float(row.get("reward", -10000.0))
        sum_reward_all += reward if reward is not None else -10000.0

        rr = row.get("reject_reason")
        has_reject = isinstance(rr, str) and rr.strip().lower() not in {"", "none"}
        if rr is not None and not isinstance(rr, str):
            has_reject = True
        if has_reject:
            rejected += 1

        violations = row.get("constraint_violations", [])
        if not isinstance(violations, list):
            violations = []
        constraint_ok = _to_int(row.get("constraint_ok", 1))
        if violations or constraint_ok == 0:
            constraint_rejected += 1

        action_pruned = _to_int(row.get("action_pruned", 0))
        if action_pruned == 1:
            pruned += 1

        mode = str(
            row.get("executed_mode_name")
            or row.get("mode_name")
            or row.get("hardware_dataflow_mode")
            or "UNKNOWN"
        )
        mode_counts[mode] += 1

        is_valid = _to_int(row.get("is_valid", 0)) == 1
        if not is_valid:
            cycles = _to_int(row.get("cycles", -1))
            simulated = _to_int(row.get("simulated", 0))
            correctness = _to_int(row.get("correctness_passed", 0))
            is_valid = (
                simulated == 1
                and correctness == 1
                and cycles is not None
                and cycles > 0
            )
        if not is_valid:
            continue

        valid += 1
        cycles = _to_int(row.get("cycles", -1))
        if cycles is None or cycles <= 0:
            continue
        sum_valid_cycles += float(cycles)
        if best_cycles is None or cycles < best_cycles:
            best_cycles = cycles

        top_valid.append(
            {
                "run_id": str(row.get("run_id", "")),
                "cycles": int(cycles),
                "tile_m": _to_int(row.get("tile_m")),
                "tile_n": _to_int(row.get("tile_n")),
                "burst_size": _to_int(row.get("burst_size")),
                "prefetch_depth": _to_int(row.get("prefetch_depth")),
                "tile_b": _to_int(row.get("tile_b")),
                "hardware_dataflow_mode": _to_int(row.get("hardware_dataflow_mode")),
            }
        )
        top_valid.sort(key=lambda r: int(r["cycles"]))
        if len(top_valid) > 5:
            top_valid = top_valid[:5]

    avg_valid_cycles = (sum_valid_cycles / valid) if valid > 0 else None
    avg_reward_all = (sum_reward_all / total) if total > 0 else None

    return {
        "total_rows": int(total),
        "valid_rows": int(valid),
        "valid_ratio": round(valid / total, 4) if total > 0 else None,
        "rejected_rows": int(rejected),
        "constraint_rejected_rows": int(constraint_rejected),
        "action_pruned_rows": int(pruned),
        "best_cycles": int(best_cycles) if best_cycles is not None else None,
        "avg_valid_cycles": round(avg_valid_cycles, 4) if avg_valid_cycles is not None else None,
        "avg_reward_all_rows": round(avg_reward_all, 6) if avg_reward_all is not None else None,
        "mode_counts": dict(mode_counts),
        "fastest_valid_examples": top_valid,
    }


@mcp.tool()
def project_status() -> Dict[str, Any]:
    """Return basic Accelera project status and latest tuning run info."""
    _refresh_all_jobs()
    running_jobs = 0
    with _JOB_LOCK:
        running_jobs = sum(1 for j in _JOB_META.values() if j.get("status") == "running")
    latest = _latest_run_manifest()
    return {
        "repo_root": str(REPO_ROOT),
        "data_dir_exists": DATA_DIR.exists(),
        "runs_dir_exists": RUNS_DIR.exists(),
        "trace_exists": TRACE_PATH.exists(),
        "jobs_dir_exists": JOBS_DIR.exists(),
        "running_jobs": int(running_jobs),
        "latest_run": latest,
    }


@mcp.tool()
def export_mnist_reference(timeout_sec: int = 600) -> Dict[str, Any]:
    """Run the reference MNIST workload export script."""
    return _run_cmd([_python_bin(), "workloads/mnist/mnist_mlp_export.py"], timeout_sec=timeout_sec)


@mcp.tool()
def run_inference(
    inference_src: str = "inference_generic.c",
    enable_dual_issue: bool = True,
    generic_cflags: str = "-O3 -fno-strict-aliasing",
    use_systolic_accel: bool = True,
    timeout_sec: int = 1800,
) -> Dict[str, Any]:
    """Run cycle-accurate inference through the existing Makefile flow."""
    clean = _run_cmd(["make", "clean"], timeout_sec=300)
    if not clean["ok"]:
        return {"stage": "make clean", "result": clean}

    try:
        make_cmd = _inference_make_cmd(
            inference_src=inference_src,
            enable_dual_issue=enable_dual_issue,
            generic_cflags=generic_cflags,
            use_systolic_accel=use_systolic_accel,
        )
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    run = _run_cmd(make_cmd, timeout_sec=timeout_sec)
    return {"stage": "make run", "result": run}


@mcp.tool()
def run_rl_daemon(
    timesteps: int = 8,
    workload: str = "all",
    eval_workloads: str = "all",
    eval_episodes: int = 2,
    n_steps: int = 4,
    batch_size: int = 4,
    timeout_sec: int = 7200,
) -> Dict[str, Any]:
    """Run the RL daemon with bounded defaults for quick iteration."""
    cmd = [
        _rl_python_bin(),
        "auto_tuner/rl_daemon.py",
        "--timesteps",
        str(int(timesteps)),
        "--workload",
        workload,
        "--eval-workloads",
        eval_workloads,
        "--eval-episodes",
        str(int(eval_episodes)),
        "--n-steps",
        str(int(n_steps)),
        "--batch-size",
        str(int(batch_size)),
    ]
    run = _run_cmd(cmd, timeout_sec=timeout_sec)
    run["latest_run"] = _latest_run_manifest()
    return run


@mcp.tool()
def run_eval_matrix(
    scenarios: str = "default",
    timesteps: int = 8,
    eval_episodes: int = 4,
    n_steps: int = 2,
    batch_size: int = 2,
    timeout_sec: int = 7200,
) -> Dict[str, Any]:
    """Run eval_matrix.py and report the latest generated CSV."""
    before = {p.name for p in DATA_DIR.glob("rl_eval_matrix_*.csv")} if DATA_DIR.exists() else set()

    cmd = [
        _rl_python_bin(),
        "auto_tuner/eval_matrix.py",
        "--scenarios",
        scenarios,
        "--timesteps",
        str(int(timesteps)),
        "--eval-episodes",
        str(int(eval_episodes)),
        "--n-steps",
        str(int(n_steps)),
        "--batch-size",
        str(int(batch_size)),
    ]
    run = _run_cmd(cmd, timeout_sec=timeout_sec)

    after = {p.name for p in DATA_DIR.glob("rl_eval_matrix_*.csv")} if DATA_DIR.exists() else set()
    new_files = sorted(after - before)
    run["new_csv_files"] = [str(DATA_DIR / name) for name in new_files]
    return run


@mcp.tool()
def latest_run_manifest() -> Dict[str, Any]:
    """Return the most recent run manifest from data/runs."""
    return _latest_run_manifest()


@mcp.tool()
def run_manifest(run_id: str) -> Dict[str, Any]:
    """Return run manifest for a specific run_id (for example: run_20260312_211438_80453)."""
    return _find_run_manifest(run_id)


@mcp.tool()
def tail_trace(limit: int = 20, run_id: str = "") -> Dict[str, Any]:
    """Return recent trace rows from data/tuning_trace.jsonl (optionally filtered by run_id)."""
    return _tail_trace(limit=limit, run_id=run_id)


@mcp.tool()
def summarize_trace(run_id: str = "", last_n: int = 0) -> Dict[str, Any]:
    """Summarize trace quality/performance; optionally filter by run_id or by last_n rows."""
    if not TRACE_PATH.exists():
        return {"exists": False, "reason": "trace file not found", "path": str(TRACE_PATH)}
    rows = _iter_trace_rows(run_id=run_id, last_n=max(0, int(last_n)))
    summary = _summarize_trace_rows(rows)
    return {
        "exists": True,
        "path": str(TRACE_PATH),
        "run_id_filter": str(run_id).strip(),
        "last_n": int(max(0, int(last_n))),
        "summary": summary,
    }


@mcp.tool()
def compare_runs(baseline_run_id: str, candidate_run_id: str) -> Dict[str, Any]:
    """Compare two run manifests and report cycle deltas (negative delta means candidate is better)."""
    base = _find_run_manifest(baseline_run_id)
    cand = _find_run_manifest(candidate_run_id)
    if not base.get("exists", False):
        return {"ok": False, "error": f"baseline run missing: {base.get('reason', 'unknown')}", "baseline": base}
    if not cand.get("exists", False):
        return {"ok": False, "error": f"candidate run missing: {cand.get('reason', 'unknown')}", "candidate": cand}

    b = base["manifest"]
    c = cand["manifest"]
    b_best = _to_float(b.get("best_cycles"))
    c_best = _to_float(c.get("best_cycles"))
    b_avg = _to_float(b.get("avg_valid_cycles"))
    c_avg = _to_float(c.get("avg_valid_cycles"))
    b_valid = _to_int(b.get("valid_simulated_rows"))
    c_valid = _to_int(c.get("valid_simulated_rows"))

    best_delta_pct = _cycles_delta_pct(c_best, b_best)
    avg_delta_pct = _cycles_delta_pct(c_avg, b_avg)
    candidate_better = bool(best_delta_pct is not None and best_delta_pct < 0.0)

    return {
        "ok": True,
        "baseline_run_id": str(b.get("run_id", baseline_run_id)),
        "candidate_run_id": str(c.get("run_id", candidate_run_id)),
        "baseline_path": base.get("path", ""),
        "candidate_path": cand.get("path", ""),
        "baseline_best_cycles": b_best,
        "candidate_best_cycles": c_best,
        "best_cycles_delta_pct": best_delta_pct,
        "baseline_avg_valid_cycles": b_avg,
        "candidate_avg_valid_cycles": c_avg,
        "avg_valid_cycles_delta_pct": avg_delta_pct,
        "baseline_valid_rows": b_valid,
        "candidate_valid_rows": c_valid,
        "candidate_is_better_on_best_cycles": candidate_better,
    }


@mcp.tool()
def start_inference_job(
    inference_src: str = "inference_generic.c",
    enable_dual_issue: bool = True,
    generic_cflags: str = "-O3 -fno-strict-aliasing",
    use_systolic_accel: bool = True,
    timeout_sec: int = 1800,
) -> Dict[str, Any]:
    """Start make clean + make run in background. Poll with get_job_status."""
    try:
        make_cmd = _inference_make_cmd(
            inference_src=inference_src,
            enable_dual_issue=enable_dual_issue,
            generic_cflags=generic_cflags,
            use_systolic_accel=use_systolic_accel,
        )
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    full_cmd = ["bash", "-lc", "make clean && " + " ".join(shlex.quote(x) for x in make_cmd)]
    return _start_job(job_type="inference", command=full_cmd, timeout_sec=max(1, int(timeout_sec)))


@mcp.tool()
def start_rl_daemon_job(
    timesteps: int = 8,
    workload: str = "all",
    eval_workloads: str = "all",
    eval_episodes: int = 2,
    n_steps: int = 4,
    batch_size: int = 4,
    timeout_sec: int = 7200,
) -> Dict[str, Any]:
    """Start RL daemon in background. Poll with get_job_status."""
    cmd = [
        _rl_python_bin(),
        "auto_tuner/rl_daemon.py",
        "--timesteps",
        str(int(timesteps)),
        "--workload",
        workload,
        "--eval-workloads",
        eval_workloads,
        "--eval-episodes",
        str(int(eval_episodes)),
        "--n-steps",
        str(int(n_steps)),
        "--batch-size",
        str(int(batch_size)),
    ]
    return _start_job(job_type="rl_daemon", command=cmd, timeout_sec=max(1, int(timeout_sec)))


@mcp.tool()
def start_eval_matrix_job(
    scenarios: str = "default",
    timesteps: int = 8,
    eval_episodes: int = 4,
    n_steps: int = 2,
    batch_size: int = 2,
    timeout_sec: int = 7200,
) -> Dict[str, Any]:
    """Start eval_matrix.py in background. Poll with get_job_status."""
    cmd = [
        _rl_python_bin(),
        "auto_tuner/eval_matrix.py",
        "--scenarios",
        scenarios,
        "--timesteps",
        str(int(timesteps)),
        "--eval-episodes",
        str(int(eval_episodes)),
        "--n-steps",
        str(int(n_steps)),
        "--batch-size",
        str(int(batch_size)),
    ]
    return _start_job(job_type="eval_matrix", command=cmd, timeout_sec=max(1, int(timeout_sec)))


@mcp.tool()
def get_job_status(job_id: str, log_tail_chars: int = 4000) -> Dict[str, Any]:
    """Get async job status and recent log tail."""
    _refresh_all_jobs()
    with _JOB_LOCK:
        meta = _JOB_META.get(str(job_id).strip())
        if not isinstance(meta, dict):
            return {"ok": False, "error": f"job not found: {job_id}"}
        out = dict(meta)
    out["log_tail"] = _log_tail(Path(out.get("log_path", "")), max_chars=max(200, int(log_tail_chars)))
    out.pop("started_ts", None)
    out.pop("csv_before", None)
    return {"ok": True, "job": out}


@mcp.tool()
def list_jobs(limit: int = 20) -> Dict[str, Any]:
    """List recent async jobs."""
    _refresh_all_jobs()
    with _JOB_LOCK:
        jobs = [dict(v) for v in _JOB_META.values()]
    jobs.sort(key=lambda j: str(j.get("started_at", "")), reverse=True)
    lim = max(1, int(limit))
    out = jobs[:lim]
    for j in out:
        j.pop("started_ts", None)
        j.pop("csv_before", None)
    return {"count": len(out), "jobs": out}


@mcp.tool()
def cancel_job(job_id: str) -> Dict[str, Any]:
    """Cancel a running async job."""
    _refresh_all_jobs()
    jid = str(job_id).strip()
    with _JOB_LOCK:
        meta = _JOB_META.get(jid)
        if not isinstance(meta, dict):
            return {"ok": False, "error": f"job not found: {job_id}"}
        if meta.get("status") != "running":
            out = dict(meta)
            out.pop("started_ts", None)
            out.pop("csv_before", None)
            return {"ok": True, "job": out, "note": "job already finished"}
        proc = _JOB_PROCS.get(jid)
        if proc is not None:
            try:
                proc.kill()
            except Exception:
                pass
            rc = proc.poll()
            if rc is None:
                rc = -1
        else:
            rc = -1
        _finalize_job_locked(jid, status="canceled", returncode=int(rc), error="canceled by user")
        out = dict(_JOB_META[jid])
    out["log_tail"] = _log_tail(Path(out.get("log_path", "")), max_chars=4000)
    out.pop("started_ts", None)
    out.pop("csv_before", None)
    return {"ok": True, "job": out}


if __name__ == "__main__":
    mcp.run()
