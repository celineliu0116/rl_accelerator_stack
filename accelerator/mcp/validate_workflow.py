#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
MCP_SERVER_PATH = REPO_ROOT / "mcp" / "accelera_mcp_server.py"


def _load_server_module():
    spec = importlib.util.spec_from_file_location("accelera_mcp_server", MCP_SERVER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {MCP_SERVER_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _compact(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            if k in {"stdout_tail", "stderr_tail", "log_tail"} and isinstance(v, str):
                out[k] = v[-900:]
            elif k == "rows" and isinstance(v, list):
                out[k] = v[-3:]
            else:
                out[k] = _compact(v)
        return out
    if isinstance(value, list):
        return [_compact(x) for x in value]
    return value


def _step_ok(res: Any) -> bool:
    if not isinstance(res, dict):
        return True
    return bool(res.get("ok", True))


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate Accelera MCP workflow top-to-bottom.")
    ap.add_argument("--rl-python", type=str, default=os.environ.get("ACCELERA_RL_PYTHON", ""))
    ap.add_argument("--timesteps", type=int, default=1)
    ap.add_argument("--eval-episodes", type=int, default=1)
    ap.add_argument("--n-steps", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--eval-scenarios", type=str, default="default")
    ap.add_argument("--output", type=str, default="")
    args = ap.parse_args()

    if args.rl_python:
        os.environ["ACCELERA_RL_PYTHON"] = args.rl_python

    mod = _load_server_module()

    report: Dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "env": {"ACCELERA_RL_PYTHON": os.environ.get("ACCELERA_RL_PYTHON", "")},
        "steps": [],
    }

    def run_step(name: str, fn):
        t0 = time.time()
        rec: Dict[str, Any] = {"name": name}
        try:
            result = fn()
            rec["ok"] = _step_ok(result)
            rec["result"] = _compact(result)
        except Exception as exc:
            rec["ok"] = False
            rec["error"] = f"{type(exc).__name__}: {exc}"
        rec["elapsed_sec"] = round(time.time() - t0, 3)
        report["steps"].append(rec)
        print(f"[mcp-validate] {name}: ok={rec['ok']} elapsed={rec['elapsed_sec']}s")
        return rec

    run_step("project_status", lambda: mod.project_status())
    run_step("export_mnist_reference", lambda: mod.export_mnist_reference(timeout_sec=240))
    run_step("run_inference", lambda: mod.run_inference(timeout_sec=1500))
    run_step(
        "run_rl_daemon_sync",
        lambda: mod.run_rl_daemon(
            timesteps=args.timesteps,
            eval_episodes=args.eval_episodes,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            timeout_sec=1200,
        ),
    )
    run_step(
        "run_eval_matrix_sync",
        lambda: mod.run_eval_matrix(
            scenarios=args.eval_scenarios,
            timesteps=args.timesteps,
            eval_episodes=args.eval_episodes,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            timeout_sec=1800,
        ),
    )

    latest = run_step("latest_run_manifest", lambda: mod.latest_run_manifest())
    latest_run_id = ""
    if isinstance(latest.get("result"), dict):
        latest_run_id = str(latest["result"].get("manifest", {}).get("run_id", ""))
    if latest_run_id:
        run_step("run_manifest_latest", lambda: mod.run_manifest(latest_run_id))
    run_step("tail_trace", lambda: mod.tail_trace(limit=5, run_id=latest_run_id))
    run_step("summarize_trace_all", lambda: mod.summarize_trace(last_n=200))

    run_files = sorted((REPO_ROOT / "data" / "runs").glob("run_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(run_files) >= 2:
        r0 = json.loads(run_files[0].read_text(encoding="utf-8")).get("run_id", run_files[0].stem)
        r1 = json.loads(run_files[1].read_text(encoding="utf-8")).get("run_id", run_files[1].stem)
        run_step("compare_runs_recent", lambda: mod.compare_runs(str(r1), str(r0)))
    else:
        report["steps"].append({"name": "compare_runs_recent", "ok": False, "error": "not enough run manifests"})

    job = run_step(
        "start_rl_daemon_job",
        lambda: mod.start_rl_daemon_job(
            timesteps=args.timesteps,
            eval_episodes=args.eval_episodes,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            timeout_sec=180,
        ),
    )
    job_id = ""
    if isinstance(job.get("result"), dict):
        job_id = str(job["result"].get("job", {}).get("job_id", ""))
    if job_id:
        run_step("get_job_status", lambda: mod.get_job_status(job_id, log_tail_chars=1000))
        run_step("cancel_job", lambda: mod.cancel_job(job_id))
    run_step("list_jobs", lambda: mod.list_jobs(limit=5))

    report["finished_at"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    report["overall_ok"] = all(step.get("ok", False) for step in report["steps"])

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = REPO_ROOT / "data" / f"mcp_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[mcp-validate] report={out_path}")
    print(f"[mcp-validate] overall_ok={report['overall_ok']}")
    return 0 if report["overall_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
