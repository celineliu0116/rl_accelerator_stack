#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_CONFIG = REPO_ROOT / ".github" / "mcp_regression_baseline.json"
DEFAULT_REPORT = REPO_ROOT / "data" / "mcp_validation_report_ci.json"
RUNS_DIR = REPO_ROOT / "data" / "runs"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_step(report: Dict[str, Any], name: str) -> Dict[str, Any] | None:
    for step in report.get("steps", []):
        if str(step.get("name", "")) == name:
            return step
    return None


def _find_manifest(run_id: str) -> Dict[str, Any]:
    rid = str(run_id).strip()
    if rid.endswith(".json"):
        rid = rid[:-5]
    candidate = RUNS_DIR / f"{rid}.json"
    if candidate.exists():
        return _load_json(candidate)
    for p in RUNS_DIR.glob("run_*.json"):
        payload = _load_json(p)
        if str(payload.get("run_id", "")) == rid:
            return payload
    raise FileNotFoundError(f"run manifest not found for run_id={run_id}")


def _safe_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def _metric_bundle(manifest: Dict[str, Any]) -> Dict[str, Any]:
    valid_rows = int(manifest.get("valid_simulated_rows", 0))
    total_rows = int(manifest.get("total_rows", 0))
    valid_ratio = (float(valid_rows) / float(total_rows)) if total_rows > 0 else None
    return {
        "run_id": str(manifest.get("run_id", "")),
        "best_cycles": _safe_float(manifest.get("best_cycles")),
        "avg_valid_cycles": _safe_float(manifest.get("avg_valid_cycles")),
        "valid_simulated_rows": valid_rows,
        "total_rows": total_rows,
        "valid_ratio": valid_ratio,
        "status": str(manifest.get("status", "")),
    }


def _delta_pct(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline <= 0:
        return None
    return ((candidate - baseline) / baseline) * 100.0


def _warn(msg: str) -> None:
    print(f"WARNING: {msg}")
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::warning::{msg}")


def _error(msg: str) -> None:
    print(f"ERROR: {msg}")
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::error::{msg}")


def _write_summary(lines: list[str]) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if not summary_path:
        return
    try:
        with open(summary_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except Exception:
        pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Gate MCP validation by baseline correctness/performance.")
    ap.add_argument("--report", type=str, default=str(DEFAULT_REPORT))
    ap.add_argument("--baseline-config", type=str, default=str(DEFAULT_BASELINE_CONFIG))
    ap.add_argument("--candidate-run-id", type=str, default="")
    ap.add_argument("--baseline-run-id", type=str, default="")
    args = ap.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        _error(f"validation report not found: {report_path}")
        return 1
    report = _load_json(report_path)

    cfg_path = Path(args.baseline_config)
    cfg: Dict[str, Any] = {}
    if cfg_path.exists():
        cfg = _load_json(cfg_path)

    baseline_run_id = args.baseline_run_id or str(cfg.get("baseline_run_id", "")).strip()
    if not baseline_run_id:
        _error("baseline_run_id is required (pass --baseline-run-id or set in baseline config)")
        return 1

    candidate_run_id = args.candidate_run_id.strip()
    if not candidate_run_id:
        latest = _find_step(report, "latest_run_manifest")
        if isinstance(latest, dict):
            candidate_run_id = str(
                latest.get("result", {}).get("manifest", {}).get("run_id", "")
            ).strip()
    if not candidate_run_id:
        _error("could not determine candidate run_id from report; pass --candidate-run-id")
        return 1

    try:
        baseline_manifest = _find_manifest(baseline_run_id)
        candidate_manifest = _find_manifest(candidate_run_id)
    except FileNotFoundError as exc:
        _error(str(exc))
        return 1

    baseline = _metric_bundle(baseline_manifest)
    candidate = _metric_bundle(candidate_manifest)

    min_valid_ratio = float(cfg.get("min_valid_ratio", 0.25))
    max_valid_ratio_drop_abs = float(cfg.get("max_valid_ratio_drop_abs", 0.20))
    warn_best_cycles_regression_pct = float(cfg.get("warn_best_cycles_regression_pct", 10.0))
    warn_avg_valid_cycles_regression_pct = float(cfg.get("warn_avg_valid_cycles_regression_pct", 10.0))

    failures: list[str] = []
    warnings: list[str] = []

    if not bool(report.get("overall_ok", False)):
        failures.append("validation report overall_ok is false")

    if candidate["status"] != "completed":
        failures.append(f"candidate run status is not completed (status={candidate['status']})")

    if candidate["best_cycles"] is None or candidate["best_cycles"] <= 0:
        failures.append(f"candidate best_cycles is invalid ({candidate['best_cycles']})")

    if candidate["avg_valid_cycles"] is None or candidate["avg_valid_cycles"] <= 0:
        failures.append(f"candidate avg_valid_cycles is invalid ({candidate['avg_valid_cycles']})")

    if candidate["valid_simulated_rows"] <= 0:
        failures.append("candidate valid_simulated_rows must be > 0")

    cvr = candidate["valid_ratio"]
    bvr = baseline["valid_ratio"]
    if cvr is None:
        failures.append("candidate valid_ratio is unavailable")
    else:
        if cvr < min_valid_ratio:
            failures.append(
                f"candidate valid_ratio {cvr:.3f} below minimum {min_valid_ratio:.3f}"
            )
        if bvr is not None and cvr < (bvr - max_valid_ratio_drop_abs):
            failures.append(
                f"candidate valid_ratio dropped too much vs baseline "
                f"({cvr:.3f} < {bvr:.3f} - {max_valid_ratio_drop_abs:.3f})"
            )

    best_delta_pct = _delta_pct(candidate["best_cycles"], baseline["best_cycles"])
    avg_delta_pct = _delta_pct(candidate["avg_valid_cycles"], baseline["avg_valid_cycles"])

    if best_delta_pct is not None and best_delta_pct > warn_best_cycles_regression_pct:
        warnings.append(
            f"best_cycles regressed by {best_delta_pct:.2f}% "
            f"(threshold {warn_best_cycles_regression_pct:.2f}%)"
        )
    if avg_delta_pct is not None and avg_delta_pct > warn_avg_valid_cycles_regression_pct:
        warnings.append(
            f"avg_valid_cycles regressed by {avg_delta_pct:.2f}% "
            f"(threshold {warn_avg_valid_cycles_regression_pct:.2f}%)"
        )

    print("MCP Regression Gate Summary")
    print(f"  baseline_run_id={baseline['run_id']}")
    print(f"  candidate_run_id={candidate['run_id']}")
    print(f"  baseline best={baseline['best_cycles']} avg={baseline['avg_valid_cycles']} valid_ratio={baseline['valid_ratio']}")
    print(f"  candidate best={candidate['best_cycles']} avg={candidate['avg_valid_cycles']} valid_ratio={candidate['valid_ratio']}")
    print(f"  best_delta_pct={best_delta_pct}")
    print(f"  avg_delta_pct={avg_delta_pct}")

    summary_lines = [
        "## MCP Regression Gate",
        f"- Baseline run: `{baseline['run_id']}`",
        f"- Candidate run: `{candidate['run_id']}`",
        f"- Baseline metrics: best `{baseline['best_cycles']}`, avg `{baseline['avg_valid_cycles']}`, valid_ratio `{baseline['valid_ratio']}`",
        f"- Candidate metrics: best `{candidate['best_cycles']}`, avg `{candidate['avg_valid_cycles']}`, valid_ratio `{candidate['valid_ratio']}`",
        f"- best_cycles delta: `{best_delta_pct}`",
        f"- avg_valid_cycles delta: `{avg_delta_pct}`",
    ]

    for msg in warnings:
        _warn(msg)
        summary_lines.append(f"- WARNING: {msg}")

    if failures:
        for msg in failures:
            _error(msg)
            summary_lines.append(f"- ERROR: {msg}")
        _write_summary(summary_lines)
        return 1

    summary_lines.append("- Result: PASS")
    _write_summary(summary_lines)
    print("Regression gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
