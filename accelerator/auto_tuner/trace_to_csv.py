#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, Any, List

from tuning_trace import load_traces, normalize_trace_row


TRACE_CSV_COLUMNS: List[str] = [
    "schema_version",
    "source_trace",
    "source_trace_size_bytes",
    "source_trace_mtime_epoch_sec",
    "run_id",
    "run_mode",
    "campaign_id",
    "campaign_stage",
    "campaign_targeted",
    "campaign_target_reason",
    "campaign_target_bucket",
    "campaign_target_weight",
    "target_key",
    "episode_id",
    "timestamp",
    "run_date_utc",
    "workload_tag",
    "shape_signature",
    "M",
    "N",
    "K",
    "activation",
    "op_type_id",
    "batch_size",
    "seq_len",
    "channels",
    "kernel_h",
    "kernel_w",
    "sparsity_pct",
    "sparsity_bucket",
    "mode1_candidate",
    "tile_m",
    "tile_n",
    "burst_size",
    "prefetch_depth",
    "tile_b",
    "hardware_dataflow_mode",
    "executed_hardware_dataflow_mode",
    "proposed_hardware_dataflow_mode",
    "mode_name",
    "executed_mode_name",
    "proposed_mode_name",
    "mode_fallback",
    "mode_fallback_reason",
    "executed_mode_source",
    "export_mode_provenance",
    "export_mode_fallback_reason",
    "simulated",
    "correctness_passed",
    "is_valid",
    "cycles",
    "stalls",
    "reward",
    "reward_raw",
    "scratchpad_avail",
    "scratch_required_bytes",
    "scratch_limit_bytes",
    "scratch_util_pct",
    "macs",
    "dma_bytes_est",
    "dense_equiv_macs_per_cycle",
    "pe_util_est",
]


def _default_workspace() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_trace_paths(workspace: str) -> List[str]:
    traces_dir = Path(workspace) / "data" / "traces"
    if traces_dir.is_dir():
        traces = sorted(traces_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        if traces:
            return [str(traces[0])]
    legacy = Path(workspace) / "data" / "tuning_trace.jsonl"
    return [str(legacy)]


def _all_trace_paths(workspace: str, trace_glob: str = "*.jsonl") -> List[str]:
    traces_dir = Path(workspace) / "data" / "traces"
    out: List[str] = []
    if traces_dir.is_dir():
        out.extend(str(p.resolve()) for p in sorted(traces_dir.glob(trace_glob)))
    legacy = Path(workspace) / "data" / "tuning_trace.jsonl"
    if legacy.exists():
        out.append(str(legacy.resolve()))
    if not out:
        out.append(str(legacy.resolve()))
    return out


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _sort_key(row: Dict[str, Any]) -> tuple[str, int, float]:
    run_id = str(row.get("run_id", ""))
    ep = _to_int(row.get("episode_id", 0), 0)
    try:
        ts = float(row.get("timestamp", 0.0))
    except Exception:
        ts = 0.0
    return run_id, ep, ts


def _run_date_utc_from_epoch(timestamp_value: Any) -> str:
    try:
        ts = float(timestamp_value)
    except Exception:
        ts = 0.0
    if ts <= 0.0:
        return "1970-01-01"
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def extract_rows(trace_paths: List[str], valid_only: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for trace_path in trace_paths:
        trace_p = Path(trace_path)
        size_bytes = int(trace_p.stat().st_size) if trace_p.exists() else -1
        mtime = float(trace_p.stat().st_mtime) if trace_p.exists() else -1.0
        for rec in load_traces(trace_path):
            row = normalize_trace_row(rec)
            row["source_trace"] = str(trace_path)
            row["source_trace_size_bytes"] = int(size_bytes)
            row["source_trace_mtime_epoch_sec"] = float(mtime)
            row["run_date_utc"] = _run_date_utc_from_epoch(row.get("timestamp"))
            if valid_only and int(row.get("is_valid", 0)) != 1:
                continue
            rows.append(row)
    rows.sort(key=_sort_key)
    return rows


def write_csv(rows: List[Dict[str, Any]], output_csv: str) -> None:
    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TRACE_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Normalize RL trace JSONL and export deterministic CSV.")
    ap.add_argument("--workspace", type=str, default=_default_workspace(),
                    help="Accelera workspace root.")
    ap.add_argument("--trace", action="append", default=[],
                    help="Trace JSONL path (repeatable). Defaults to latest trace in data/traces.")
    ap.add_argument("--all-traces", action="store_true",
                    help="Use all traces from data/traces/*.jsonl plus legacy data/tuning_trace.jsonl.")
    ap.add_argument("--trace-glob", type=str, default="*.jsonl",
                    help="Glob pattern for --all-traces in data/traces/ (default: *.jsonl).")
    ap.add_argument("--output-csv", type=str, default="",
                    help="Output CSV path. Defaults to data/trace_dataset_latest.csv.")
    ap.add_argument("--valid-only", action="store_true",
                    help="Keep only valid simulated rows (is_valid==1).")
    args = ap.parse_args(argv)

    if args.trace:
        trace_paths = [str(Path(p).resolve()) for p in args.trace]
    elif args.all_traces:
        trace_paths = _all_trace_paths(args.workspace, trace_glob=str(args.trace_glob))
    else:
        trace_paths = _default_trace_paths(args.workspace)
    if not args.output_csv:
        out = Path(args.workspace) / "data" / "trace_dataset_latest.csv"
        output_csv = str(out)
    else:
        output_csv = str(Path(args.output_csv).resolve())

    rows = extract_rows(trace_paths=trace_paths, valid_only=bool(args.valid_only))
    write_csv(rows, output_csv)

    valid_count = sum(1 for r in rows if int(r.get("is_valid", 0)) == 1)
    print(
        f"[TraceToCSV] traces={len(trace_paths)} rows={len(rows)} valid_rows={valid_count} "
        f"valid_only={int(bool(args.valid_only))} output={output_csv}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
