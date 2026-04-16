#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from trace_to_csv import _all_trace_paths, _default_trace_paths, extract_rows


def _default_workspace() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _read_parquet(path: Path) -> List[Dict[str, Any]]:
    try:
        import pyarrow.dataset as ds
    except Exception as exc:
        raise RuntimeError(
            "pyarrow is required to read Parquet datasets. Install with `pip install pyarrow`."
        ) from exc
    dataset = ds.dataset(str(path), format="parquet", partitioning="hive")
    table = dataset.to_table()
    return [dict(row) for row in table.to_pylist()]


def _size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += int(child.stat().st_size)
    return int(total)


def _series_count(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    ctr: Counter[str] = Counter()
    for row in rows:
        val = str(row.get(key, ""))
        ctr[val] += 1
    return {k: int(v) for k, v in sorted(ctr.items(), key=lambda kv: (-kv[1], kv[0]))}


def _knob_diversity(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    knobs = [
        "executed_hardware_dataflow_mode",
        "tile_m",
        "tile_n",
        "burst_size",
        "prefetch_depth",
        "tile_b",
    ]
    out: Dict[str, Any] = {}
    for knob in knobs:
        values = sorted({str(row.get(knob, "")) for row in rows})
        out[knob] = {
            "unique_count": int(len(values)),
            "values": values,
        }
    return out


def _key_from_row(row: Dict[str, Any]) -> str:
    m = _to_int(row.get("M", -1), -1)
    n = _to_int(row.get("N", -1), -1)
    k = _to_int(row.get("K", -1), -1)
    act = _to_int(row.get("activation", 0), 0)
    workload = str(row.get("workload_tag", ""))
    sp = _to_int(row.get("sparsity_bucket", 0), 0)
    return f"{m}x{n}x{k}_act{act}_{workload}_sp{sp}"


def _coverage(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_key: Counter[str] = Counter()
    for row in rows:
        by_key[_key_from_row(row)] += 1
    if not by_key:
        return {
            "key_count": 0,
            "min_rows_per_key": 0,
            "max_rows_per_key": 0,
            "mean_rows_per_key": 0.0,
            "low_coverage_keys_le_2": [],
        }
    counts = list(by_key.values())
    low_keys = [k for k, v in by_key.items() if v <= 2]
    return {
        "key_count": int(len(by_key)),
        "min_rows_per_key": int(min(counts)),
        "max_rows_per_key": int(max(counts)),
        "mean_rows_per_key": float(sum(counts) / max(1, len(counts))),
        "low_coverage_keys_le_2": sorted(low_keys)[:50],
    }


def _summarize_rows(rows: List[Dict[str, Any]], source_label: str) -> Dict[str, Any]:
    row_count = int(len(rows))
    valid_rows = sum(1 for row in rows if _to_int(row.get("is_valid", 0), 0) == 1)
    invalid_rows = max(0, row_count - valid_rows)
    shape_set = {str(row.get("shape_signature", "")) for row in rows if str(row.get("shape_signature", "")).strip()}
    run_ids = {str(row.get("run_id", "")) for row in rows if str(row.get("run_id", "")).strip()}
    mode_counts = _series_count(rows, "executed_hardware_dataflow_mode")
    if not mode_counts:
        mode_counts = _series_count(rows, "hardware_dataflow_mode")
    mean_cycles_valid = -1.0
    valid_cycles = [
        _to_float(row.get("cycles", -1), -1.0)
        for row in rows
        if _to_int(row.get("is_valid", 0), 0) == 1 and _to_float(row.get("cycles", -1), -1.0) > 0
    ]
    if valid_cycles:
        mean_cycles_valid = float(sum(valid_cycles) / len(valid_cycles))
    return {
        "source": source_label,
        "row_count": row_count,
        "valid_row_count": int(valid_rows),
        "invalid_row_count": int(invalid_rows),
        "invalid_rate": float(invalid_rows / max(1, row_count)),
        "unique_run_id_count": int(len(run_ids)),
        "unique_shape_count": int(len(shape_set)),
        "workload_distribution": _series_count(rows, "workload_tag"),
        "sparsity_bucket_distribution": _series_count(rows, "sparsity_bucket"),
        "mode_distribution": mode_counts,
        "knob_diversity": _knob_diversity(rows),
        "coverage": _coverage(rows),
        "mean_cycles_valid": float(mean_cycles_valid),
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize trace datasets for coverage and quality.")
    ap.add_argument("--workspace", type=str, default=_default_workspace(), help="Accelera workspace root.")
    ap.add_argument("--trace", action="append", default=[], help="Trace JSONL path (repeatable).")
    ap.add_argument("--all-traces", action="store_true", help="Use all traces under data/traces plus legacy trace.")
    ap.add_argument("--trace-glob", type=str, default="*.jsonl", help="Glob for --all-traces.")
    ap.add_argument("--input-csv", type=str, default="", help="Summarize this normalized CSV dataset.")
    ap.add_argument("--input-parquet", type=str, default="", help="Summarize this Parquet dataset directory or file.")
    ap.add_argument("--valid-only", action="store_true", help="Applied only for trace JSONL source path mode.")
    ap.add_argument("--output-json", type=str, default="", help="Output summary JSON path.")
    args = ap.parse_args(argv)

    workspace = str(Path(args.workspace).resolve())
    outputs: Dict[str, Any] = {
        "summary_schema_version": 1,
        "workspace": workspace,
        "sources": {},
    }

    if args.input_csv:
        csv_path = Path(args.input_csv).resolve()
        rows = _read_csv(csv_path)
        outputs["sources"]["csv"] = _summarize_rows(rows, source_label=str(csv_path))
        outputs["sources"]["csv"]["storage_size_bytes"] = _size_bytes(csv_path)

    if args.input_parquet:
        pq_path = Path(args.input_parquet).resolve()
        try:
            rows = _read_parquet(pq_path)
            outputs["sources"]["parquet"] = _summarize_rows(rows, source_label=str(pq_path))
            outputs["sources"]["parquet"]["storage_size_bytes"] = _size_bytes(pq_path)
        except Exception as exc:
            outputs["sources"]["parquet"] = {
                "source": str(pq_path),
                "error": str(exc),
                "storage_size_bytes": _size_bytes(pq_path),
            }

    if args.trace or args.all_traces or (not args.input_csv and not args.input_parquet):
        if args.trace:
            trace_paths = [str(Path(p).resolve()) for p in args.trace]
        elif args.all_traces:
            trace_paths = _all_trace_paths(workspace, trace_glob=str(args.trace_glob))
        else:
            trace_paths = _default_trace_paths(workspace)
        rows = extract_rows(trace_paths=trace_paths, valid_only=bool(args.valid_only))
        outputs["sources"]["jsonl_trace"] = _summarize_rows(rows, source_label="trace_jsonl")
        outputs["sources"]["jsonl_trace"]["trace_paths"] = trace_paths
        outputs["sources"]["jsonl_trace"]["storage_size_bytes"] = int(sum(_size_bytes(Path(p)) for p in trace_paths))

    if "csv" in outputs["sources"] and "parquet" in outputs["sources"] and "error" not in outputs["sources"]["parquet"]:
        csv_size = int(outputs["sources"]["csv"].get("storage_size_bytes", 0))
        pq_size = int(outputs["sources"]["parquet"].get("storage_size_bytes", 0))
        ratio = float(pq_size / max(1, csv_size))
        outputs["storage_comparison"] = {
            "csv_size_bytes": int(csv_size),
            "parquet_size_bytes": int(pq_size),
            "parquet_over_csv_ratio": float(ratio),
            "estimated_reduction_pct": float((1.0 - ratio) * 100.0),
        }

    if args.output_json:
        out_path = Path(args.output_json).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(outputs, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[DatasetSummary] output={out_path}")
    else:
        print(json.dumps(outputs, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
