#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from trace_to_csv import _all_trace_paths, _default_trace_paths, extract_rows


def _default_workspace() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_output_dir(workspace: str) -> str:
    return str((Path(workspace) / "data" / "parquet" / "trace_dataset_v1").resolve())


def _default_meta_path(output_dir: str) -> str:
    return str((Path(output_dir) / "_dataset_meta.json").resolve())


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _parse_partition_cols(raw: str) -> List[str]:
    cols = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not cols:
        return ["workload_tag", "sparsity_bucket", "run_date_utc"]
    return cols


def _prepare_rows(rows: List[Dict[str, Any]], partition_cols: List[str], dataset_name: str, run_id: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        rec = dict(row)
        rec.setdefault("dataset_name", str(dataset_name))
        rec.setdefault("parquet_export_run_id", str(run_id))
        for col in partition_cols:
            if col not in rec:
                if col in {"sparsity_bucket", "schema_version", "is_valid"}:
                    rec[col] = _to_int(rec.get(col, 0), 0)
                else:
                    rec[col] = "unknown"
        out.append(rec)
    return out


def _dataset_sizes(base_dir: Path) -> Dict[str, int]:
    parquet_files = list(base_dir.rglob("*.parquet"))
    total_bytes = 0
    for p in parquet_files:
        try:
            total_bytes += int(p.stat().st_size)
        except Exception:
            continue
    return {
        "parquet_file_count": int(len(parquet_files)),
        "parquet_total_bytes": int(total_bytes),
    }


def _write_metadata(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Normalize RL trace JSONL and export partitioned Parquet dataset.")
    ap.add_argument("--workspace", type=str, default=_default_workspace(), help="Accelera workspace root.")
    ap.add_argument("--trace", action="append", default=[], help="Trace JSONL path (repeatable).")
    ap.add_argument("--all-traces", action="store_true", help="Use all traces in data/traces plus legacy trace.")
    ap.add_argument("--trace-glob", type=str, default="*.jsonl", help="Glob used with --all-traces.")
    ap.add_argument("--valid-only", action="store_true", help="Keep only valid rows (is_valid==1).")
    ap.add_argument("--max-rows", type=int, default=0, help="If >0, keep only latest N rows after sorting.")
    ap.add_argument("--output-dir", type=str, default="", help="Output Parquet dataset directory.")
    ap.add_argument("--metadata-json", type=str, default="", help="Metadata JSON output path.")
    ap.add_argument("--dataset-name", type=str, default="trace_dataset_v1", help="Logical dataset name for metadata.")
    ap.add_argument("--partition-cols", type=str, default="workload_tag,sparsity_bucket,run_date_utc", help="Comma-separated partition columns.")
    ap.add_argument("--compression", type=str, default="zstd", help="Parquet compression codec (zstd|snappy|gzip|brotli|none).")
    ap.add_argument("--append", action="store_true", help="Append data to existing output directory.")
    args = ap.parse_args(argv)

    try:
        import pyarrow as pa
        import pyarrow.dataset as ds
    except Exception as exc:
        raise RuntimeError(
            "pyarrow is required for Parquet export. Install with `pip install pyarrow` in your project venv."
        ) from exc

    workspace = str(Path(args.workspace).resolve())
    output_dir = str(Path(args.output_dir).resolve()) if args.output_dir else _default_output_dir(workspace)
    metadata_path = str(Path(args.metadata_json).resolve()) if args.metadata_json else _default_meta_path(output_dir)

    if args.trace:
        trace_paths = [str(Path(p).resolve()) for p in args.trace]
    elif args.all_traces:
        trace_paths = _all_trace_paths(workspace, trace_glob=str(args.trace_glob))
    else:
        trace_paths = _default_trace_paths(workspace)

    rows = extract_rows(trace_paths=trace_paths, valid_only=bool(args.valid_only))
    if int(args.max_rows) > 0:
        rows = rows[-int(args.max_rows):]

    partition_cols = _parse_partition_cols(args.partition_cols)
    export_run_id = time.strftime("pq_%Y%m%d_%H%M%S", time.localtime())
    prepared_rows = _prepare_rows(
        rows,
        partition_cols=partition_cols,
        dataset_name=str(args.dataset_name),
        run_id=export_run_id,
    )

    out_dir = Path(output_dir)
    if out_dir.exists() and not args.append:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written_rows = int(len(prepared_rows))
    if prepared_rows:
        table = pa.Table.from_pylist(prepared_rows)
        basename_template = f"part-{int(time.time())}-{{i}}.parquet"
        ds.write_dataset(
            data=table,
            base_dir=str(out_dir),
            format="parquet",
            partitioning=partition_cols,
            partitioning_flavor="hive",
            existing_data_behavior="overwrite_or_ignore",
            basename_template=basename_template,
            file_options=ds.ParquetFileFormat().make_write_options(
                compression=None if str(args.compression).lower() == "none" else str(args.compression).lower()
            ),
        )

    aggregate_rows = 0
    aggregate_valid_rows = 0
    aggregate_schema_versions: List[int] = []
    try:
        dataset = ds.dataset(str(out_dir), format="parquet", partitioning="hive")
        aggregate_rows = int(dataset.count_rows())
        tbl = dataset.to_table(columns=["is_valid", "schema_version"])
        valid_col = tbl.column("is_valid") if "is_valid" in tbl.column_names else None
        schema_col = tbl.column("schema_version") if "schema_version" in tbl.column_names else None
        if valid_col is not None:
            for val in valid_col.to_pylist():
                aggregate_valid_rows += int(_to_int(val, 0) == 1)
        if schema_col is not None:
            aggregate_schema_versions = sorted({int(_to_int(v, 0)) for v in schema_col.to_pylist()})
    except Exception:
        aggregate_rows = int(written_rows)
        aggregate_valid_rows = int(sum(1 for r in prepared_rows if _to_int(r.get("is_valid"), 0) == 1))
        aggregate_schema_versions = sorted({int(_to_int(r.get("schema_version"), 0)) for r in prepared_rows})

    size_info = _dataset_sizes(out_dir)
    payload: Dict[str, Any] = {
        "dataset_schema_version": 1,
        "trace_schema_versions": aggregate_schema_versions,
        "dataset_name": str(args.dataset_name),
        "workspace": workspace,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "append_mode": int(bool(args.append)),
        "partition_cols": partition_cols,
        "compression": str(args.compression).lower(),
        "source_trace_count": int(len(trace_paths)),
        "source_traces": trace_paths,
        "last_export_run_id": export_run_id,
        "last_export_rows": int(written_rows),
        "aggregate_rows": int(aggregate_rows),
        "aggregate_valid_rows": int(aggregate_valid_rows),
        "aggregate_invalid_rate": float(1.0 - (aggregate_valid_rows / max(1, aggregate_rows))),
        "output_dir": str(out_dir.resolve()),
        "metadata_json": str(Path(metadata_path).resolve()),
        **size_info,
    }
    _write_metadata(Path(metadata_path), payload)

    print(
        f"[TraceToParquet] traces={len(trace_paths)} export_rows={written_rows} "
        f"aggregate_rows={aggregate_rows} valid_rows={aggregate_valid_rows} "
        f"parquet_files={size_info['parquet_file_count']} output={out_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
