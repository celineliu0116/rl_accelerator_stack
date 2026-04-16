#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTO_TUNER_DIR = REPO_ROOT / "auto_tuner"
MODULE_PATH = AUTO_TUNER_DIR / "trace_to_parquet.py"


def _load_module():
    if str(AUTO_TUNER_DIR) not in sys.path:
        sys.path.insert(0, str(AUTO_TUNER_DIR))
    spec = importlib.util.spec_from_file_location("accelera_trace_to_parquet", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TraceToParquetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load_module()

    def test_export_writes_metadata(self) -> None:
        try:
            import pyarrow  # noqa: F401
        except Exception:
            self.skipTest("pyarrow not installed")

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            trace = td_path / "trace.jsonl"
            trace.write_text(
                "\n".join(
                    [
                        json.dumps({"run_id": "r1", "episode_id": 1, "workload_tag": "gemm", "M": 64, "N": 64, "K": 64, "sparsity_bucket": 0, "simulated": 1, "correctness_passed": 1, "cycles": 1000, "hardware_dataflow_mode": 0}),
                        json.dumps({"run_id": "r1", "episode_id": 2, "workload_tag": "gemm", "M": 64, "N": 64, "K": 64, "sparsity_bucket": 0, "simulated": 1, "correctness_passed": 0, "cycles": -1, "hardware_dataflow_mode": 1}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            out_dir = td_path / "pq"
            meta = out_dir / "_dataset_meta.json"
            rc = self.mod.main([
                "--trace", str(trace),
                "--output-dir", str(out_dir),
                "--metadata-json", str(meta),
                "--dataset-name", "unit_test",
            ])
            self.assertEqual(rc, 0)
            self.assertTrue(meta.exists())
            payload = json.loads(meta.read_text(encoding="utf-8"))
            self.assertEqual(int(payload.get("aggregate_rows", -1)), 2)
            self.assertIn("workload_tag", payload.get("partition_cols", []))
            parquet_files = list(out_dir.rglob("*.parquet"))
            self.assertTrue(parquet_files)


if __name__ == "__main__":
    unittest.main()
