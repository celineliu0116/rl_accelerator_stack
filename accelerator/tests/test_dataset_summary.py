#!/usr/bin/env python3
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTO_TUNER_DIR = REPO_ROOT / "auto_tuner"
MODULE_PATH = AUTO_TUNER_DIR / "dataset_summary.py"


def _load_module():
    if str(AUTO_TUNER_DIR) not in sys.path:
        sys.path.insert(0, str(AUTO_TUNER_DIR))
    spec = importlib.util.spec_from_file_location("accelera_dataset_summary", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class DatasetSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load_module()

    def test_summary_reports_counts_and_diversity(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            csv_path = td_path / "trace.csv"
            out_json = td_path / "summary.json"
            cols = [
                "run_id", "workload_tag", "shape_signature", "M", "N", "K", "activation",
                "sparsity_bucket", "is_valid", "cycles", "executed_hardware_dataflow_mode",
                "tile_m", "tile_n", "burst_size", "prefetch_depth", "tile_b",
            ]
            rows = [
                {
                    "run_id": "r1", "workload_tag": "gemm", "shape_signature": "64x128x128", "M": 64, "N": 128, "K": 128,
                    "activation": 0, "sparsity_bucket": 0, "is_valid": 1, "cycles": 1000,
                    "executed_hardware_dataflow_mode": 0, "tile_m": 8, "tile_n": 16, "burst_size": 32, "prefetch_depth": 1, "tile_b": 1,
                },
                {
                    "run_id": "r1", "workload_tag": "convolution", "shape_signature": "128x256x128", "M": 128, "N": 256, "K": 128,
                    "activation": 1, "sparsity_bucket": 2, "is_valid": 1, "cycles": 900,
                    "executed_hardware_dataflow_mode": 1, "tile_m": 12, "tile_n": 24, "burst_size": 64, "prefetch_depth": 2, "tile_b": 2,
                },
                {
                    "run_id": "r1", "workload_tag": "convolution", "shape_signature": "128x256x128", "M": 128, "N": 256, "K": 128,
                    "activation": 1, "sparsity_bucket": 2, "is_valid": 0, "cycles": -1,
                    "executed_hardware_dataflow_mode": 1, "tile_m": 12, "tile_n": 24, "burst_size": 64, "prefetch_depth": 2, "tile_b": 2,
                },
            ]
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=cols)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)

            rc = self.mod.main([
                "--input-csv", str(csv_path),
                "--output-json", str(out_json),
            ])
            self.assertEqual(rc, 0)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            src = payload["sources"]["csv"]
            self.assertEqual(int(src["row_count"]), 3)
            self.assertEqual(int(src["valid_row_count"]), 2)
            self.assertGreaterEqual(int(src["unique_shape_count"]), 2)
            self.assertIn("tile_b", src["knob_diversity"])


if __name__ == "__main__":
    unittest.main()
