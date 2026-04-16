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
MODULE_PATH = AUTO_TUNER_DIR / "build_targeted_campaign.py"


def _load_module():
    if str(AUTO_TUNER_DIR) not in sys.path:
        sys.path.insert(0, str(AUTO_TUNER_DIR))
    spec = importlib.util.spec_from_file_location("accelera_build_targeted_campaign", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class BuildTargetedCampaignTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load_module()

    def test_campaign_contains_regret_and_coverage_targets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            regret_csv = td_path / "regret.csv"
            trace_csv = td_path / "trace.csv"
            out_json = td_path / "campaign.json"

            regret_cols = [
                "workload_tag", "shape_signature", "sparsity_bucket",
                "topk_regret_pct", "within_regret_pct",
                "topk_matches", "within_matches",
                "topk_cycles", "within_cycles",
            ]
            regret_rows = [
                {
                    "workload_tag": "cnn_train_fwd", "shape_signature": "256x64x288", "sparsity_bucket": 0,
                    "topk_regret_pct": 55.0, "within_regret_pct": 10.0,
                    "topk_matches": 0, "within_matches": 1,
                    "topk_cycles": 1200, "within_cycles": 900,
                },
                {
                    "workload_tag": "gemm", "shape_signature": "64x128x128", "sparsity_bucket": 0,
                    "topk_regret_pct": 5.0, "within_regret_pct": 6.0,
                    "topk_matches": 3, "within_matches": 3,
                    "topk_cycles": 1000, "within_cycles": 1000,
                },
            ]
            with regret_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=regret_cols)
                writer.writeheader()
                for row in regret_rows:
                    writer.writerow(row)

            trace_cols = ["workload_tag", "sparsity_bucket", "is_valid"]
            trace_rows = [
                {"workload_tag": "cnn_train_fwd", "sparsity_bucket": 0, "is_valid": 1},
                {"workload_tag": "cnn_train_fwd", "sparsity_bucket": 0, "is_valid": 1},
                {"workload_tag": "gemm", "sparsity_bucket": 0, "is_valid": 1},
            ]
            with trace_csv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=trace_cols)
                writer.writeheader()
                for row in trace_rows:
                    writer.writerow(row)

            rc = self.mod.main([
                "--regret-csv", str(regret_csv),
                "--trace-csv", str(trace_csv),
                "--top-regret-k", "1",
                "--disagreement-k", "1",
                "--underrepresented-min-rows", "2",
                "--underrepresented-k", "2",
                "--output-json", str(out_json),
            ])
            self.assertEqual(rc, 0)
            payload = json.loads(out_json.read_text(encoding="utf-8"))
            self.assertGreaterEqual(int(payload["summary"]["target_count"]), 1)
            targets = payload.get("targets", [])
            self.assertTrue(any("high_regret" in t.get("reasons", []) for t in targets))


if __name__ == "__main__":
    unittest.main()
