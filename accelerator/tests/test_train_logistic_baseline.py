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
MODULE_PATH = AUTO_TUNER_DIR / "train_logistic_baseline.py"


def _load_module():
    if str(AUTO_TUNER_DIR) not in sys.path:
        sys.path.insert(0, str(AUTO_TUNER_DIR))
    spec = importlib.util.spec_from_file_location("accelera_train_logistic_baseline", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class TrainLogisticBaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load_module()

    def _write_input_csv(self, path: Path) -> None:
        cols = sorted(set(
            self.mod.FEATURE_COLUMNS
            + self.mod.TARGET_COLUMNS
            + [
                "is_valid",
                "cycles",
                "workload_tag",
                "shape_signature",
                "sparsity_bucket",
            ]
        ))
        rows = [
            {
                "M": 64, "N": 128, "K": 128, "activation": 0,
                "sparsity_pct": 5, "sparsity_bucket": 0, "mode1_candidate": 0, "op_type_id": 0,
                "batch_size": 1, "seq_len": 1, "channels": 1, "kernel_h": 1, "kernel_w": 1,
                "scratchpad_avail": 1000, "macs": 1048576, "dma_bytes_est": 65536,
                "executed_hardware_dataflow_mode": 0, "tile_m": 8, "tile_n": 16,
                "burst_size": 32, "prefetch_depth": 1, "tile_b": 1,
                "is_valid": 1, "cycles": 1000, "workload_tag": "gemm", "shape_signature": "64x128x128",
            },
            {
                "M": 64, "N": 128, "K": 128, "activation": 0,
                "sparsity_pct": 5, "sparsity_bucket": 0, "mode1_candidate": 0, "op_type_id": 0,
                "batch_size": 1, "seq_len": 1, "channels": 1, "kernel_h": 1, "kernel_w": 1,
                "scratchpad_avail": 1000, "macs": 1048576, "dma_bytes_est": 65536,
                "executed_hardware_dataflow_mode": 1, "tile_m": 12, "tile_n": 16,
                "burst_size": 32, "prefetch_depth": 2, "tile_b": 1,
                "is_valid": 1, "cycles": 1400, "workload_tag": "gemm", "shape_signature": "64x128x128",
            },
            {
                "M": 128, "N": 256, "K": 128, "activation": 1,
                "sparsity_pct": 20, "sparsity_bucket": 2, "mode1_candidate": 1, "op_type_id": 2,
                "batch_size": 8, "seq_len": 64, "channels": 64, "kernel_h": 3, "kernel_w": 3,
                "scratchpad_avail": 1200, "macs": 4194304, "dma_bytes_est": 262144,
                "executed_hardware_dataflow_mode": 1, "tile_m": 12, "tile_n": 24,
                "burst_size": 64, "prefetch_depth": 2, "tile_b": 2,
                "is_valid": 1, "cycles": 900, "workload_tag": "convolution", "shape_signature": "128x256x128",
            },
            {
                "M": 128, "N": 256, "K": 128, "activation": 1,
                "sparsity_pct": 20, "sparsity_bucket": 2, "mode1_candidate": 1, "op_type_id": 2,
                "batch_size": 8, "seq_len": 64, "channels": 64, "kernel_h": 3, "kernel_w": 3,
                "scratchpad_avail": 1200, "macs": 4194304, "dma_bytes_est": 262144,
                "executed_hardware_dataflow_mode": 0, "tile_m": 8, "tile_n": 16,
                "burst_size": 32, "prefetch_depth": 1, "tile_b": 1,
                "is_valid": 1, "cycles": 1200, "workload_tag": "convolution", "shape_signature": "128x256x128",
            },
        ]
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow(row)

    def test_pipeline_emits_best_params_and_materialized_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            input_csv = td_path / "trace_dataset.csv"
            output_json = td_path / "best_params_v1.json"
            ledger_out = td_path / "bkm_ledger.materialized.json"
            self._write_input_csv(input_csv)

            rc = self.mod.main([
                "--input-csv", str(input_csv),
                "--output-json", str(output_json),
                "--materialize-ledger-out", str(ledger_out),
                "--epochs", "40",
                "--seed", "3",
            ])
            self.assertEqual(rc, 0)

            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(int(payload["schema_version"]), 1)
            entries = payload["entries"]
            self.assertTrue(entries)
            self.assertIn("64x128x128_act0_gemm_sp0", entries)
            self.assertIn("128x256x128_act1_convolution_sp2", entries)

            mat = json.loads(ledger_out.read_text(encoding="utf-8"))
            self.assertIn("64x128x128_act0_gemm_sp0", mat)
            self.assertIn("128x256x128_act1_convolution_sp2", mat)
            self.assertIn("hardware_dataflow_mode", mat["64x128x128_act0_gemm_sp0"])

    def test_topk_training_keeps_best1_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            input_csv = td_path / "trace_dataset.csv"
            output_json = td_path / "best_params_topk.json"
            self._write_input_csv(input_csv)

            rc = self.mod.main([
                "--input-csv", str(input_csv),
                "--output-json", str(output_json),
                "--selection-policy", "topk",
                "--top-k", "2",
                "--epochs", "40",
                "--seed", "11",
            ])
            self.assertEqual(rc, 0)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            stats = payload.get("dataset_stats", {})
            self.assertEqual(str(stats.get("selection_policy")), "topk")
            self.assertGreater(int(stats.get("train_selected_rows", 0)), int(stats.get("materialization_best1_rows", 0)))
            entries = payload.get("entries", {})
            self.assertEqual(int(stats.get("materialization_best1_rows", -1)), len(entries))

    def test_confidence_fallback_uses_oracle_best_labels(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            input_csv = td_path / "trace_dataset.csv"
            output_json = td_path / "best_params_fallback.json"
            self._write_input_csv(input_csv)

            rc = self.mod.main([
                "--input-csv", str(input_csv),
                "--output-json", str(output_json),
                "--selection-policy", "topk",
                "--top-k", "2",
                "--epochs", "40",
                "--seed", "13",
                "--confidence-fallback-threshold", "1.1",
            ])
            self.assertEqual(rc, 0)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            fallback = payload.get("fallback_stats", {})
            self.assertEqual(int(fallback.get("fallback_rows", 0)), 2)
            counts = fallback.get("fallback_counts_by_target", {})
            for target in self.mod.TARGET_COLUMNS:
                self.assertEqual(int(counts.get(target, -1)), 2)

            entries = payload.get("entries", {})
            gemm = entries["64x128x128_act0_gemm_sp0"]
            conv = entries["128x256x128_act1_convolution_sp2"]
            self.assertEqual(int(gemm["hardware_dataflow_mode"]), 0)
            self.assertEqual(int(gemm["tile_m"]), 8)
            self.assertEqual(int(gemm["tile_n"]), 16)
            self.assertEqual(int(conv["hardware_dataflow_mode"]), 1)
            self.assertEqual(int(conv["tile_m"]), 12)
            self.assertEqual(int(conv["tile_n"]), 24)

    def test_pipeline_accepts_parquet_input(self) -> None:
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception:
            self.skipTest("pyarrow not installed")

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            input_csv = td_path / "trace_dataset.csv"
            input_parquet = td_path / "trace_dataset.parquet"
            output_json = td_path / "best_params_parquet.json"
            self._write_input_csv(input_csv)

            rows = []
            with input_csv.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(dict(row))
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, input_parquet)

            rc = self.mod.main([
                "--input-parquet", str(input_parquet),
                "--output-json", str(output_json),
                "--epochs", "40",
                "--seed", "7",
            ])
            self.assertEqual(rc, 0)
            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(str(payload.get("input_kind")), "parquet")
            self.assertTrue(payload.get("entries"))

    def test_artifact_manifest_emission(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            input_csv = td_path / "trace_dataset.csv"
            output_json = td_path / "best_params_manifest.json"
            manifest_json = td_path / "best_params_manifest.meta.json"
            dataset_meta = td_path / "dataset_meta.json"
            rl_space = td_path / "rl_space.json"
            self._write_input_csv(input_csv)
            dataset_meta.write_text(json.dumps({"dataset_schema_version": 1, "aggregate_rows": 4}), encoding="utf-8")
            rl_space.write_text(json.dumps({"contract_version": 1, "exploration_space": {"action_space_cardinality": 768}}), encoding="utf-8")

            rc = self.mod.main([
                "--input-csv", str(input_csv),
                "--output-json", str(output_json),
                "--artifact-manifest-out", str(manifest_json),
                "--dataset-meta-json", str(dataset_meta),
                "--rl-space-contract-json", str(rl_space),
                "--policy-id", "logistic_topk_v1",
                "--epochs", "40",
                "--seed", "9",
            ])
            self.assertEqual(rc, 0)
            manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
            self.assertEqual(str(manifest.get("policy_id")), "logistic_topk_v1")
            self.assertEqual(str(manifest.get("input_kind")), "csv")
            self.assertIn("best_params_json_sha256", manifest["outputs"])
            self.assertEqual(int(manifest["dataset_meta"].get("aggregate_rows", -1)), 4)
            self.assertEqual(int(manifest["rl_space_contract"].get("contract_version", -1)), 1)


if __name__ == "__main__":
    unittest.main()
