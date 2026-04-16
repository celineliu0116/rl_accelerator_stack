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
MODULE_PATH = AUTO_TUNER_DIR / "export_rl_space_contract.py"


def _load_module():
    if str(AUTO_TUNER_DIR) not in sys.path:
        sys.path.insert(0, str(AUTO_TUNER_DIR))
    spec = importlib.util.spec_from_file_location("accelera_export_rl_space_contract", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {MODULE_PATH}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class ExportRLSpaceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            self.mod = _load_module()
        except ModuleNotFoundError as exc:
            self.skipTest(f"missing dependency for rl space export: {exc}")

    def test_exports_base_and_stage2_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            base = td_path / "base.json"
            stage2 = td_path / "stage2.json"

            rc0 = self.mod.main(["--output-json", str(base)])
            rc1 = self.mod.main(["--output-json", str(stage2), "--enable-tile-b4"])
            self.assertEqual(rc0, 0)
            self.assertEqual(rc1, 0)

            b = json.loads(base.read_text(encoding="utf-8"))
            s = json.loads(stage2.read_text(encoding="utf-8"))
            b_space = b["exploration_space"]
            s_space = s["exploration_space"]
            b_tile = [k for k in b_space["knobs"] if k["name"] == "tile_b"][0]
            s_tile = [k for k in s_space["knobs"] if k["name"] == "tile_b"][0]
            self.assertEqual(b_tile["values"], [1, 2])
            self.assertEqual(s_tile["values"], [1, 2, 4])
            self.assertGreater(int(s_space["action_space_cardinality"]), int(b_space["action_space_cardinality"]))


if __name__ == "__main__":
    unittest.main()
