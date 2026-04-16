#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from env.systolic_env import SystolicEnv


def _default_workspace() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_output(workspace: str) -> str:
    return str((Path(workspace) / "data" / "canonical" / "policy" / "rl_exploration_space_v1.json").resolve())


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export source-of-truth RL exploration space contract.")
    ap.add_argument("--workspace", type=str, default=_default_workspace(), help="Accelera workspace root.")
    ap.add_argument("--output-json", type=str, default="", help="Output contract JSON path.")
    ap.add_argument("--enable-tile-b4", action="store_true", help="Emit stage-2 tile_b contract with tile_b=4 active.")
    args = ap.parse_args(argv)

    workspace = str(Path(args.workspace).resolve())
    output = str(Path(args.output_json).resolve()) if args.output_json else _default_output(workspace)

    payload: Dict[str, Any] = {
        "contract_version": 1,
        "workspace": workspace,
        "exploration_space": SystolicEnv.exploration_space_contract(enable_tile_b4=bool(args.enable_tile_b4)),
    }

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[ExportRLSpaceContract] output={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
