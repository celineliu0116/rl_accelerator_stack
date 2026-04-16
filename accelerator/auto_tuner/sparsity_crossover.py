#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import struct
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


PASS_RE = re.compile(r"\*\*\* PASSED \*\*\* after (\d+) simulation cycles")
FAIL_RE = re.compile(r"\*\*\* FAILED \*\*\* \(tohost = (\d+)\)")
TIMEOUT_RE = re.compile(r"\*\*\* TIMEOUT \*\*\* after (\d+) simulation cycles")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_sparsities(args: argparse.Namespace) -> List[int]:
    if args.sparsity_list.strip():
        vals = [int(x.strip()) for x in args.sparsity_list.split(",") if x.strip()]
        return sorted(set(max(0, min(99, v)) for v in vals))
    start = int(args.sparsity_start)
    end = int(args.sparsity_end)
    step = max(1, int(args.sparsity_step))
    if end < start:
        start, end = end, start
    return list(range(start, end + 1, step))


def _write_override(path: Path, m: int, n: int, k: int, tile_m: int, tile_n: int,
                    burst: int, prefetch: int, tile_b: int, hw_mode: int) -> None:
    payload = {
        "M": int(m),
        "N": int(n),
        "K": int(k),
        "tile_m": int(tile_m),
        "tile_n": int(tile_n),
        "burst_size": int(burst),
        "prefetch_depth": int(prefetch),
        "tile_b": int(tile_b),
        "hardware_dataflow_mode": int(hw_mode),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _read_layer0_mode(model_bin_path: Path) -> int:
    blob = model_bin_path.read_bytes()
    # layer header starts at byte 16; flags is the 12th uint32 in layer header.
    flags = struct.unpack_from("<I", blob, 16 + 11 * 4)[0]
    return int((flags >> 5) & 0x3)


def _run(cmd: List[str], cwd: Path, env: Dict[str, str], timeout_s: int) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )


def _extract_result(log: str) -> tuple[str, int, int]:
    m = PASS_RE.search(log)
    if m:
        return "passed", int(m.group(1)), 0
    m = FAIL_RE.search(log)
    if m:
        return "failed", -1, int(m.group(1))
    m = TIMEOUT_RE.search(log)
    if m:
        return "timeout", int(m.group(1)), 0
    return "unknown", -1, 0


def _fallback_reason(export_log: str) -> str:
    # Example: "[Export] Layer 0: hw_mode=1 rejected (mode1_disabled_env); falling back..."
    m = re.search(r"hw_mode=1 rejected \(([^)]+)\)", export_log)
    if m:
        return m.group(1)
    m = re.search(r"hw_mode=1 self-check failed \(([^)]+)\)", export_log)
    if m:
        return m.group(1)
    return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep sparsity and compare dense vs sparse mode-1 cycles.")
    ap.add_argument("--workload-tag", type=str, default="sparse_mlp")
    ap.add_argument("--kind-id", type=int, default=11)
    ap.add_argument("--activation", type=int, choices=[0, 1], default=0)
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tile-m", type=int, default=8)
    ap.add_argument("--tile-n", type=int, default=16)
    ap.add_argument("--burst-size", type=int, default=32)
    ap.add_argument("--prefetch-depth", type=int, default=2)
    ap.add_argument("--tile-b", type=int, default=1)
    ap.add_argument("--modes", type=str, default="0,1",
                    help="Comma-separated hardware modes to test (typically 0,1).")
    ap.add_argument("--sparsity-list", type=str, default="")
    ap.add_argument("--sparsity-start", type=int, default=0)
    ap.add_argument("--sparsity-end", type=int, default=95)
    ap.add_argument("--sparsity-step", type=int, default=5)
    ap.add_argument("--max-cycles", type=int, default=50000000)
    ap.add_argument("--sim-timeout-sec", type=int, default=240)
    ap.add_argument("--retry-on-timeout", type=int, default=1,
                    help="Number of re-runs after timeout/unknown before recording failure.")
    ap.add_argument("--disable-sw-mode2-policy", type=int, default=1,
                    help="Set 1 to force requested mode semantics (disable exporter auto mode-2 override).")
    ap.add_argument("--output-csv", type=str, default="")
    args = ap.parse_args()

    root = _repo_root()
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.output_csv.strip():
        out_csv = Path(args.output_csv)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_csv = data_dir / f"sparsity_crossover_{stamp}.csv"

    override_path = root / "auto_tuner" / "rl_override.json"
    model_bin_path = root / "firmware" / "include" / "model.bin"

    sparsities = _parse_sparsities(args)
    modes = [int(x.strip()) for x in args.modes.split(",") if x.strip()]
    rows: List[Dict[str, object]] = []
    run_case_id = 0

    for sparsity in sparsities:
        for mode in modes:
            run_case_id += 1
            _write_override(
                override_path, m=args.m, n=args.n, k=args.k,
                tile_m=args.tile_m, tile_n=args.tile_n,
                burst=args.burst_size, prefetch=args.prefetch_depth,
                tile_b=args.tile_b, hw_mode=mode
            )

            env = dict(os.environ)
            env["ACCELERA_SPARSE_MODE1_MIN_SPARSITY_PCT"] = "0"
            if int(args.disable_sw_mode2_policy) != 0:
                env["ACCELERA_ENABLE_SW_SPARSE_MODE2"] = "0"
            if mode == 1:
                env["ACCELERA_ENABLE_SPARSE_MODE1"] = "1"
            else:
                env.pop("ACCELERA_ENABLE_SPARSE_MODE1", None)

            export_cmd = [
                sys.executable, "auto_tuner/workload_export.py",
                "--workload-tag", str(args.workload_tag),
                "--kind-id", str(args.kind_id),
                "--m", str(args.m),
                "--n", str(args.n),
                "--k", str(args.k),
                "--sparsity-pct", str(sparsity),
                "--seed", str(args.seed),
                "--activation", str(args.activation),
            ]
            ex = _run(export_cmd, cwd=root, env=env, timeout_s=60)
            export_log = (ex.stdout or "") + (ex.stderr or "")
            if ex.returncode != 0:
                rows.append({
                    "sparsity_pct": sparsity,
                    "requested_mode": mode,
                    "executed_mode": -1,
                    "status": "export_failed",
                    "cycles": -1,
                    "tohost": 0,
                    "fallback_reason": "",
                })
                print(f"[Sweep] sp={sparsity}% mode={mode}: export_failed")
                continue

            executed_mode = _read_layer0_mode(model_bin_path)
            fallback_reason = _fallback_reason(export_log)

            run_cmd = [
                "make", "run",
                "INFERENCE_SRC=inference_generic.c",
                "ENABLE_DUAL_ISSUE=1",
                "GENERIC_CFLAGS=-O3 -fno-strict-aliasing",
                f"CFLAGS_EXTRA=-DRL_AUTOTUNE_MODE -DRL_EPISODE_ID={run_case_id}",
                f"EXTRA_FLAGS=+max-cycles={int(args.max_cycles)}",
                "EXTRA_VFLAGS=-DUSE_SYSTOLIC_ACCEL",
            ]
            attempts_left = max(0, int(args.retry_on_timeout))
            status = "unknown"
            cycles = -1
            tohost = 0
            while True:
                rr = _run(run_cmd, cwd=root, env=env, timeout_s=int(args.sim_timeout_sec))
                sim_log = (rr.stdout or "") + (rr.stderr or "")
                status, cycles, tohost = _extract_result(sim_log)
                if status in {"passed", "failed"}:
                    break
                if attempts_left <= 0:
                    break
                attempts_left -= 1

            rows.append({
                "sparsity_pct": sparsity,
                "requested_mode": mode,
                "executed_mode": executed_mode,
                "status": status,
                "cycles": cycles,
                "tohost": tohost,
                "fallback_reason": fallback_reason,
            })
            print(
                f"[Sweep] sp={sparsity:2d}% req_mode={mode} exec_mode={executed_mode} "
                f"status={status} cycles={cycles} "
                f"{'(fallback:'+fallback_reason+')' if fallback_reason else ''}"
            )

    # Compute per-sparsity speedup if both dense and mode1 executed successfully.
    by_sp: Dict[int, Dict[int, Dict[str, object]]] = {}
    for r in rows:
        sp = int(r["sparsity_pct"])
        req_mode = int(r["requested_mode"])
        by_sp.setdefault(sp, {})[req_mode] = r

    for sp, rs in by_sp.items():
        r0 = rs.get(0)
        r1 = rs.get(1)
        speedup = ""
        if r0 and r1:
            if int(r0["executed_mode"]) == 0 and int(r1["executed_mode"]) == 1:
                c0 = int(r0["cycles"])
                c1 = int(r1["cycles"])
                if c0 > 0 and c1 > 0:
                    speedup = f"{(c0 / c1):.4f}"
        if speedup:
            print(f"[Sweep] sp={sp:2d}% dense/mode1 speedup={speedup}x")

    cols = [
        "sparsity_pct", "requested_mode", "executed_mode", "status",
        "cycles", "tohost", "fallback_reason"
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[Sweep] wrote {out_csv}")


if __name__ == "__main__":
    main()
