#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from rl_daemon import run_daemon, _default_workspace


SCENARIOS: Dict[str, Dict[str, Any]] = {
    # Family generalization splits
    "family_split_gemm_sparse_to_attn_conv": {
        "workload_diversity": True,
        "workload": "gemm,sparse_mlp",
        "eval_workloads": "attention,convolution",
    },
    "family_split_gemm_attn_to_sparse": {
        "workload_diversity": True,
        "workload": "gemm,attention",
        "eval_workloads": "sparse_mlp",
    },
    # Unseen-shape split within same family distribution.
    "unseen_shapes_all_families": {
        "workload_diversity": True,
        "workload": "all",
        "train_shape_split": "train",
        "eval_workloads": "all",
        "eval_shape_split": "test",
    },
    # Ablations
    "ablate_no_workload_features": {
        "workload_diversity": True,
        "workload": "all",
        "eval_workloads": "all",
        "include_workload_features": False,
    },
    "ablate_mnist_only_train": {
        "workload_diversity": False,
        "workload": "all",
        "eval_workloads": "all",
    },
    "ablate_generic_ledger_lookup": {
        "workload_diversity": True,
        "workload": "all",
        "eval_workloads": "all",
        "workload_aware_lookup": False,
    },
}


def _parse_scenarios(selector: str) -> List[str]:
    raw = (selector or "default").strip().lower()
    if raw in {"all", "*"}:
        return list(SCENARIOS.keys())
    if raw in {"default", ""}:
        return [
            "family_split_gemm_sparse_to_attn_conv",
            "family_split_gemm_attn_to_sparse",
            "unseen_shapes_all_families",
        ]
    names = [x.strip() for x in raw.split(",") if x.strip()]
    bad = [n for n in names if n not in SCENARIOS]
    if bad:
        raise ValueError(f"unknown scenarios: {','.join(bad)}")
    return names


def _run_one(name: str, cfg: Dict[str, Any], args: argparse.Namespace) -> List[Dict[str, Any]]:
    print("\n==================================================")
    print(f"[EvalMatrix] Scenario: {name}")
    print("==================================================")
    rows = run_daemon(
        total_timesteps=args.timesteps,
        target_M=args.M,
        target_N=args.N,
        target_K=args.K,
        workspace_dir=args.workspace,
        fresh_start=True,
        ppo_n_steps=args.n_steps,
        ppo_batch_size=args.batch_size,
        workload_diversity=bool(cfg.get("workload_diversity", True)),
        workload_selector=str(cfg.get("workload", "all")),
        train_shape_split=str(cfg.get("train_shape_split", "all")),
        include_workload_features=bool(cfg.get("include_workload_features", True)),
        workload_aware_lookup=bool(cfg.get("workload_aware_lookup", True)),
        eval_workload_selector=str(cfg.get("eval_workloads", "all")),
        eval_episodes=args.eval_episodes,
        eval_shape_split=str(cfg.get("eval_shape_split", "all")),
    )
    if not rows:
        rows = [{
            "workload_family": "none",
            "shape_signature": "none",
            "episodes": 0,
            "valid_runs": 0,
            "avg_cycles": -1.0,
            "avg_reward": -10000.0,
        }]

    out = []
    for r in rows:
        rr = dict(r)
        rr["scenario"] = name
        rr["train_workloads"] = str(cfg.get("workload", "all"))
        rr["train_shape_split"] = str(cfg.get("train_shape_split", "all"))
        rr["eval_workloads"] = str(cfg.get("eval_workloads", "all"))
        rr["eval_shape_split"] = str(cfg.get("eval_shape_split", "all"))
        rr["with_workload_features"] = int(bool(cfg.get("include_workload_features", True)))
        rr["workload_aware_lookup"] = int(bool(cfg.get("workload_aware_lookup", True)))
        rr["workload_diversity"] = int(bool(cfg.get("workload_diversity", True)))
        out.append(rr)
    return out


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_cols = [
        "scenario",
        "train_workloads",
        "train_shape_split",
        "eval_workloads",
        "eval_shape_split",
        "with_workload_features",
        "workload_aware_lookup",
        "workload_diversity",
        "workload_family",
        "shape_signature",
        "episodes",
        "valid_runs",
        "avg_cycles",
        "avg_reward",
    ]
    # Keep CSV forward-compatible when run_daemon adds new eval fields.
    extra_cols = sorted({k for r in rows for k in r.keys() if k not in set(base_cols)})
    cols = base_cols + extra_cols
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run RL train/eval matrix and ablations.")
    ap.add_argument("--scenarios", type=str, default="default",
                    help="default|all|comma-list of scenario keys.")
    ap.add_argument("--timesteps", type=int, default=8,
                    help="Train timesteps per scenario.")
    ap.add_argument("--eval-episodes", type=int, default=4,
                    help="Eval episodes per scenario.")
    ap.add_argument("--n-steps", type=int, default=2, help="PPO n_steps.")
    ap.add_argument("--batch-size", type=int, default=2, help="PPO batch size.")
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--N", type=int, default=128)
    ap.add_argument("--K", type=int, default=784)
    ap.add_argument("--workspace", type=str, default=_default_workspace())
    ap.add_argument("--output-csv", type=str, default="")
    args = ap.parse_args()

    picked = _parse_scenarios(args.scenarios)
    all_rows: List[Dict[str, Any]] = []
    for name in picked:
        all_rows.extend(_run_one(name, SCENARIOS[name], args))

    if args.output_csv:
        out_path = Path(args.output_csv)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(args.workspace) / "data" / f"rl_eval_matrix_{stamp}.csv"
    _write_csv(out_path, all_rows)

    print("\n[EvalMatrix] Results:")
    for r in all_rows:
        valid_runs = r.get("valid_runs")
        episodes = r.get("episodes")
        if valid_runs is None:
            valid_ratio_mean = r.get("valid_ratio_mean")
            eps_per_repeat = r.get("episodes_per_repeat", r.get("episodes", 0))
            repeats = r.get("repeats", 1)
            if isinstance(valid_ratio_mean, (int, float)):
                valid_runs = int(round(float(valid_ratio_mean) * float(eps_per_repeat) * float(repeats)))
            else:
                valid_runs = "n/a"
        if episodes is None:
            eps_per_repeat = r.get("episodes_per_repeat")
            repeats = r.get("repeats")
            if isinstance(eps_per_repeat, (int, float)) and isinstance(repeats, (int, float)):
                episodes = int(float(eps_per_repeat) * float(repeats))
            else:
                episodes = "n/a"

        avg_cycles = r.get("avg_cycles", r.get("cycles_mean", -1.0))
        avg_reward = r.get("avg_reward", r.get("reward_mean", -10000.0))
        print(
            f"scenario={r['scenario']} family={r['workload_family']} "
            f"shape={r['shape_signature']} valid={valid_runs}/{episodes} "
            f"avg_cycles={avg_cycles} avg_reward={float(avg_reward):.3f}"
        )
    print(f"[EvalMatrix] Wrote CSV: {out_path}")


if __name__ == "__main__":
    main()
