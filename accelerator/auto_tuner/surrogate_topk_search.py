#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env.systolic_env import SystolicEnv
from surrogate_model import SurrogateModel


def _default_workspace() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _sample_actions(env: SystolicEnv, count: int) -> list[np.ndarray]:
    return [env.action_space.sample() for _ in range(max(1, count))]


def _predict_cycles(env: SystolicEnv, surrogate: SurrogateModel, action: np.ndarray) -> float:
    tile_m, tile_n, burst_size, prefetch_depth, tile_b, hw_mode = env._decode_action(action)
    if env._constraint_violations(tile_m, tile_n, burst_size, prefetch_depth, tile_b, hw_mode):
        return float("inf")
    rec = env._base_trace_record(tile_m, tile_n, burst_size, prefetch_depth, tile_b, hw_mode)
    return float(surrogate.predict_cycles(rec))


def main() -> None:
    ap = argparse.ArgumentParser(description="Surrogate-assisted top-K schedule search.")
    ap.add_argument("--workspace", type=str, default=_default_workspace())
    ap.add_argument("--workload", type=str, default="all")
    ap.add_argument("--shape-split", type=str, default="all")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--candidates", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--warmup-sim", type=int, default=20)
    ap.add_argument("--retrain-every", type=int, default=5)
    ap.add_argument("--trace-path", type=str, default="")
    ap.add_argument("--surrogate-model-path", type=str, default="")
    args = ap.parse_args()

    trace_path = args.trace_path or os.path.join(args.workspace, "data", "tuning_trace.jsonl")
    model_path = args.surrogate_model_path or os.path.join(args.workspace, "data", "surrogate_model.pt")

    env = SystolicEnv(
        M_target=128, N_target=128, K_target=784,
        workspace_dir=args.workspace,
        workload_diversity=True,
        workload_selector=args.workload,
        shape_split=args.shape_split,
        include_workload_features=True,
        trace_path=trace_path,
        surrogate_enabled=False,
        surrogate_model_path=model_path,
    )
    surrogate = SurrogateModel(model_path)

    print(f"[TopK] warmup simulations: {args.warmup_sim}")
    for _ in range(max(0, args.warmup_sim)):
        env.reset()
        a = env.action_space.sample()
        env.step(a)

    m = SurrogateModel.train_from_trace(trace_path, model_path, min_records=20)
    surrogate.refresh_if_stale()
    if m is not None:
        print(f"[TopK] initial surrogate: n_train={m.n_train} mae={m.mae_cycles:.1f} mape={m.mape:.3f}")

    topk = max(1, int(args.top_k))
    rounds = max(1, int(args.rounds))
    for r in range(rounds):
        env.reset()
        pool = _sample_actions(env, args.candidates)

        if surrogate.ready:
            pred = [(i, _predict_cycles(env, surrogate, a)) for i, a in enumerate(pool)]
            pred.sort(key=lambda x: x[1])
            finite = [(i, p) for i, p in pred if np.isfinite(p)]
            if finite:
                picked = [pool[i] for i, _ in finite[:topk]]
                print(f"[TopK] round={r+1}/{rounds} surrogate picks min_pred={finite[0][1]:.1f}")
            else:
                picked = pool[:topk]
                print(f"[TopK] round={r+1}/{rounds} no feasible predicted candidates, fallback random")
        else:
            picked = pool[:topk]
            print(f"[TopK] round={r+1}/{rounds} surrogate unavailable, using random top-k")

        best_cycles = None
        best_reward = None
        valid = 0
        for a in picked:
            _, rew, _, _, info = env.step(a)
            if "cycles" in info:
                valid += 1
                cyc = int(info["cycles"])
                if best_cycles is None or cyc < best_cycles:
                    best_cycles = cyc
                    best_reward = rew

        print(f"[TopK] round={r+1} valid={valid}/{len(picked)} best_cycles={best_cycles} best_reward={best_reward}")

        if ((r + 1) % max(1, int(args.retrain_every))) == 0:
            mm = SurrogateModel.train_from_trace(trace_path, model_path, min_records=20)
            surrogate.refresh_if_stale()
            if mm is not None:
                print(f"[TopK] retrain round={r+1}: n_train={mm.n_train} mae={mm.mae_cycles:.1f} mape={mm.mape:.3f}")


if __name__ == "__main__":
    main()
