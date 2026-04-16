#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from surrogate_model import SurrogateModel


def _default_workspace() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    ap = argparse.ArgumentParser(description="Train surrogate model from tuning trace JSONL.")
    ap.add_argument("--workspace", type=str, default=_default_workspace())
    ap.add_argument("--trace-path", type=str, default="")
    ap.add_argument("--model-path", type=str, default="")
    ap.add_argument("--min-records", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

    trace_path = args.trace_path or os.path.join(args.workspace, "data", "tuning_trace.jsonl")
    model_path = args.model_path or os.path.join(args.workspace, "data", "surrogate_model.pt")

    metrics = SurrogateModel.train_from_trace(
        trace_path=trace_path,
        model_path=model_path,
        min_records=args.min_records,
        epochs=args.epochs,
    )
    if metrics is None:
        print(f"[SurrogateTrain] not enough records in {trace_path} (need >= {args.min_records})")
        return
    print(
        f"[SurrogateTrain] done model={model_path} "
        f"n_train={metrics.n_train} n_val={metrics.n_val} "
        f"mae_cycles={metrics.mae_cycles:.1f} mape={metrics.mape:.3f}"
    )


if __name__ == "__main__":
    main()

