#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


FEATURE_COLUMNS: List[str] = [
    "M",
    "N",
    "K",
    "sparsity_pct",
    "sparsity_bucket",
    "mode1_candidate",
    "op_type_id",
    "activation",
    "batch_size",
    "seq_len",
    "channels",
    "kernel_h",
    "kernel_w",
    "scratchpad_avail",
    "macs",
    "dma_bytes_est",
]

TARGET_COLUMNS: List[str] = [
    "executed_hardware_dataflow_mode",
    "tile_m",
    "tile_n",
    "burst_size",
    "prefetch_depth",
    "tile_b",
]

SELECTION_POLICIES = ("best1", "topk", "within_pct")


def _default_workspace() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _default_csv_path(workspace: str) -> str:
    return str((Path(workspace) / "data" / "trace_dataset_latest.csv").resolve())


def _default_output_path(workspace: str) -> str:
    return str((Path(workspace) / "data" / "best_params_v1.json").resolve())


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _sparsity_bucket_from_row(row: Dict[str, Any]) -> int:
    sp = _to_int(row.get("sparsity_bucket", -1), -1)
    if sp >= 0:
        return int(max(0, min(10, sp)))
    sp_pct = _to_int(row.get("sparsity_pct", 0), 0)
    return int(max(0, min(10, sp_pct // 10)))


def _ledger_key(row: Dict[str, Any]) -> str:
    m_dim = _to_int(row.get("M", 0), 0)
    n_dim = _to_int(row.get("N", 0), 0)
    k_dim = _to_int(row.get("K", 0), 0)
    activation = _to_int(row.get("activation", 0), 0)
    workload_tag = str(row.get("workload_tag", "")).strip()
    sparsity_bucket = _sparsity_bucket_from_row(row)
    base = f"{m_dim}x{n_dim}x{k_dim}_act{activation}"
    if workload_tag:
        return f"{base}_{workload_tag}_sp{sparsity_bucket}"
    return base


def _group_id(row: Dict[str, Any]) -> str:
    split_key = str(row.get("_split_group_key", "")).strip()
    if split_key:
        return split_key
    workload_tag = str(row.get("workload_tag", "")).strip()
    shape_sig = str(row.get("shape_signature", "")).strip()
    activation = _to_int(row.get("activation", 0), 0)
    sparsity_bucket = _sparsity_bucket_from_row(row)
    return f"{workload_tag}:{shape_sig}:act{activation}:sp{sparsity_bucket}"


def _stable_u01(key: str, seed: int) -> float:
    token = f"{int(seed)}::{key}".encode("utf-8")
    h = hashlib.sha256(token).hexdigest()
    x = int(h[:16], 16)
    return float((x % 1_000_000) / 1_000_000.0)


def _stable_group_split(group_keys: List[str], eval_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    train_idx: List[int] = []
    eval_idx: List[int] = []
    eval_frac = float(max(0.0, min(0.9, eval_fraction)))
    for i, key in enumerate(group_keys):
        if _stable_u01(str(key), int(seed)) < eval_frac:
            eval_idx.append(i)
        else:
            train_idx.append(i)
    if not train_idx and eval_idx:
        train_idx.append(eval_idx.pop())
    if not eval_idx and train_idx:
        eval_idx.append(train_idx.pop())
    return train_idx, eval_idx


def _row_order_key(row: Dict[str, Any]) -> Tuple[int, str, int, float]:
    cycles = _to_int(row.get("cycles", 1 << 30), 1 << 30)
    run_id = str(row.get("run_id", ""))
    ep = _to_int(row.get("episode_id", 0), 0)
    ts = _to_float(row.get("timestamp", 0.0), 0.0)
    return int(cycles), run_id, int(ep), float(ts)


def _build_dataset(
    rows: List[Dict[str, Any]],
    *,
    selection_policy: str,
    top_k: int,
    within_pct: float,
    regret_weight_alpha: float,
    min_sample_weight: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
    dropped_invalid = 0
    for row in rows:
        is_valid = _to_int(row.get("is_valid", 0), 0)
        cycles = _to_int(row.get("cycles", -1), -1)
        if is_valid != 1 or cycles <= 0:
            dropped_invalid += 1
            continue
        key = _ledger_key(row)
        rec = dict(row)
        rec["_split_group_key"] = key
        grouped_rows.setdefault(key, []).append(rec)

    for key in list(grouped_rows.keys()):
        grouped_rows[key].sort(key=_row_order_key)

    materialization_samples: List[Dict[str, Any]] = []
    train_samples: List[Dict[str, Any]] = []
    policy = str(selection_policy).strip().lower()
    if policy not in SELECTION_POLICIES:
        raise ValueError(f"unknown selection policy: {selection_policy}")

    for key in sorted(grouped_rows.keys()):
        bucket = grouped_rows[key]
        if not bucket:
            continue
        best = bucket[0]
        best_cycles = float(max(1, _to_int(best.get("cycles", 1), 1)))
        materialization_samples.append(best)
        if policy == "best1":
            selected = [best]
        elif policy == "topk":
            k = max(1, int(top_k))
            selected = bucket[:k]
        else:
            pct = max(0.0, float(within_pct))
            thr = float(_to_int(best.get("cycles", 1 << 30), 1 << 30)) * (1.0 + pct)
            selected = [r for r in bucket if float(_to_int(r.get("cycles", 1 << 30), 1 << 30)) <= thr]
            if not selected:
                selected = [best]
        # Attach deterministic regret-aware sample weights.
        alpha = max(0.0, float(regret_weight_alpha))
        min_w = max(0.0, float(min_sample_weight))
        for row in selected:
            cyc = float(max(1, _to_int(row.get("cycles", 1), 1)))
            regret_pct = max(0.0, ((cyc - best_cycles) / best_cycles) * 100.0)
            if alpha > 0.0:
                w = float(np.exp(-alpha * (regret_pct / 100.0)))
            else:
                w = 1.0
            if min_w > 0.0:
                w = max(min_w, w)
            row["_oracle_best_cycles"] = int(best_cycles)
            row["_regret_pct"] = float(regret_pct)
            row["_sample_weight"] = float(w)
        train_samples.extend(selected)

    train_samples.sort(key=lambda r: (str(r.get("_split_group_key", "")),) + _row_order_key(r))
    materialization_samples.sort(key=lambda r: str(r.get("_split_group_key", "")))
    stats = {
        "input_rows": int(len(rows)),
        "dropped_invalid_rows": int(dropped_invalid),
        "unique_key_count": int(len(grouped_rows)),
        "materialization_best1_rows": int(len(materialization_samples)),
        "train_selected_rows": int(len(train_samples)),
        "selection_policy": policy,
        "top_k": int(max(1, int(top_k))),
        "within_pct": float(max(0.0, float(within_pct))),
        "regret_weight_alpha": float(max(0.0, float(regret_weight_alpha))),
        "min_sample_weight": float(max(0.0, float(min_sample_weight))),
    }
    return train_samples, materialization_samples, stats


def _read_csv_rows(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(dict(row))
    return out


def _read_parquet_rows(path: str) -> List[Dict[str, Any]]:
    try:
        import pyarrow.dataset as ds
    except Exception as exc:
        raise RuntimeError(
            "pyarrow is required for --input-parquet. Install with `pip install pyarrow`."
        ) from exc
    dataset = ds.dataset(str(Path(path).resolve()), format="parquet", partitioning="hive")
    table = dataset.to_table()
    return [dict(row) for row in table.to_pylist()]


def _sha256_path(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(1 << 20)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _read_json_if_exists(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        return {}
    return dict(obj) if isinstance(obj, dict) else {}


def _standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std == 0.0] = 1.0
    return mean, std


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def _sigmoid(z: np.ndarray) -> np.ndarray:
    zc = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-zc))


def _fit_logistic_ovr(
    x: np.ndarray,
    y: np.ndarray,
    classes: List[int],
    epochs: int,
    lr: float,
    l2: float,
    sample_weight: np.ndarray | None = None,
) -> Dict[str, Any]:
    if len(classes) == 1:
        return {
            "single_class": int(classes[0]),
            "classes": [int(classes[0])],
            "w": None,
            "b": None,
        }

    n, d = x.shape
    if sample_weight is None:
        sw = np.ones((n,), dtype=np.float64)
    else:
        sw = np.asarray(sample_weight, dtype=np.float64).reshape((n,))
    sw = np.clip(sw, 1e-6, 1e6)
    sw_norm = float(np.sum(sw))
    if sw_norm <= 0.0:
        sw = np.ones((n,), dtype=np.float64)
        sw_norm = float(n)
    w = np.zeros((len(classes), d), dtype=np.float64)
    b = np.zeros((len(classes),), dtype=np.float64)
    for ci, cls in enumerate(classes):
        y_bin = (y == int(cls)).astype(np.float64)
        wi = np.zeros((d,), dtype=np.float64)
        bi = 0.0
        for _ in range(max(1, int(epochs))):
            logits = (x @ wi) + bi
            p = _sigmoid(logits)
            err = (p - y_bin) * sw
            grad_w = (x.T @ err) / sw_norm + (float(l2) * wi)
            grad_b = float(np.sum(err) / sw_norm)
            wi -= float(lr) * grad_w
            bi -= float(lr) * grad_b
        w[ci, :] = wi
        b[ci] = bi
    return {
        "single_class": None,
        "classes": [int(c) for c in classes],
        "w": w,
        "b": b,
    }


def _predict_logistic_ovr_with_confidence(model: Dict[str, Any], x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    single_class = model.get("single_class")
    if single_class is not None:
        return (
            np.full((x.shape[0],), int(single_class), dtype=np.int64),
            np.ones((x.shape[0],), dtype=np.float64),
        )
    classes = np.asarray(model["classes"], dtype=np.int64)
    w = np.asarray(model["w"], dtype=np.float64)
    b = np.asarray(model["b"], dtype=np.float64)
    scores = (x @ w.T) + b
    idx = np.argmax(scores, axis=1)
    # OVR scores are converted with softmax for relative confidence.
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(np.clip(shifted, -60.0, 60.0))
    prob = exp_scores / np.maximum(np.sum(exp_scores, axis=1, keepdims=True), 1e-12)
    conf = prob[np.arange(prob.shape[0]), idx]
    return classes[idx], conf.astype(np.float64)


def _predict_logistic_ovr(model: Dict[str, Any], x: np.ndarray) -> np.ndarray:
    pred, _ = _predict_logistic_ovr_with_confidence(model, x)
    return pred


def _dataset_xy(samples: List[Dict[str, Any]], target: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    x_rows: List[List[float]] = []
    y_rows: List[int] = []
    w_rows: List[float] = []
    keys: List[str] = []
    for row in samples:
        x_rows.append([_to_float(row.get(c, 0.0), 0.0) for c in FEATURE_COLUMNS])
        y_rows.append(_to_int(row.get(target, 0), 0))
        w_rows.append(_to_float(row.get("_sample_weight", 1.0), 1.0))
        keys.append(_group_id(row))
    x = np.asarray(x_rows, dtype=np.float64)
    y = np.asarray(y_rows, dtype=np.int64)
    w = np.asarray(w_rows, dtype=np.float64)
    return x, y, w, keys


def _fit_target_models(
    train_samples: List[Dict[str, Any]],
    materialization_samples: List[Dict[str, Any]],
    eval_fraction: float,
    seed: int,
    epochs: int,
    lr: float,
    l2: float,
    confidence_fallback_threshold: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    predictions: Dict[str, np.ndarray] = {}
    confidences: Dict[str, np.ndarray] = {}
    metrics: Dict[str, Any] = {}
    for ti, target in enumerate(TARGET_COLUMNS):
        x_all, y_all, w_all, group_keys = _dataset_xy(train_samples, target)
        train_idx, eval_idx = _stable_group_split(group_keys, eval_fraction=float(eval_fraction), seed=(int(seed) + ti * 17))
        x_train = x_all[train_idx, :]
        y_train = y_all[train_idx]
        w_train = w_all[train_idx]
        x_eval = x_all[eval_idx, :]
        y_eval = y_all[eval_idx]

        mean, std = _standardize_fit(x_train)
        x_train_n = _standardize_apply(x_train, mean, std)
        x_eval_n = _standardize_apply(x_eval, mean, std)
        classes = sorted({int(v) for v in y_train.tolist()})
        model_eval = _fit_logistic_ovr(
            x_train_n,
            y_train,
            classes=classes,
            epochs=epochs,
            lr=lr,
            l2=l2,
            sample_weight=w_train,
        )
        pred_eval = _predict_logistic_ovr(model_eval, x_eval_n)
        acc = float(np.mean((pred_eval == y_eval).astype(np.float64))) if len(y_eval) > 0 else -1.0

        # Train final model on all samples for materialization.
        mean_all, std_all = _standardize_fit(x_all)
        x_all_n = _standardize_apply(x_all, mean_all, std_all)
        classes_all = sorted({int(v) for v in y_all.tolist()})
        model_all = _fit_logistic_ovr(
            x_all_n,
            y_all,
            classes=classes_all,
            epochs=epochs,
            lr=lr,
            l2=l2,
            sample_weight=w_all,
        )
        x_mat, _, _, _ = _dataset_xy(materialization_samples, target)
        x_mat_n = _standardize_apply(x_mat, mean_all, std_all)
        pred_mat, conf_mat = _predict_logistic_ovr_with_confidence(model_all, x_mat_n)
        predictions[target] = pred_mat.astype(np.int64)
        confidences[target] = conf_mat.astype(np.float64)
        train_groups = {group_keys[i] for i in train_idx}
        eval_groups = {group_keys[i] for i in eval_idx}
        class_counts_train: Dict[str, int] = {}
        class_counts_all: Dict[str, int] = {}
        for v in y_train.tolist():
            kk = str(int(v))
            class_counts_train[kk] = int(class_counts_train.get(kk, 0) + 1)
        for v in y_all.tolist():
            kk = str(int(v))
            class_counts_all[kk] = int(class_counts_all.get(kk, 0) + 1)
        metrics[target] = {
            "eval_accuracy": float(acc),
            "materialization_confidence_mean": float(np.mean(conf_mat)) if len(conf_mat) > 0 else -1.0,
            "materialization_confidence_min": float(np.min(conf_mat)) if len(conf_mat) > 0 else -1.0,
            "materialization_confidence_max": float(np.max(conf_mat)) if len(conf_mat) > 0 else -1.0,
            "confidence_fallback_threshold": float(confidence_fallback_threshold),
            "train_rows": int(len(train_idx)),
            "eval_rows": int(len(eval_idx)),
            "train_group_count": int(len(train_groups)),
            "eval_group_count": int(len(eval_groups)),
            "class_values_train": [int(v) for v in classes],
            "class_values_all": [int(v) for v in classes_all],
            "class_counts_train": class_counts_train,
            "class_counts_all": class_counts_all,
        }
    return predictions, confidences, metrics


def _entry_from_sample(sample: Dict[str, Any], preds: Dict[str, int]) -> Dict[str, Any]:
    cycles = _to_int(sample.get("cycles", -1), -1)
    hw_mode = int(max(0, min(1, int(preds["executed_hardware_dataflow_mode"]))))
    sparsity_bucket = _sparsity_bucket_from_row(sample)
    workload_tag = str(sample.get("workload_tag", "")).strip()
    ipc = float(3500.0 / max(1, cycles)) if cycles > 0 else 0.0
    return {
        "tile_m": int(preds["tile_m"]),
        "tile_n": int(preds["tile_n"]),
        "burst_size": int(preds["burst_size"]),
        "prefetch_depth": int(preds["prefetch_depth"]),
        "tile_b": int(preds["tile_b"]),
        "hardware_dataflow_mode": int(hw_mode),
        "ipc": float(ipc),
        "workload_tag": workload_tag,
        "sparsity_bucket": int(sparsity_bucket),
        "source": "logistic_baseline_v1",
        "observed_best_cycles": int(cycles),
    }


def _materialize_ledger(entries: Dict[str, Dict[str, Any]], ledger_in: str, ledger_out: str) -> Dict[str, Any]:
    src = Path(ledger_in)
    merged: Dict[str, Any] = {}
    if src.exists():
        with src.open("r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                merged = loaded
    for key, cfg in entries.items():
        merged[key] = {
            "tile_m": int(cfg["tile_m"]),
            "tile_n": int(cfg["tile_n"]),
            "burst_size": int(cfg["burst_size"]),
            "prefetch_depth": int(cfg["prefetch_depth"]),
            "tile_b": int(cfg["tile_b"]),
            "hardware_dataflow_mode": int(cfg["hardware_dataflow_mode"]),
            "ipc": float(cfg["ipc"]),
            "workload_tag": str(cfg.get("workload_tag", "")),
            "sparsity_bucket": int(cfg.get("sparsity_bucket", -1)),
        }
    out = Path(ledger_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, sort_keys=True)
    return {
        "ledger_in": str(src.resolve()),
        "ledger_out": str(out.resolve()),
        "entry_count": int(len(merged)),
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train deterministic logistic baselines from trace CSV and emit best_params_v1.")
    ap.add_argument("--workspace", type=str, default=_default_workspace(),
                    help="Accelera workspace root.")
    ap.add_argument("--input-csv", type=str, default="",
                    help="Trace CSV path from trace_to_csv.py.")
    ap.add_argument("--input-parquet", type=str, default="",
                    help="Optional Parquet dataset path (directory or file). When set, overrides --input-csv.")
    ap.add_argument("--output-json", type=str, default="",
                    help="Output best_params_v1.json path.")
    ap.add_argument("--seed", type=int, default=17,
                    help="Deterministic seed for split/training.")
    ap.add_argument("--eval-fraction", type=float, default=0.2,
                    help="Holdout fraction for deterministic group-safe evaluation.")
    ap.add_argument("--epochs", type=int, default=400,
                    help="Logistic OVR gradient steps per class.")
    ap.add_argument("--learning-rate", type=float, default=0.15,
                    help="Logistic OVR learning rate.")
    ap.add_argument("--l2", type=float, default=1e-4,
                    help="L2 regularization strength.")
    ap.add_argument("--selection-policy", type=str, default="best1", choices=list(SELECTION_POLICIES),
                    help="Training sample selection per key: best1|topk|within_pct.")
    ap.add_argument("--top-k", type=int, default=5,
                    help="Used when --selection-policy=topk: keep top-k rows per key by cycles.")
    ap.add_argument("--within-pct", type=float, default=0.05,
                    help="Used when --selection-policy=within_pct: keep rows within this fractional delta from best cycles.")
    ap.add_argument("--regret-weight-alpha", type=float, default=0.0,
                    help="Regret-aware sample weighting strength (0 disables weighting).")
    ap.add_argument("--min-sample-weight", type=float, default=0.05,
                    help="Floor for regret-weighted samples to avoid zeroing hard examples.")
    ap.add_argument("--confidence-fallback-threshold", type=float, default=0.0,
                    help="If >0, low-confidence predictions fallback to oracle best-per-key labels during materialization.")
    ap.add_argument("--materialize-ledger-out", type=str, default="",
                    help="Optional output ledger JSON path to merge/update with emitted entries.")
    ap.add_argument("--ledger-in", type=str, default="",
                    help="Optional source ledger path when materializing (default auto_tuner/compiler/bkm_ledger.json).")
    ap.add_argument("--artifact-manifest-out", type=str, default="",
                    help="Optional policy artifact manifest JSON output path.")
    ap.add_argument("--policy-id", type=str, default="logistic_baseline_v1",
                    help="Policy identifier written to artifact manifest.")
    ap.add_argument("--dataset-meta-json", type=str, default="",
                    help="Optional dataset metadata JSON to embed in artifact manifest.")
    ap.add_argument("--rl-space-contract-json", type=str, default="",
                    help="Optional RL exploration-space contract JSON to embed in artifact manifest.")
    ap.add_argument("--canonical-contract-json", type=str, default="",
                    help="Optional canonical policy contract JSON for provenance in artifact manifest.")
    args = ap.parse_args(argv)

    input_csv = args.input_csv or _default_csv_path(args.workspace)
    output_json = args.output_json or _default_output_path(args.workspace)
    input_kind = "csv"
    if str(args.input_parquet).strip():
        input_kind = "parquet"
        rows = _read_parquet_rows(str(Path(args.input_parquet).resolve()))
    else:
        rows = _read_csv_rows(str(Path(input_csv).resolve()))
    train_samples, materialization_samples, stats = _build_dataset(
        rows,
        selection_policy=str(args.selection_policy),
        top_k=int(args.top_k),
        within_pct=float(args.within_pct),
        regret_weight_alpha=float(args.regret_weight_alpha),
        min_sample_weight=float(args.min_sample_weight),
    )
    if len(train_samples) < 2:
        raise RuntimeError(f"not enough selected training samples in {input_csv}; need >=2, got {len(train_samples)}")
    if len(materialization_samples) < 1:
        raise RuntimeError(f"no materialization samples available from {input_csv}")

    preds, confs, metrics = _fit_target_models(
        train_samples=train_samples,
        materialization_samples=materialization_samples,
        eval_fraction=float(args.eval_fraction),
        seed=int(args.seed),
        epochs=int(args.epochs),
        lr=float(args.learning_rate),
        l2=float(args.l2),
        confidence_fallback_threshold=float(args.confidence_fallback_threshold),
    )

    entries: Dict[str, Dict[str, Any]] = {}
    fallback_counts: Dict[str, int] = {t: 0 for t in TARGET_COLUMNS}
    fallback_rows: int = 0
    conf_threshold = max(0.0, float(args.confidence_fallback_threshold))
    for i, sample in enumerate(materialization_samples):
        key = _ledger_key(sample)
        pred_map: Dict[str, int] = {}
        row_had_fallback = False
        for t in TARGET_COLUMNS:
            p = int(preds[t][i])
            c = float(confs[t][i]) if t in confs else 1.0
            if conf_threshold > 0.0 and c < conf_threshold:
                p = _to_int(sample.get(t, p), p)
                fallback_counts[t] = int(fallback_counts[t] + 1)
                row_had_fallback = True
            pred_map[t] = int(p)
        if row_had_fallback:
            fallback_rows += 1
        entries[key] = _entry_from_sample(sample, pred_map)

    payload: Dict[str, Any] = {
        "schema_version": 1,
        "generator": "auto_tuner/train_logistic_baseline.py",
        "generated_at_epoch_sec": float(time.time()),
        "input_csv": str(Path(input_csv).resolve()),
        "input_parquet": str(Path(args.input_parquet).resolve()) if str(args.input_parquet).strip() else "",
        "input_kind": input_kind,
        "feature_columns": list(FEATURE_COLUMNS),
        "target_columns": list(TARGET_COLUMNS),
        "seed": int(args.seed),
        "eval_fraction": float(max(0.0, min(0.9, float(args.eval_fraction)))),
        "selection_policy": str(args.selection_policy),
        "top_k": int(max(1, int(args.top_k))),
        "within_pct": float(max(0.0, float(args.within_pct))),
        "regret_weight_alpha": float(max(0.0, float(args.regret_weight_alpha))),
        "min_sample_weight": float(max(0.0, float(args.min_sample_weight))),
        "confidence_fallback_threshold": float(max(0.0, float(args.confidence_fallback_threshold))),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "l2": float(args.l2),
        "dataset_stats": stats,
        "target_metrics": metrics,
        "fallback_stats": {
            "confidence_fallback_threshold": float(conf_threshold),
            "fallback_rows": int(fallback_rows),
            "fallback_counts_by_target": fallback_counts,
        },
        "entries": entries,
    }

    out = Path(output_json).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[LogisticBaseline] selected_train_rows={len(train_samples)} "
        f"materialization_rows={len(materialization_samples)} output={out}"
    )

    materialized_ledger_out = ""
    if args.materialize_ledger_out:
        ledger_in = args.ledger_in or str((Path(args.workspace) / "auto_tuner" / "compiler" / "bkm_ledger.json").resolve())
        mat = _materialize_ledger(
            entries=entries,
            ledger_in=ledger_in,
            ledger_out=str(Path(args.materialize_ledger_out).resolve()),
        )
        materialized_ledger_out = str(mat["ledger_out"])
        print(
            f"[LogisticBaseline] materialized ledger_out={mat['ledger_out']} "
            f"entries={mat['entry_count']}"
        )

    if args.artifact_manifest_out:
        ds_meta = _read_json_if_exists(str(Path(args.dataset_meta_json).resolve())) if str(args.dataset_meta_json).strip() else {}
        rl_space = _read_json_if_exists(str(Path(args.rl_space_contract_json).resolve())) if str(args.rl_space_contract_json).strip() else {}
        canonical_contract = _read_json_if_exists(str(Path(args.canonical_contract_json).resolve())) if str(args.canonical_contract_json).strip() else {}
        manifest: Dict[str, Any] = {
            "schema_version": 1,
            "generator": "auto_tuner/train_logistic_baseline.py",
            "generated_at_epoch_sec": float(time.time()),
            "policy_id": str(args.policy_id).strip() or "logistic_baseline_v1",
            "input_kind": input_kind,
            "input_csv": str(Path(input_csv).resolve()),
            "input_parquet": str(Path(args.input_parquet).resolve()) if str(args.input_parquet).strip() else "",
            "dataset_meta_json": str(Path(args.dataset_meta_json).resolve()) if str(args.dataset_meta_json).strip() else "",
            "dataset_meta": ds_meta,
            "rl_space_contract_json": str(Path(args.rl_space_contract_json).resolve()) if str(args.rl_space_contract_json).strip() else "",
            "rl_space_contract": rl_space,
            "canonical_contract_json": str(Path(args.canonical_contract_json).resolve()) if str(args.canonical_contract_json).strip() else "",
            "canonical_contract": canonical_contract,
            "training_config": {
                "seed": int(args.seed),
                "eval_fraction": float(max(0.0, min(0.9, float(args.eval_fraction)))),
                "selection_policy": str(args.selection_policy),
                "top_k": int(max(1, int(args.top_k))),
                "within_pct": float(max(0.0, float(args.within_pct))),
                "regret_weight_alpha": float(max(0.0, float(args.regret_weight_alpha))),
                "min_sample_weight": float(max(0.0, float(args.min_sample_weight))),
                "confidence_fallback_threshold": float(max(0.0, float(args.confidence_fallback_threshold))),
                "epochs": int(args.epochs),
                "learning_rate": float(args.learning_rate),
                "l2": float(args.l2),
            },
            "dataset_stats": stats,
            "fallback_stats": payload.get("fallback_stats", {}),
            "target_metrics": payload.get("target_metrics", {}),
            "outputs": {
                "best_params_json": str(out),
                "best_params_json_sha256": _sha256_path(str(out)),
                "best_params_json_size_bytes": int(out.stat().st_size),
                "entry_count": int(len(entries)),
                "materialized_ledger_out": materialized_ledger_out,
            },
        }
        if materialized_ledger_out:
            led_p = Path(materialized_ledger_out)
            if led_p.exists():
                manifest["outputs"]["materialized_ledger_sha256"] = _sha256_path(str(led_p))
                manifest["outputs"]["materialized_ledger_size_bytes"] = int(led_p.stat().st_size)
        manifest_out = Path(args.artifact_manifest_out).resolve()
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        manifest_out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[LogisticBaseline] artifact_manifest={manifest_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
