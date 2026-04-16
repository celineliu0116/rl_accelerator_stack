#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import numpy as np

_repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_repo_root / "compiler"))

from export_model import ACT_NONE, ACT_RELU, build_model_blob, write_model_blob_header


def _weight_matrix(n_dim: int, k_dim: int, kind_id: int, sparsity_pct: int, seed: int) -> np.ndarray:
    n_idx = np.arange(n_dim, dtype=np.uint64)[:, None]
    k_idx = np.arange(k_dim, dtype=np.uint64)[None, :]
    h = (
        np.uint64(seed)
        ^ (n_idx * np.uint64(1315423911))
        ^ (k_idx * np.uint64(2654435761))
        ^ (np.uint64(kind_id) * np.uint64(97531))
    ) & np.uint64(0xFFFFFFFF)

    vals = (h % np.uint64(15)).astype(np.int16) - 7
    sparse_mask = (h % np.uint64(100)) < np.uint64(sparsity_pct)
    vals[sparse_mask] = 0
    return vals.astype(np.int8, copy=False)


def _bias_vector(n_dim: int, kind_id: int, seed: int) -> np.ndarray:
    n_idx = np.arange(n_dim, dtype=np.uint64)
    h = (
        np.uint64(seed)
        ^ (n_idx * np.uint64(2246822519))
        ^ (np.uint64(kind_id) * np.uint64(3266489917))
    ) & np.uint64(0xFFFFFFFF)
    return ((h % np.uint64(17)).astype(np.int32) - 8).astype(np.int32, copy=False)


def _write_autotune_meta(path: Path, workload_tag: str, kind_id: int, m_dim: int, n_dim: int,
                         k_dim: int, sparsity_pct: int, seed: int, activation: int) -> None:
    lines = [
        "#ifndef AUTOTUNE_WORKLOAD_H",
        "#define AUTOTUNE_WORKLOAD_H",
        "",
        f'#define AUTOTUNE_WORKLOAD_TAG "{workload_tag}"',
        f"#define AUTOTUNE_WORKLOAD_KIND {kind_id}u",
        f"#define AUTOTUNE_M {m_dim}u",
        f"#define AUTOTUNE_N {n_dim}u",
        f"#define AUTOTUNE_K {k_dim}u",
        f"#define AUTOTUNE_SPARSITY_PCT {sparsity_pct}u",
        f"#define AUTOTUNE_SEED {seed}u",
        f"#define AUTOTUNE_ACTIVATION {activation}u",
        "",
        "#endif",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export synthetic workload model for RL autotuning.")
    ap.add_argument("--workload-tag", required=True)
    ap.add_argument("--kind-id", type=int, required=True)
    ap.add_argument("--m", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--sparsity-pct", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--activation", type=int, choices=[0, 1], required=True)
    args = ap.parse_args()

    fw_include = _repo_root / "firmware" / "include"
    fw_include.mkdir(parents=True, exist_ok=True)

    n_dim = int(args.n)
    k_dim = int(args.k)
    m_dim = int(args.m)
    kind_id = int(args.kind_id)
    sparsity_pct = int(args.sparsity_pct)
    seed = int(args.seed) & 0xFFFFFFFF
    activation = ACT_RELU if int(args.activation) == 1 else ACT_NONE

    w = _weight_matrix(n_dim=n_dim, k_dim=k_dim, kind_id=kind_id,
                       sparsity_pct=sparsity_pct, seed=seed)
    b = _bias_vector(n_dim=n_dim, kind_id=kind_id, seed=seed)

    layers = [{
        "M": m_dim,
        "N": n_dim,
        "K": k_dim,
        "W": w,
        "B": b,
        "activation": activation,
        "weight_scale": (1 << 16),
        "workload_tag": str(args.workload_tag),
    }]

    blob = build_model_blob(layers=layers, input_size=k_dim, output_size=n_dim)
    model_header_path = fw_include / "model_blob.h"
    model_bin_path = fw_include / "model.bin"
    meta_header_path = fw_include / "autotune_workload.h"

    model_bin_path.write_bytes(blob)
    write_model_blob_header(blob, model_header_path)
    _write_autotune_meta(
        path=meta_header_path,
        workload_tag=str(args.workload_tag),
        kind_id=kind_id,
        m_dim=m_dim,
        n_dim=n_dim,
        k_dim=k_dim,
        sparsity_pct=sparsity_pct,
        seed=seed,
        activation=int(args.activation),
    )

    print(
        f"[AutotuneExport] workload={args.workload_tag} "
        f"MxNxK={m_dim}x{n_dim}x{k_dim} sparsity={sparsity_pct}% activation={int(args.activation)}"
    )


if __name__ == "__main__":
    main()

