from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import numpy as np


@dataclass(frozen=True)
class WorkloadSpec:
    tag: str
    op_type_id: int
    kind_id: int
    m_dim: int
    n_dim: int
    k_dim: int
    sparsity_pct: int
    activation: int  # 0=none, 1=relu
    batch_size: int
    seq_len: int
    channels: int
    kernel_h: int
    kernel_w: int


# All dimensions are multiples of 4 to match systolic packing assumptions.
WORKLOAD_LIBRARY: Dict[str, List[WorkloadSpec]] = {
    "gemm": [
        WorkloadSpec("gemm", 0, 0, 64, 128, 128, 5, 0, 16, 1, 1, 1, 1),
        WorkloadSpec("gemm", 0, 0, 128, 256, 128, 8, 0, 32, 1, 1, 1, 1),
        WorkloadSpec("gemm", 0, 0, 256, 128, 192, 10, 0, 32, 1, 1, 1, 1),
    ],
    "sparse_mlp": [
        WorkloadSpec("sparse_mlp", 1, 1, 64, 512, 256, 70, 1, 64, 1, 1, 1, 1),
        WorkloadSpec("sparse_mlp", 1, 1, 128, 768, 256, 80, 1, 64, 1, 1, 1, 1),
        WorkloadSpec("sparse_mlp", 1, 1, 256, 1024, 128, 85, 1, 128, 1, 1, 1, 1),
    ],
    # Convolution lowered to GEMM (im2col): M=OH*OW, N=C_out, K=C_in*R*S.
    "convolution": [
        WorkloadSpec("convolution", 2, 2, 196, 64, 288, 15, 1, 8, 196, 32, 3, 3),   # 14x14, Cin=32, 3x3
        WorkloadSpec("convolution", 2, 2, 64, 128, 576, 20, 1, 8, 64, 64, 3, 3),     # 8x8, Cin=64, 3x3
        WorkloadSpec("convolution", 2, 2, 49, 256, 576, 25, 1, 4, 49, 64, 3, 3),      # 7x7, Cin=64, 3x3
    ],
    # Attention-like dense projections (Q/K/V or output projection style).
    "attention": [
        WorkloadSpec("attention", 3, 3, 64, 128, 128, 12, 0, 1, 64, 128, 1, 1),
        WorkloadSpec("attention", 3, 3, 128, 256, 256, 15, 0, 1, 128, 256, 1, 1),
        WorkloadSpec("attention", 3, 3, 256, 128, 128, 18, 0, 1, 256, 128, 1, 1),
    ],
}


FAMILY_ORDER = ["gemm", "sparse_mlp", "convolution", "attention"]
ALL_FAMILIES = tuple(FAMILY_ORDER)
SHAPE_SPLITS = ("all", "train", "test")

def parse_workload_selector(selector: str) -> List[str]:
    selector = (selector or "all").strip().lower()
    if selector in {"all", "*"}:
        return list(FAMILY_ORDER)
    families = [x.strip() for x in selector.split(",") if x.strip()]
    if not families:
        raise ValueError("workload selector is empty")
    invalid = [f for f in families if f not in WORKLOAD_LIBRARY]
    if invalid:
        raise ValueError(f"unknown workload family: {','.join(invalid)}")
    return families


def parse_shape_split(shape_split: str) -> str:
    s = (shape_split or "all").strip().lower()
    if s not in SHAPE_SPLITS:
        raise ValueError(f"unknown shape split: {shape_split}")
    return s


def _entries_for_split(entries: List[WorkloadSpec], shape_split: str) -> List[WorkloadSpec]:
    split = parse_shape_split(shape_split)
    if split == "all":
        return entries

    n = len(entries)
    if n <= 1:
        return entries

    holdout_idx = n - 1
    if split == "test":
        return [entries[holdout_idx]]
    return [entries[i] for i in range(n) if i != holdout_idx]


def candidate_workloads(selector: str = "all", shape_split: str = "all") -> List[WorkloadSpec]:
    families = parse_workload_selector(selector)
    out: List[WorkloadSpec] = []
    for fam in families:
        entries = _entries_for_split(WORKLOAD_LIBRARY[fam], shape_split)
        if not entries:
            entries = WORKLOAD_LIBRARY[fam]
        out.extend(entries)
    return out


def sample_workload(np_rng: np.random.Generator, episode_idx: int,
                    selector: str = "all",
                    shape_split: str = "all") -> Dict[str, int | str]:
    families = parse_workload_selector(selector)
    # Family-balanced sampling: pick family uniformly, then a concrete shape.
    fam = families[int(np_rng.integers(0, len(families)))]
    entries = _entries_for_split(WORKLOAD_LIBRARY[fam], shape_split)
    if not entries:
        entries = WORKLOAD_LIBRARY[fam]
    spec = entries[int(np_rng.integers(0, len(entries)))]

    # Per-episode seed to vary tensor values while keeping deterministic replay.
    synth_seed = int((episode_idx * 2654435761 + spec.kind_id * 2246822519) & 0xFFFFFFFF)

    return {
        "workload_tag": spec.tag,
        "op_type_id": int(spec.op_type_id),
        "workload_kind_id": int(spec.kind_id),
        "M": int(spec.m_dim),
        "N": int(spec.n_dim),
        "K": int(spec.k_dim),
        "sparsity_pct": int(spec.sparsity_pct),
        "activation": int(spec.activation),
        "batch_size": int(spec.batch_size),
        "seq_len": int(spec.seq_len),
        "channels": int(spec.channels),
        "kernel_h": int(spec.kernel_h),
        "kernel_w": int(spec.kernel_w),
        "seed": synth_seed,
    }
