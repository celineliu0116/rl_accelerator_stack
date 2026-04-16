#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List
try:
    import fcntl
except Exception:
    fcntl = None


FEATURE_ORDER: List[str] = [
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
    "tile_m",
    "tile_n",
    "burst_size",
    "prefetch_depth",
    "tile_b",
    "hardware_dataflow_mode",
]

TRACE_SCHEMA_VERSION = 1
DATAFLOW_MODE_NAMES = {
    0: "DENSE",
    1: "SPARSE_ISECT",
    2: "HIGHLY_SPARSE",
}


def ensure_parent(path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


def append_trace(path: str, record: Dict[str, Any]) -> None:
    ensure_parent(path)
    row = normalize_trace_row(record)
    row.setdefault("timestamp", time.time())
    with open(path, "a", encoding="utf-8") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_traces(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rows.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return rows


def estimate_dma_bytes(m_dim: int, n_dim: int, k_dim: int) -> int:
    # Dense systolic transfer estimate: each (k/4, n/4) chunk is a 256-bit beat.
    kp = ((int(k_dim) + 3) // 4) * 4
    npad = ((int(n_dim) + 3) // 4) * 4
    return int((kp // 4) * (npad // 4) * 32)


def estimate_macs(m_dim: int, n_dim: int, k_dim: int) -> int:
    return int(m_dim) * int(n_dim) * int(k_dim)


def estimate_pe_util(cycles: int, m_dim: int, n_dim: int, k_dim: int, pe_count: int = 16) -> float:
    if cycles <= 0:
        return -1.0
    macs = estimate_macs(m_dim, n_dim, k_dim)
    util = float(macs) / float(max(1, cycles) * max(1, pe_count))
    if util < 0.0:
        return 0.0
    if util > 1.0:
        return 1.0
    return util


def vectorize_features(record: Dict[str, Any]) -> List[float]:
    return [float(record.get(k, 0.0)) for k in FEATURE_ORDER]


def normalize_trace_row(record: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(record)
    row.setdefault("schema_version", TRACE_SCHEMA_VERSION)
    row.setdefault("run_id", "")
    row.setdefault("run_mode", "")
    row.setdefault("campaign_id", "")
    row.setdefault("campaign_stage", "")
    row.setdefault("campaign_targeted", 0)
    row.setdefault("campaign_target_reason", "")
    row.setdefault("campaign_target_bucket", "")
    row.setdefault("campaign_target_weight", 0.0)
    row.setdefault("target_key", "")
    try:
        sp = int(row.get("sparsity_pct", 0))
    except Exception:
        sp = 0
    row.setdefault("sparsity_bucket", max(0, min(10, sp // 10)))
    row.setdefault("mode1_candidate", 0)

    executed_hw_mode = int(row.get("executed_hardware_dataflow_mode", row.get("hardware_dataflow_mode", -1)))
    row.setdefault("executed_hardware_dataflow_mode", executed_hw_mode)
    hw_mode = int(row.get("hardware_dataflow_mode", executed_hw_mode))
    row.setdefault("hardware_dataflow_mode", executed_hw_mode if hw_mode < 0 else hw_mode)
    proposed_hw_mode = int(row.get("proposed_hardware_dataflow_mode", executed_hw_mode))
    row.setdefault("proposed_hardware_dataflow_mode", proposed_hw_mode)
    row.setdefault("mode_name", DATAFLOW_MODE_NAMES.get(executed_hw_mode, "UNKNOWN"))
    row.setdefault("executed_mode_name", DATAFLOW_MODE_NAMES.get(executed_hw_mode, "UNKNOWN"))
    row.setdefault("proposed_mode_name", DATAFLOW_MODE_NAMES.get(proposed_hw_mode, "UNKNOWN"))
    row.setdefault("reward_raw", float(row.get("reward", -10000.0)))

    violations = row.get("constraint_violations")
    if not isinstance(violations, list):
        violations = []
    row["constraint_violations"] = violations

    reject_reason = None
    if violations:
        reject_reason = str(violations[0])
    elif row.get("error"):
        reject_reason = str(row.get("error"))
    elif int(row.get("correctness_passed", -1)) == 0:
        reject_reason = "correctness_failed"
    row["reject_reason"] = reject_reason

    cycles = int(row.get("cycles", -1))
    is_valid = int(
        int(row.get("simulated", 0)) == 1
        and int(row.get("correctness_passed", 0)) == 1
        and cycles > 0
    )
    row["is_valid"] = is_valid
    row.setdefault("valid_simulated", is_valid)
    row.setdefault("dense_equiv_macs_per_cycle", -1.0)
    return row
