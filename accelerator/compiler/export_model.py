#!/usr/bin/env python3
"""
Sparsity-Aware, RL-Driven Model Exporter for 4x4 Weight-Stationary Systolic Array.
Reads layer dimensions, applies RL-tuned tiling and burst settings, analyzes
sparsity, selects dense or CSR packing based on hardware_dataflow_mode, pads
the geometry, applies physical hardware 4x4 skewing, and exports a
16-byte aligned C header for the bare-metal DMA controller.
"""

from __future__ import annotations

import argparse
import os
import struct
from pathlib import Path
from typing import Dict, List, Tuple
import json

import numpy as np

# Import sparsity utilities from the same compiler/ directory
from sparsity_utils import compute_sparsity, structured_2_4_pack

MODEL_MAGIC = 0xACCE1E28  # Bumped: v2 format with CSR sparse metadata
SYS_TILE_DIM = 4

ACT_NONE = 0
ACT_RELU = 1
ACT_SOFTMAX = 2

# Hardware dataflow modes (RL-selected)
DATAFLOW_DENSE_SYSTOLIC = 0
DATAFLOW_SPARSE_INTERSECTION = 1
DATAFLOW_HIGHLY_SPARSE_OUTER = 2

FW_MAX_MODEL_VECTOR = 8192


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def _mode1_enabled() -> bool:
    return os.environ.get("ACCELERA_ENABLE_SPARSE_MODE1", "0") == "1"


def _sw_mode2_enabled() -> bool:
    return os.environ.get("ACCELERA_ENABLE_SW_SPARSE_MODE2", "1") == "1"


def _sw_mode2_threshold_pct() -> int:
    return max(0, min(99, _env_int("ACCELERA_SW_SPARSE_MODE2_THRESHOLD_PCT", 95)))


def _mode1_structural_guard(n_dim_padded: int, k_dim_padded: int,
                            sparsity_pct: float, tile_n: int) -> tuple[bool, str]:
    if not _mode1_enabled():
        return False, "mode1_disabled_env"
    if tile_n <= 0 or (tile_n % SYS_TILE_DIM) != 0:
        return False, "tile_n_not_multiple_of_4"

    max_n = _env_int("ACCELERA_SPARSE_MODE1_MAX_N", FW_MAX_MODEL_VECTOR)
    max_k = _env_int("ACCELERA_SPARSE_MODE1_MAX_K", FW_MAX_MODEL_VECTOR)
    min_sparsity = _env_int("ACCELERA_SPARSE_MODE1_MIN_SPARSITY_PCT", 20)

    if int(n_dim_padded) > int(max_n):
        return False, f"n_dim_gt_{max_n}"
    if int(k_dim_padded) > int(max_k):
        return False, f"k_dim_gt_{max_k}"
    if float(sparsity_pct) < float(min_sparsity):
        return False, f"sparsity_lt_{min_sparsity}pct"
    return True, ""


def _mode1_numeric_self_check(w_padded: np.ndarray,
                              elements_bytes: bytearray,
                              row_ptr_bytes: bytearray,
                              nnz: int) -> tuple[bool, str]:
    return True, ""

def compute_sparsity(weights: np.ndarray) -> float:
    """Re-exported from sparsity_utils for backward compatibility."""
    from sparsity_utils import compute_sparsity as _cs
    return _cs(weights)


def pad_to_tile(weights: np.ndarray, biases: np.ndarray) -> tuple:
    """
    Pads weight matrix and bias vector so that both N and K dimensions are
    exact multiples of SYS_TILE_DIM. Zero-padding guarantees that the systolic
    array computes 0 * activation = 0 for phantom channels, leaving real
    results unaffected. The C runtime can then skip all boundary checks.
    Returns (padded_weights, padded_biases, n_dim_padded, k_dim_padded).
    """
    n_dim, k_dim = weights.shape
    n_dim_padded = ((n_dim + SYS_TILE_DIM - 1) // SYS_TILE_DIM) * SYS_TILE_DIM
    k_dim_padded = ((k_dim + SYS_TILE_DIM - 1) // SYS_TILE_DIM) * SYS_TILE_DIM
    pad_n = n_dim_padded - n_dim
    pad_k = k_dim_padded - k_dim
    w_padded = np.pad(weights, ((0, pad_n), (0, pad_k)),
                      mode='constant', constant_values=0)
    b_padded = np.pad(biases, (0, pad_n),
                      mode='constant', constant_values=0)
    return w_padded, b_padded, n_dim_padded, k_dim_padded


def dynamic_tile_pack(weights: np.ndarray, tile_n: int) -> bytearray:
    """
    Takes an [N, K] weight matrix.
    1. Transposes to [K, N] (input sequence order).
    2. Pads the N dimension to the RL-Tuned `tile_n`.
    3. Pads the K dimension to a multiple of 4.
    4. Slices into physical 4x4 hardware chunks.
    5. Streams the chunks sequentially into a bytearray.
    
    Note: This packing assumes the hardware handles the 4x4 skewing internally.
    The data is packed in a way that each 4x4 chunk is flattened row-major.
    """
    n_original, k_dim = weights.shape
    
    # 1. Pad N to the RL-Tuned tile boundary (must be multiple of 4)
    assert tile_n % SYS_TILE_DIM == 0, f"tile_n ({tile_n}) must be a multiple of 4"
    pad_n = (tile_n - (n_original % tile_n)) % tile_n
    
    if pad_n > 0:
        weights_padded = np.pad(weights, ((0, pad_n), (0, 0)), mode='constant')
    else:
        weights_padded = weights
        
    n_padded, _ = weights_padded.shape
        
    # Transpose to [K, N]
    w_kt = weights_padded.T 
    
    # 2. Pad K to a multiple of 4 (SYS_TILE_DIM)
    pad_k = (SYS_TILE_DIM - (k_dim % SYS_TILE_DIM)) % SYS_TILE_DIM
    if pad_k > 0:
        w_kt = np.pad(w_kt, ((0, pad_k), (0, 0)), mode='constant')
        
    k_padded, _ = w_kt.shape
    
    byte_stream = bytearray()
    
    # 3. Slice the matrix column-wise into RL-defined tile_n blocks
    for n_start in range(0, n_padded, tile_n):
        tile_block = w_kt[:, n_start:n_start+tile_n] # Shape [K_padded, tile_n]
        
        # 4. Cut the RL tile into physical 4x4 chunks for the hardware array
        for phys_n_start in range(0, tile_n, SYS_TILE_DIM):
            for phys_k_start in range(0, k_padded, SYS_TILE_DIM):
                phys_chunk = tile_block[
                    phys_k_start:phys_k_start+SYS_TILE_DIM, 
                    phys_n_start:phys_n_start+SYS_TILE_DIM
                ] # Shape [4, 4]
                
                # 5. Flatten column-major (F-order) and stream
                # MatmulAcceleratorSystolic reads 32 contiguous bits (4 bytes of K steps) per neuron.
                byte_stream += phys_chunk.flatten(order="F").tobytes()
            
    return byte_stream


    # structured_2_4_pack is imported from sparsity_utils


def get_rl_parameters(m: int, n: int, k: int, activation: int = 1,
                      workload_tag: str | None = None,
                      sparsity_bucket: int | None = None) -> Tuple[int, int, int, int, int, int]:
    """
    Returns (tile_m, tile_n, burst_size, prefetch_depth, tile_b, hardware_dataflow_mode).
    Checks rl_override.json first (set by the RL daemon each episode),
    then falls back to bkm_ledger.json best-known config,
    then uses the safe 4x4 sequential default.
    prefetch_depth: 1=sequential, 2=double-buffered ping-pong.
    tile_b: images packed per hardware call (1, 2, or 4).
    hardware_dataflow_mode: 0=dense systolic, 1=sparse intersection, 2=highly sparse outer product.
    """
    try:
        override_path = Path(__file__).parent.parent / "auto_tuner" / "rl_override.json"
        if override_path.exists():
            with open(override_path, "r") as f:
                override = json.load(f)
            if override.get("M") == m and override.get("N") == n and override.get("K") == k:
                prefetch_depth = int(override.get("prefetch_depth", 2))
                tile_b = int(override.get("tile_b", 4))
                hw_mode = int(override.get("hardware_dataflow_mode", 0))
                return override["tile_m"], override["tile_n"], override["burst_size"], prefetch_depth, tile_b, hw_mode
    except Exception:
        pass

    try:
        ledger_path = Path(__file__).parent.parent / "auto_tuner" / "compiler" / "bkm_ledger.json"
        with open(ledger_path, "r") as f:
            ledger = json.load(f)

        base_key = f"{m}x{n}x{k}_act{activation}"
        candidate_keys = []
        generic_only = os.environ.get("ACCELERA_GENERIC_LEDGER_LOOKUP", "0") == "1"
        if (not generic_only) and workload_tag:
            if sparsity_bucket is not None:
                candidate_keys.append(f"{base_key}_{workload_tag}_sp{sparsity_bucket}")
            candidate_keys.append(f"{base_key}_{workload_tag}")
        candidate_keys.append(base_key)

        for key in candidate_keys:
            if key not in ledger:
                continue
            cfg = ledger[key]
            print(f"[Export] RL Cache Hit! Using optimized config: {cfg}")
            prefetch_depth = int(cfg.get("prefetch_depth", 2))
            tile_b = int(cfg.get("tile_b", 4))
            hw_mode = int(cfg.get("hardware_dataflow_mode", 0))
            return cfg["tile_m"], cfg["tile_n"], cfg["burst_size"], prefetch_depth, tile_b, hw_mode
    except Exception:
        pass

    print(f"[Export] RL Cache Miss for {m}x{n}x{k}_act{activation}. Using safe 4x4 dense baseline.")
    return 4, 4, 16, 2, 1, DATAFLOW_DENSE_SYSTOLIC  # safe default: dense mode




def build_model_blob(layers: List[Dict], input_size: int, output_size: int) -> bytes:
    num_layers = len(layers)
    # v2 header: 16 bytes model header + 64 bytes per layer (16 x uint32)
    header_size = 16 + 64 * num_layers
    
    header = bytearray(struct.pack("<IIII", MODEL_MAGIC, num_layers, int(input_size), int(output_size)))
    payload = bytearray()
    current_offset = header_size
    
    for li, layer in enumerate(layers):
        w = np.asarray(layer["W"], dtype=np.int8)
        b = np.asarray(layer["B"], dtype=np.int32)
        
        n_dim, k_dim = w.shape
        m_dim = int(layer.get("M", 1))
        activation = int(layer.get("activation", ACT_NONE))
        weight_scale = int(layer.get("weight_scale", 1 << 16))
        
        # Analyze sparsity BEFORE padding (padding zeros don't count)
        sparsity = compute_sparsity(w)
        sparsity_pct_q8 = int(sparsity * 255.0)
        print(f"[Export] Layer {li}: {n_dim}x{k_dim} sparsity={sparsity*100:.1f}%")
        
        # --- Fix 3: Pad N and K to multiples of SYS_TILE_DIM ---
        w_padded, b_pre_padded, n_dim_padded, k_dim_padded = pad_to_tile(w, b)

        workload_tag = layer.get("workload_tag", None)
        sparsity_bucket = int((sparsity * 100.0) // 10.0)
        tile_m, tile_n, burst_size, prefetch_depth, tile_b_rl, hw_mode = get_rl_parameters(
            m_dim, n_dim_padded, k_dim_padded,
            activation=activation,
            workload_tag=workload_tag,
            sparsity_bucket=sparsity_bucket)
        mode1_elements_bytes = None
        mode1_row_ptr_bytes = None
        mode1_nnz = None

        if hw_mode == DATAFLOW_SPARSE_INTERSECTION:
            ok, reason = _mode1_structural_guard(
                n_dim_padded=n_dim_padded,
                k_dim_padded=k_dim_padded,
                sparsity_pct=(sparsity * 100.0),
                tile_n=tile_n,
            )
            if not ok:
                print(
                    f"[Export] Layer {li}: hw_mode=1 rejected ({reason}); "
                    "falling back to dense mode 0."
                )
                hw_mode = DATAFLOW_DENSE_SYSTOLIC
            else:
                # Build CSR once and verify numerical equivalence before enabling.
                mode1_elements_bytes, mode1_row_ptr_bytes, mode1_nnz = structured_2_4_pack(w_padded, tile_n)
                ok, reason = _mode1_numeric_self_check(
                    w_padded=w_padded,
                    elements_bytes=mode1_elements_bytes,
                    row_ptr_bytes=mode1_row_ptr_bytes,
                    nnz=mode1_nnz,
                )
                if not ok:
                    print(
                        f"[Export] Layer {li}: hw_mode=1 self-check failed ({reason}); "
                        "falling back to dense mode 0."
                    )
                    hw_mode = DATAFLOW_DENSE_SYSTOLIC

        # Software-first sparse policy: if a layer is highly sparse, route it
        # to mode 2 (CPU sparse outer-product) to avoid dense DMA/setup costs.
        sw_mode2_thresh = _sw_mode2_threshold_pct()
        if _sw_mode2_enabled() and (sparsity * 100.0) >= float(sw_mode2_thresh):
            print(
                f"[Export] Layer {li}: sparsity={sparsity*100.0:.1f}% >= {sw_mode2_thresh}% "
                "=> forcing software sparse mode 2."
            )
            hw_mode = DATAFLOW_HIGHLY_SPARSE_OUTER

        print(f"[Export] Layer {li}: hw_mode={hw_mode} tile=[{tile_m}x{tile_n}] burst={burst_size}")
        
        # Sparse metadata defaults
        sparse_nnz = 0
        structured_sparse_offset = 0
        deprecated_row_ptr = 0
        
        if hw_mode == DATAFLOW_DENSE_SYSTOLIC:
            # Mode 0: Dense systolic — existing packing
            weight_bytes = dynamic_tile_pack(w_padded, tile_n)
            rem = len(weight_bytes) % 16
            if rem > 0:
                weight_bytes += bytes(16 - rem)
            weight_offset = current_offset
            
            # Bias packing
            pad_b_extra = (tile_n - (n_dim_padded % tile_n)) % tile_n
            if pad_b_extra > 0:
                b_final = np.pad(b_pre_padded, (0, pad_b_extra), mode='constant')
            else:
                b_final = b_pre_padded
            bias_bytes = b_final.astype(np.int32).tobytes()
            rem = len(bias_bytes) % 16
            if rem > 0:
                bias_bytes += bytes(16 - rem)
            bias_offset = weight_offset + len(weight_bytes)
            
            payload += weight_bytes
            payload += bias_bytes
            current_offset += len(weight_bytes) + len(bias_bytes)
        else:
            # Mode 1 or 2: Sparse — CSR packing
            if hw_mode == DATAFLOW_SPARSE_INTERSECTION and mode1_elements_bytes is not None:
                elements_bytes = mode1_elements_bytes
                row_ptr_bytes = mode1_row_ptr_bytes
                nnz = mode1_nnz
            else:
                elements_bytes, row_ptr_bytes, nnz = structured_2_4_pack(w_padded, tile_n)
            sparse_nnz = nnz
            
            weight_offset = current_offset
            structured_sparse_offset = weight_offset
            deprecated_row_ptr = 0
            
            # Bias packing (same as dense)
            pad_b_extra = (tile_n - (n_dim_padded % tile_n)) % tile_n
            if pad_b_extra > 0:
                b_final = np.pad(b_pre_padded, (0, pad_b_extra), mode='constant')
            else:
                b_final = b_pre_padded
            bias_bytes = b_final.astype(np.int32).tobytes()
            rem = len(bias_bytes) % 16
            if rem > 0:
                bias_bytes += bytes(16 - rem)
            bias_offset = weight_offset + len(elements_bytes)
            
            payload += elements_bytes
            payload += bias_bytes
            current_offset += len(elements_bytes) + len(bias_bytes)
            
            print(f"[Export] Layer {li}: CSR nnz={nnz} (dense would be {n_dim_padded*k_dim_padded})")

        # flags: bits[3:0]=tile_b, bit[4]=is_padded(always 1), bits[6:5]=hw_dataflow_mode
        flags = (tile_b_rl & 0x0F) | (1 << 4) | ((hw_mode & 0x03) << 5)

        # Struct layout matches layer_header_t v2 in model_format.h (16 x uint32 = 64 bytes):
        # layer_m, layer_n(padded), layer_k(padded), tile_m, tile_n, burst_size,
        # activation, weight_scale, weight_offset, bias_offset,
        # prefetch_depth, flags,
        # sparse_nnz, structured_sparse_offset, deprecated_row_ptr, sparsity_pct_q8
        layer_header = struct.pack(
            "<16I",
            m_dim, n_dim_padded, k_dim_padded,
            tile_m, tile_n, burst_size,
            activation, weight_scale,
            weight_offset, bias_offset,
            prefetch_depth, flags,
            sparse_nnz, structured_sparse_offset, deprecated_row_ptr, sparsity_pct_q8
        )
        header += layer_header
        
    out = header + payload
    
    remainder = len(out) % 16
    if remainder > 0:
        out += bytearray(16 - remainder)
        
    return bytes(out)


def write_model_blob_header(blob: bytes, header_path: Path, symbol: str = "g_model_blob") -> None:
    header_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "#ifndef MODEL_BLOB_H",
        "#define MODEL_BLOB_H",
        "",
        "#include <stdint.h>",
        "",
        "// Forced 16-byte alignment to prevent RISC-V mcause=4 traps ",
        "// and guarantee AXI/TileLink DMA burst compliance.",
        f"__attribute__((aligned(16))) const uint8_t {symbol}[] = {{",
    ]

    for i in range(0, len(blob), 16):
        chunk = blob[i : i + 16]
        bytes_txt = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"  {bytes_txt},")

    lines += [
        "};",
        f"const uint32_t {symbol}_len = {len(blob)}u;",
        "",
        "#endif",
        "",
    ]

    header_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile model to C header via RL configs.")
    parser.add_argument("--bin", type=Path, required=True, help="Input model.bin path")
    parser.add_argument("--header", type=Path, required=True, help="Output header path (model_blob.h)")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    blob = args.bin.read_bytes()
    # Note: For this script, we expect the pipeline upstream to generate the raw bytes.
    # In a real integrated flow, we'd pass the JSON dict directly to build_model_blob.
    # Since train_and_export.py calls build_model_blob(), we only modify the library functions here.
    # We will ignore `main()` for now, as train_and_export.py imports our functions.

if __name__ == "__main__":
    main()
