#!/usr/bin/env python3
"""
Sparsity analysis and CSR compression utilities for the Accelera compiler.
Extracted from export_model.py to separate concerns.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

SYS_TILE_DIM = 4


def compute_sparsity(weights: np.ndarray) -> float:
    """Returns the fraction of zero elements in the weight matrix (0.0–1.0)."""
    total = weights.size
    if total == 0:
        return 0.0
    return float(np.count_nonzero(weights == 0)) / total


def structured_2_4_pack(weights: np.ndarray, tile_n: int) -> Tuple[bytearray, bytearray, int]:
    """
    Converts a padded [N, K] INT8 weight matrix into 2:4 Structured Sparsity format.
    For every 1x4 block of weights (along K), keeps the top 2 elements by absolute magnitude.
    
    Each 1x4 block becomes 2 bytes of weights and 1 byte of indices:
    Packed format per 4 K-elements (for a single N row):
    - 2 nonzero INT8 weights (16 bits)
    - 2 2-bit column indices (4 bits, padded to 8 bits)
    Total: 24 bits (3 bytes), but we will pack them into uint32 for simple 16-byte aligned DMA.
    Specifically, we will interleave elements such that the hardware reads 64-bits of metadata+weights 
    to drive 128-bits of X processing.
    
    Let's use a simpler 32-bit layout per 4-element block:
    [Pad(8) | idx1(2), idx0(2), Pad(4) | val1(8) | val0(8)]
    This gives 32 bits per block, which means a 50% compression over dense (which is 4 bytes).
    
    Returns (elements_bytes, bytearray(), nnz):
      - elements_bytes: 32-bit packed blocks, 16-byte aligned
      - nnz: total number of non-zero elements retained
    """
    n_original, k_dim = weights.shape

    # Pad N to tile_n boundary (same as dynamic_tile_pack)
    assert tile_n % SYS_TILE_DIM == 0, f"tile_n ({tile_n}) must be a multiple of 4"
    pad_n = (tile_n - (n_original % tile_n)) % tile_n
    if pad_n > 0:
        weights_padded = np.pad(weights, ((0, pad_n), (0, 0)), mode='constant')
    else:
        weights_padded = weights
    n_padded = weights_padded.shape[0]

    # Pad K to multiple of 4 (for 2:4 blocks) and then further pad if needed
    pad_k = (4 - (k_dim % 4)) % 4
    if pad_k > 0:
        weights_padded = np.pad(weights_padded, ((0, 0), (0, pad_k)), mode='constant')
    k_padded = weights_padded.shape[1]

    elements = []
    nnz = 0
    
    w_uint32 = []
    
    # Process each 4-neuron physical tile together to find a shared sparsity mask
    for n_start in range(0, n_padded, 4):
        for k_blk in range(0, k_padded, 4):
            block_4x4 = weights_padded[n_start:n_start+4, k_blk:k_blk+4]
            # Sum magnitudes across the 4 neurons (N dimension) to find the most important 2 K-steps
            magnitudes_sum = np.sum(np.abs(block_4x4), axis=0) # shape: (4,)
            top2_idx = np.argsort(magnitudes_sum)[-2:]
            top2_idx = np.sort(top2_idx)
            
            idx0 = top2_idx[0]
            idx1 = top2_idx[1]
            
            for phys_n in range(4):
                val0 = int(block_4x4[phys_n, idx0]) & 0xFF
                val1 = int(block_4x4[phys_n, idx1]) & 0xFF
                
                if block_4x4[phys_n, idx0] != 0: nnz += 1
                if block_4x4[phys_n, idx1] != 0: nnz += 1
                
                metadata = ((idx1 & 0x3) << 2) | (idx0 & 0x3)
                packed = (metadata << 16) | (val1 << 8) | val0
                w_uint32.append(packed)

    w_uint32 = np.array(w_uint32, dtype=np.uint32).reshape(n_padded // 4, k_padded // 4, 4)
    w_kt_u32 = w_uint32.transpose((1, 0, 2))  # Shape [K_padded // 4, n_padded // 4, 4_neurons]
    w_kt_u32 = w_kt_u32.reshape(k_padded // 4, n_padded) # Shape [K_padded // 4, n_padded]
    
    byte_stream = bytearray()
    
    # Slice the matrix column-wise into RL-defined tile_n blocks
    for n_start in range(0, n_padded, tile_n):
        tile_block = w_kt_u32[:, n_start:n_start+tile_n] # Shape [K_padded // 4, tile_n]
        
        # Cut the RL tile into physical 1x4 chunks (since 1 K-block = 4 K-steps)
        for phys_n_start in range(0, tile_n, SYS_TILE_DIM):
            for phys_k_blk in range(0, k_padded // 4):
                # We take 1 K_block and SYS_TILE_DIM (4) neurons
                phys_chunk = tile_block[
                    phys_k_blk:phys_k_blk+1, 
                    phys_n_start:phys_n_start+SYS_TILE_DIM
                ] # Shape [1, 4]
                # Flatten column-major (F-order)
                byte_stream += phys_chunk.flatten(order="F").tobytes()

    # 16-byte alignment padding
    rem = len(byte_stream) % 16
    if rem > 0:
        byte_stream += bytearray(16 - rem)

    return byte_stream, bytearray(), nnz
