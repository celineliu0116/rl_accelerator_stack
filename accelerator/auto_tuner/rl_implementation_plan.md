# Asynchronous RL Auto-Tuner Implementation Plan

## Overview
This document outlines the architecture and implementation of the "Anytime Intervention" Reinforcement Learning (RL) auto-tuner. The system acts as a background compiler optimization daemon designed for the custom bare-metal RISC-V ML SoC. It explores and optimizes DMA burst sizes, tiling strategies, memory layouts, and **hardware dataflow modes** to maximize IPC without altering the underlying Verilog hardware.

The system relies on a decoupled architecture where the RL agent populates a shared ledger, and a lightweight JIT compiler performs $O(1)$ lookups against this ledger during deployment.

**Status:** All three phases are **implemented and operational** as of 2026-03-05.

---

## Phase 1: The BKM (Best Known Method) Ledger and JIT Exporter ✅

### `bkm_ledger.json`
- **Purpose:** Shared database mapping neural network layer topologies to optimized hardware configurations.
- **Schema:**
  - **Key:** `{M}x{N}x{K}_act{activation}` (e.g., `128x128x784_act1`)
  - **Values:**
    - `tile_m`, `tile_n`, `burst_size` — tiling and DMA geometry
    - `prefetch_depth` — 1=sequential, 2=ping-pong DMA
    - `tile_b` — images per hardware call
    - `hardware_dataflow_mode` — 0=dense systolic, 1=sparse intersection, 2=highly sparse outer product
    - `ipc` — throughput proxy (`3500 / total_cycles`)
- **Constraints:** All reads and writes use file-level `fcntl` locking to prevent corruption between the daemon and compiler.

### `jit_exporter.py`
- **Purpose:** User-facing compiler frontend with O(1) ledger lookup.
- **Cache Hit:** Returns stored tiling, DMA, and dataflow mode parameters.
- **Cache Miss:** Falls back to safe configuration: `tile_m=4, tile_n=4, burst_size=16, hardware_dataflow_mode=0`.

---

## Phase 2: The Verilator Gym Environment ✅

### `SystolicEnv` (inherits from `gymnasium.Env`)
- **Observation Space (5D):**
  - $M, N, K$ dimensions of the current layer
  - Available scratchpad headroom
  - **Sparsity percentage** (0–100) — enables the agent to correlate tensor density with dataflow mode rewards
- **Action Space (`MultiDiscrete([4, 8, 3, 2, 2, 3])` = 1,152 configs):**
  - `[0]` tile_m ∈ {4, 8, 12, 16}
  - `[1]` tile_n ∈ {4, 8, 12, 16, 20, 24, 28, 32}
  - `[2]` burst_size ∈ {16, 32, 64}
  - `[3]` prefetch_depth ∈ {1, 2}
  - `[4]` tile_b ∈ {1, 2}
  - `[5]` **hardware_dataflow_mode ∈ {0, 1, 2}** — dense systolic, sparse intersection, highly sparse outer product
- **Step Function:**
  1. Agent selects a 6D action.
  2. Environment writes `rl_override.json` with all parameters including `hardware_dataflow_mode`.
  3. Calls `mnist_mlp_export.py` — the compiler analyzes sparsity and packs weights as dense tiles (mode 0) or CSR format (modes 1, 2).
  4. Triggers `make run` with Verilator simulation.
  5. Parses cycle count and PASSED/FAILED status.
- **Reward:** `1,000,000 / total_cycles` (positive for PASSED), -10,000 penalty for math corruption or timeout.

---

## Phase 3: The Interruptible RL Daemon ✅

### `rl_daemon.py`
- **Algorithm:** Stable Baselines3 PPO, `n_steps=64`, `batch_size=16`.
- **Custom Callback:** After every episode, decodes the 6D action, compares cycle count against ledger, and atomically updates if a new record is found.
- **Graceful Shutdown:** `try/except KeyboardInterrupt` → saves policy weights to `ppo_systolic_agent_latest.zip`.
- **Warm Starting:** Loads existing policy from disk if available; otherwise initializes a new random policy.

---

## Phase 4: Sparsity-Aware Compiler ✅ (added 2026-03-05)

### `export_model.py` — Sparsity Analysis and CSR Packing
- **`compute_sparsity(weights)`** — measures zero-element ratio before padding
- **`csr_tile_pack(weights, tile_n)`** — converts weight matrix to CSR format:
  - INT8 values, UINT16 column indices, UINT32 row pointers
  - All arrays 16-byte aligned for DMA compliance
- **`build_model_blob()`** — routes to dense or CSR packing based on `hardware_dataflow_mode`
- **v2 binary format** — 64-byte layer headers with `csr_nnz`, `csr_col_idx_offset`, `csr_row_ptr_offset`, `sparsity_pct_q8`

### `inference_generic.c` — 3-Mode Dispatcher
- `dispatch_dense_systolic()` — hardware-accelerated (mode 0)
- `dispatch_sparse_intersection()` — CPU-side CSR matmul (mode 1)
- `dispatch_highly_sparse_outer_product()` — CPU-side outer-product matmul (mode 2)

The RL agent learns the optimal crossover point where sparse mode overhead beats or loses to dense hardware for each layer's sparsity level.

