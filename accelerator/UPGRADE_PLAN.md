# RISC-V Accelerator Upgrade Plan: "The Outer Product Engine"

## Goal
Improve performance (>2.1M cycle reduction) and capability (CNN/RNN support) by transforming the simple dot-product accelerator into a decoupled, systolic-like **Outer Product Engine**.

## 1. Architecture Overview

The new accelerator will transition from a "Row-Vector" (Inner Product) architecture to a **Block-Output-Stationary** (Outer Product) architecture.

### Current vs. New
| Feature | Current Implementation | Proposed Upgrade |
|:---|:---|:---|
| **Core Operation** | 4-Lane Dot Product ($1 \times 4$ Row * Vector) | **4x4 Outer Product** ($4 \times 1$ Col $\otimes$ $1 \times 4$ Row) |
| **MACs per Cycle** | 4 | **16** (Virtual or Physical) |
| **Arithmetic Intensity** | 0.8 IOps/Load (4 ops / 5 loads) | **2.0 IOps/Load** (16 ops / 8 loads) |
| **Accumulators** | 4 (Line Buffer) | **16** (4x4 Register Tile) |
| **Control** | CPU Stalls/Polls | **Command FIFO** (Decoupled) |
| **Memory Access** | Linear DMA | **2D Strided DMA** (for CNN patches) |

## 2. Detailed Features

### A. Phase 1: The Outer Product Core (4x4)
We will implement a 4x4 Register Tile that accumulates partial sums.
*   **Registers**: `acc[4][4]` (16 x 32-bit registers).
*   **Operation**:
    *   Load Column Vector $A$ (4 elements, e.g., 4 inputs from different batch items or 4 rows of image).
    *   Load Row Vector $B$ (4 elements, e.g., 4 weights).
    *   Compute $C_{ij} += A_i \times B_j$ for all $i,j \in [0,3]$.
This aligns perfectly with **Batched Inference** (Batch=4) or **Convolution** (4 filters x 4 spatial pixels).

### B. Phase 2: Decoupled Command Queue
Currently, the CPU waits for the accelerator. We will add a FIFO interface:
*   CPU writes commands (e.g., `SET_ADDR`, `START_OP`) to a FIFO.
*   CPU continues execution immediately.
*   Accelerator pulls commands and executes.
*   **Fence** instruction/register added to wait for completion only when needed.

### C. Phase 3: Strided DMA (CNN Support)
For CNNs, we need to fetch $3 \times 3$ patches from a 2D image.
*   Standard DMA: `addr++`
*   Strided DMA: `addr += (end_of_row ? stride_y : 1)`
This allows the accelerator to "walk" through an image without software `im2col` overhead.

### D. Phase 4: Post-Processing Unit (RNN Support)
RNNs require `Tanh` or `Sigmoid`. Deep CNNs use `ReLU`.
*   Add a pipeline stage after accumulation to perform:
    *   Quantization (Int32 -> Int8)
    *   Activation (ReLU, LUT-based Tanh)
    *   Write-back to memory.

## 3. Implementation Steps

1.  **Refactor Interface**: Implement the Command Queue (`dma_cmd_fifo`).
2.  **Build 4x4 Core**: Replace `packed_dot` logic with `outer_product_4x4` module.
3.  **Update Driver**: Rewrite `inference_bare.c` to use the new "Block" API.
4.  **Verify**: Run `make run` and check for correctness + cycle count reduction.

## 4. Feasibility Analysis
*   **Achievable?** Yes. The complexity is moderate. usage of registers increases (16 accs + buffers), but the FPGA/ASIC area is negligible for this size.
*   **Impact**: Moving from 4 MACs to 16 MACs theoretically quadruples peak throughput. Memory bandwidth improvements (2.5x efficiency) ensure we can feed it.
