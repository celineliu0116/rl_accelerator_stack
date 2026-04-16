#ifndef MODEL_FORMAT_H
#define MODEL_FORMAT_H

#include <stdint.h>

#define MODEL_MAGIC 0xACCE1E28u // v2: sparse CSR metadata support

// Standard 4x4 Systolic Array Data Pipeline Config
#define SYS_TILE_DIM 4u

#define ACT_NONE 0u
#define ACT_RELU 1u
#define ACT_SOFTMAX 2u

// Hardware dataflow modes (RL-selected)
#define DATAFLOW_DENSE_SYSTOLIC 0u
#define DATAFLOW_SPARSE_INTERSECTION 1u
#define DATAFLOW_HIGHLY_SPARSE_OUTER 2u

// Structs must be strictly packed to guarantee exact matching
// between the Python struct.pack() binary output and the C memory footprint.
// We explicitly use uint32_t to maintain native 32-bit RISC-V alignment.

typedef struct __attribute__((packed)) {
  uint32_t layer_m; // Layer spatial height (batch rows)
  uint32_t layer_n; // Output channels — padded to multiple of SYS_TILE_DIM when
                    // is_padded=1
  uint32_t layer_k; // Input channels  — padded to multiple of SYS_TILE_DIM when
                    // is_padded=1
  uint32_t tile_m;  // RL-Tuned: Hardware tile spatial height
  uint32_t tile_n;  // RL-Tuned: Hardware tile spatial width
  uint32_t burst_size;    // RL-Tuned: DMA chunk size in bytes
  uint32_t activation;    // ACT_NONE, ACT_RELU, etc.
  uint32_t weight_scale;  // Quantization scale (Q16 format)
  uint32_t weight_offset; // Byte offset from start of blob to the skewed
                          // weights (dense mode) or CSR values (sparse mode)
  uint32_t bias_offset;   // Byte offset from start of blob to the biases
  uint32_t
      prefetch_depth; // RL-Tuned: 1=sequential, 2=double-buffered ping-pong
  // flags bit layout:
  //   bits [3:0]  tile_b              — batch tile (1, 2, or 4 images per HW
  //   call) bit  [4]    is_padded           — 1: layer_n/layer_k are padded
  //   multiples of
  //                                     SYS_TILE_DIM; C drops all boundary
  //                                     checks
  //   bits [6:5]  hardware_dataflow_mode — 0=dense systolic, 1=sparse
  //   intersection,
  //                                        2=highly sparse outer product
  //   bits [31:7] reserved
  uint32_t flags;
  // --- v2 Structured Sparse 2:4 metadata (4 × uint32 = 16 bytes) ---
  uint32_t sparse_nnz; // Number of non-zero elements (0 for dense mode)
  uint32_t structured_sparse_offset; // Byte offset to 2:4 compressed block
                                     // array. Aliases weight_offset.
  uint32_t deprecated_row_ptr;       // Unused in 2:4 structured sparsity.
  uint32_t sparsity_pct_q8; // Sparsity × 255 (0=fully dense, 255=fully sparse)
} layer_header_t;

#define LAYER_TILE_B(l) ((l)->flags & 0x0Fu)
#define LAYER_IS_PADDED(l) (((l)->flags >> 4u) & 0x01u)
#define LAYER_DATAFLOW_MODE(l) (((l)->flags >> 5u) & 0x03u)

typedef struct __attribute__((packed)) {
  uint32_t magic;       // Magic word to verify binary integrity (0xACCE1E28)
  uint32_t num_layers;  // Total layers in the network
  uint32_t input_size;  // Number of 8-bit input elements (vector length)
  uint32_t output_size; // Number of 32-bit output elements
  // We use a zero-length array to map any number of layers attached
  // sequentially to the end of the base header.
  layer_header_t layers[];
} model_header_t;

// Verification Check:
// layer_header_t = 16 * 4 bytes = 64 bytes (divisible by 16)
// model_header_t (base) = 4 * 4 bytes = 16 bytes (divisible by 16)
// Total model_header_t size = 16 + 64 = 80 bytes.
// This guarantees that weight_offset (starting at byte 80) is 16-byte aligned
// for AXI/TileLink DMA burst fetch safety.

#endif // MODEL_FORMAT_H
