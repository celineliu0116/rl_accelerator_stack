// inference_bare.c — Bare-metal fixed-point MLP inference (Phase 2)

#include "test_images.h"
#include "weights.h"
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define csr_tohost(val)                                                        \
  do {                                                                         \
    asm volatile("csrw 0x51e, %[v]" ::[v] "r"(val));                           \
  } while (0)

void *memset(void *s, int c, size_t n) {
  unsigned char *p = (unsigned char *)s;
  while (n--)
    *p++ = (unsigned char)c;
  return s;
}

void *memcpy(void *dest, const void *src, size_t n) {
  unsigned char *d = (unsigned char *)dest;
  const unsigned char *s = (const unsigned char *)src;
  while (n--)
    *d++ = *s++;
  return dest;
}

// Layer Sizes
#define INPUT_SIZE 784
#define HIDDEN_SIZE 128
#define OUTPUT_SIZE 10

#ifdef ILP_MICROBENCH
#ifndef ILP_ITERS
#define ILP_ITERS 250000u
#endif

static int run_ilp_microbench(void) {
  uint32_t a0 = 0x13579bdfu;
  uint32_t a1 = 0x2468ace0u;
  uint32_t b0 = 0xdeadbeefu;
  uint32_t b1 = 0x31415927u;
  uint32_t acc0 = 0u;
  uint32_t acc1 = 0u;

  // Two mostly independent ALU streams to expose ILP to dual-issue pairing.
  for (uint32_t i = 0; i < ILP_ITERS; i++) {
    uint32_t t0 = ((a0 << 1) ^ b0) + (i * 3u + 0x9e3779b9u);
    uint32_t t1 = ((a1 << 2) ^ b1) + (i * 5u + 0x7f4a7c15u);

    acc0 += t0;
    acc1 += t1;

    a0 += (b0 ^ (t1 >> 3)) + 0x11u;
    a1 += (b1 ^ (t0 >> 2)) + 0x33u;

    b0 = (b0 << 3) ^ (a0 >> 1) ^ 0xa5a5a5a5u;
    b1 = (b1 << 5) ^ (a1 >> 2) ^ 0x5a5a5a5au;
  }

  uint32_t rot = (acc1 << 1) | (acc1 >> 31);
  uint32_t checksum = acc0 ^ rot ^ a0 ^ (a1 >> 3) ^ b0 ^ (b1 << 7);
  const uint32_t expected = 0x9c89e29du;

  if (checksum == expected) {
    csr_tohost(1);
  } else {
    csr_tohost(2);
  }

  for (;;)
    asm volatile("nop");
}
#endif

// =============================================================
// Matmul Accelerator MMIO (UPDATED MAP: results start at 0x18)
// =============================================================
#define ACCEL_BASE 0x80000000
#define ACCEL_CTRL (*(volatile uint32_t *)(ACCEL_BASE + 0x00))
#define ACCEL_STATUS (*(volatile uint32_t *)(ACCEL_BASE + 0x00))
#define ACCEL_W_ADDR (*(volatile uint32_t *)(ACCEL_BASE + 0x04))
#define ACCEL_X_ADDR (*(volatile uint32_t *)(ACCEL_BASE + 0x08))
#define ACCEL_M_DIM (*(volatile uint32_t *)(ACCEL_BASE + 0x0C))
#define ACCEL_N_DIM (*(volatile uint32_t *)(ACCEL_BASE + 0x10))
#define ACCEL_K_DIM (*(volatile uint32_t *)(ACCEL_BASE + 0x14))
#define ACCEL_X_STRIDE (*(volatile uint32_t *)(ACCEL_BASE + 0x58))
#define ACCEL_K_ROW_LEN (*(volatile uint32_t *)(ACCEL_BASE + 0x5C))

// RESULTS moved to start at 0x18 (so K_DIM at 0x14 doesn't overlap)
#define ACCEL_RESULT_BASE (ACCEL_BASE + 0x18)

static inline int32_t read_result(int row, int col) {
  volatile int32_t *addr =
      (volatile int32_t *)(ACCEL_RESULT_BASE + (row * 4 + col) * 4);
  return *addr;
}

// =============================================================
// Data Packing
// =============================================================

// Pack Inputs: 4 input vectors -> words over k
// For each k: word = [in3, in2, in1, in0] with in0 in bits[7:0]
void pack_input_batch(const int8_t **inputs, int8_t *dst, int K) {
  int32_t *dst32 = (int32_t *)dst;
  for (int k = 0; k < K; k++) {
    uint8_t b0 = (uint8_t)inputs[0][k];
    uint8_t b1 = (uint8_t)inputs[1][k];
    uint8_t b2 = (uint8_t)inputs[2][k];
    uint8_t b3 = (uint8_t)inputs[3][k];
    uint32_t packed = ((uint32_t)b3 << 24) | ((uint32_t)b2 << 16) |
                      ((uint32_t)b1 << 8) | ((uint32_t)b0);
    dst32[k] = (int32_t)packed;
  }
}

// Pack Weights for a 4-neuron block starting at n_start.
// Assumes src_weights is row-major: src_weights[n*K + k].
// For each k: word = [w3, w2, w1, w0] where wj = weight[(n_start+j), k]
void pack_weight_block(const int8_t *src_weights, int8_t *dst, int n_start,
                       int K, int N_total) {
  int32_t *dst32 = (int32_t *)dst;

  for (int k = 0; k < K; k++) {
    uint8_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;

    int n0 = n_start + 0;
    int n1 = n_start + 1;
    int n2 = n_start + 2;
    int n3 = n_start + 3;

    if (n0 < N_total)
      w0 = (uint8_t)src_weights[n0 * K + k];
    if (n1 < N_total)
      w1 = (uint8_t)src_weights[n1 * K + k];
    if (n2 < N_total)
      w2 = (uint8_t)src_weights[n2 * K + k];
    if (n3 < N_total)
      w3 = (uint8_t)src_weights[n3 * K + k];

    uint32_t packed = ((uint32_t)w3 << 24) | ((uint32_t)w2 << 16) |
                      ((uint32_t)w1 << 8) | ((uint32_t)w0);
    dst32[k] = (int32_t)packed;
  }
}

// =============================================================
// Buffers (Aligned)
// =============================================================
int8_t __attribute__((aligned(16))) input_batch_hw[784 * 4];

// Packed weights are stored as:
// blocks = ceil(M/4)
// each block consumes (K * 4) bytes (K int32 words)
int8_t __attribute__((aligned(16))) fc1_W_hw[128 * 784]; // 100,352 bytes
int8_t __attribute__((aligned(16))) fc2_W_hw[12 * 128];  // 6,144 bytes

// =============================================================
// Driver
// =============================================================
static inline void run_accelerator_4x4(const int8_t *W_addr,
                                       const int8_t *X_addr, int M_dim,
                                       int N_dim, int K_dim) {
  ACCEL_W_ADDR = (uint32_t)W_addr;
  ACCEL_X_ADDR = (uint32_t)X_addr;
  ACCEL_M_DIM = (uint32_t)M_dim;
  ACCEL_N_DIM = (uint32_t)N_dim;
  ACCEL_K_DIM = (uint32_t)K_dim;
  ACCEL_X_STRIDE = 16; // Default linear
  ACCEL_K_ROW_LEN = (K_dim + 3) / 4;
  ACCEL_CTRL = 1; // START
}

static inline void run_accelerator_strided(const int8_t *W_addr,
                                           const int8_t *X_addr, int M_dim,
                                           int N_dim, int K_dim, int x_stride,
                                           int k_row_len) {
  ACCEL_W_ADDR = (uint32_t)W_addr;
  ACCEL_X_ADDR = (uint32_t)X_addr;
  ACCEL_M_DIM = (uint32_t)M_dim;
  ACCEL_N_DIM = (uint32_t)N_dim;
  ACCEL_K_DIM = (uint32_t)K_dim;
  ACCEL_X_STRIDE = (uint32_t)x_stride;
  ACCEL_K_ROW_LEN = (uint32_t)k_row_len;
  ACCEL_CTRL = 1; // START
}

void layer_dense_4x4(const int8_t *W_packed_base, const int8_t *X_packed,
                     int32_t **outputs, int M, int K) {

  const int blocks = (M + 3) / 4;
  const int bytes_per_block = K * 4; // K int32 words

  for (int blk = 0; blk < blocks; blk++) {
    const int m = blk * 4;

    // wait if FIFO full (bit2=full)
    while (ACCEL_STATUS & (1u << 2)) {
      ;
    }

    const int8_t *w_ptr = W_packed_base + (blk * bytes_per_block);

    run_accelerator_4x4(w_ptr, X_packed, 4, 4, K);

    // wait done (bit1=done)
    while (!(ACCEL_STATUS & (1u << 1))) {
      ;
    }

    // read results: c[row=batch][col=neuron within block]
    for (int b = 0; b < 4; b++) {
      int out_base = m;
      outputs[b][out_base + 0] = read_result(b, 0);
      if (out_base + 1 < M)
        outputs[b][out_base + 1] = read_result(b, 1);
      if (out_base + 2 < M)
        outputs[b][out_base + 2] = read_result(b, 2);
      if (out_base + 3 < M)
        outputs[b][out_base + 3] = read_result(b, 3);
    }
  }
}

// =============================================================
// Math helpers (unchanged)
// =============================================================
static int32_t soft_div(int32_t numer, int32_t denom) {
  if (denom == 0)
    return 0;
  const bool neg = ((numer < 0) ^ (denom < 0));
  uint64_t a = (numer < 0) ? (uint64_t)(-(int64_t)numer) : (uint64_t)numer;
  const uint64_t b =
      (denom < 0) ? (uint64_t)(-(int64_t)denom) : (uint64_t)denom;
  uint32_t q = 0;
  for (int i = 31; i >= 0; i--) {
    const uint64_t shifted = b << i;
    if (shifted <= a) {
      a -= shifted;
      q |= (1u << i);
    }
  }
  return neg ? -(int32_t)q : (int32_t)q;
}

static void fused_bias_relu_rescale(int32_t *raw, const int32_t *bias,
                                    int8_t *out, int size) {
  int32_t max_val = 0;
  for (int i = 0; i < size; i++) {
    int32_t v = raw[i] + bias[i];
    v = v & ~(v >> 31);
    raw[i] = v;
    if (v > max_val)
      max_val = v;
  }
  if (max_val == 0) {
    for (int i = 0; i < size; i++)
      out[i] = 0;
    return;
  }
  int32_t recip = soft_div((127 << 16), max_val);
  for (int i = 0; i < size; i++)
    out[i] = (int8_t)((raw[i] * recip) >> 16);
}

static void fused_bias_relu_rescale_4(int32_t *raw0, int32_t *raw1,
                                      int32_t *raw2, int32_t *raw3,
                                      const int32_t *bias, int8_t *out0,
                                      int8_t *out1, int8_t *out2, int8_t *out3,
                                      int size) {
  int32_t max0 = 0, max1 = 0, max2 = 0, max3 = 0;
  for (int i = 0; i < size; i++) {
    const int32_t b = bias[i];

    int32_t v0 = raw0[i] + b;
    int32_t v1 = raw1[i] + b;
    int32_t v2 = raw2[i] + b;
    int32_t v3 = raw3[i] + b;

    v0 = v0 & ~(v0 >> 31);
    v1 = v1 & ~(v1 >> 31);
    v2 = v2 & ~(v2 >> 31);
    v3 = v3 & ~(v3 >> 31);

    raw0[i] = v0;
    raw1[i] = v1;
    raw2[i] = v2;
    raw3[i] = v3;

    if (v0 > max0)
      max0 = v0;
    if (v1 > max1)
      max1 = v1;
    if (v2 > max2)
      max2 = v2;
    if (v3 > max3)
      max3 = v3;
  }

  const int nz0 = (max0 != 0);
  const int nz1 = (max1 != 0);
  const int nz2 = (max2 != 0);
  const int nz3 = (max3 != 0);

  const int32_t recip0 = nz0 ? soft_div((127 << 16), max0) : 0;
  const int32_t recip1 = nz1 ? soft_div((127 << 16), max1) : 0;
  const int32_t recip2 = nz2 ? soft_div((127 << 16), max2) : 0;
  const int32_t recip3 = nz3 ? soft_div((127 << 16), max3) : 0;

  for (int i = 0; i < size; i++) {
    out0[i] = nz0 ? (int8_t)((raw0[i] * recip0) >> 16) : 0;
    out1[i] = nz1 ? (int8_t)((raw1[i] * recip1) >> 16) : 0;
    out2[i] = nz2 ? (int8_t)((raw2[i] * recip2) >> 16) : 0;
    out3[i] = nz3 ? (int8_t)((raw3[i] * recip3) >> 16) : 0;
  }
}

static void add_bias(int32_t *out, const int32_t *bias, int size) {
  int i = 0;
  for (; i + 3 < size; i += 4) {
    out[i + 0] += bias[i + 0];
    out[i + 1] += bias[i + 1];
    out[i + 2] += bias[i + 2];
    out[i + 3] += bias[i + 3];
  }
  for (; i < size; i++)
    out[i] += bias[i];
}

static int argmax(int32_t *x, int size) {
  if (size <= 1)
    return 0;

  int max_idx0 = 0;
  int max_idx1 = 1;
  int32_t max_val0 = x[0];
  int32_t max_val1 = x[1];

  int i = 2;
  for (; i + 1 < size; i += 2) {
    const int32_t v0 = x[i];
    const int32_t v1 = x[i + 1];
    if (v0 > max_val0) {
      max_val0 = v0;
      max_idx0 = i;
    }
    if (v1 > max_val1) {
      max_val1 = v1;
      max_idx1 = i + 1;
    }
  }
  if (i < size) {
    const int32_t v = x[i];
    if (v > max_val0) {
      max_val0 = v;
      max_idx0 = i;
    }
  }

  return (max_val1 > max_val0) ? max_idx1 : max_idx0;
}

static inline void add_bias_and_argmax_4(int32_t *x0, int32_t *x1, int32_t *x2,
                                         int32_t *x3, const int32_t *bias,
                                         int size, int *pred0, int *pred1,
                                         int *pred2, int *pred3) {
  int max_idx0 = 0, max_idx1 = 0, max_idx2 = 0, max_idx3 = 0;
  int32_t max_val0 = x0[0] + bias[0];
  int32_t max_val1 = x1[0] + bias[0];
  int32_t max_val2 = x2[0] + bias[0];
  int32_t max_val3 = x3[0] + bias[0];

  x0[0] = max_val0;
  x1[0] = max_val1;
  x2[0] = max_val2;
  x3[0] = max_val3;

  for (int i = 1; i < size; i++) {
    const int32_t b = bias[i];
    const int32_t v0 = x0[i] + b;
    const int32_t v1 = x1[i] + b;
    const int32_t v2 = x2[i] + b;
    const int32_t v3 = x3[i] + b;

    x0[i] = v0;
    x1[i] = v1;
    x2[i] = v2;
    x3[i] = v3;

    if (v0 > max_val0) {
      max_val0 = v0;
      max_idx0 = i;
    }
    if (v1 > max_val1) {
      max_val1 = v1;
      max_idx1 = i;
    }
    if (v2 > max_val2) {
      max_val2 = v2;
      max_idx2 = i;
    }
    if (v3 > max_val3) {
      max_val3 = v3;
      max_idx3 = i;
    }
  }

  *pred0 = max_idx0;
  *pred1 = max_idx1;
  *pred2 = max_idx2;
  *pred3 = max_idx3;
}

// =============================================================
// Buffers for Batch=4
// =============================================================
#define BATCH_SIZE 4
int32_t l1_raw[BATCH_SIZE][HIDDEN_SIZE];
int8_t l1_q[BATCH_SIZE][HIDDEN_SIZE];
int32_t l2_raw[BATCH_SIZE][16];

int32_t *ptr_l1_raw[BATCH_SIZE];
int32_t *ptr_l2_raw[BATCH_SIZE];

int main(void) {
#ifdef ILP_MICROBENCH
  return run_ilp_microbench();
#endif

  // FC1: 128x784 => blocks=32, bytes_per_block = 784*4
  for (int i = 0; i < 128; i += 4) {
    int blk = i / 4;
    pack_weight_block(fc1_weight, fc1_W_hw + (blk * (784 * 4)), i, 784, 128);
  }

  // FC2: 10x128 padded to 12 => blocks=3, bytes_per_block = 128*4
  for (int i = 0; i < 12; i += 4) {
    int blk = i / 4;
    pack_weight_block(fc2_weight, fc2_W_hw + (blk * (128 * 4)), i, 128, 10);
  }

  for (int b = 0; b < BATCH_SIZE; b++) {
    ptr_l1_raw[b] = l1_raw[b];
    ptr_l2_raw[b] = l2_raw[b];
  }

  int correct = 0;
  for (int i = 0; i < NUM_TEST_IMAGES; i += BATCH_SIZE) {
    int batch = (NUM_TEST_IMAGES - i >= BATCH_SIZE) ? BATCH_SIZE
                                                    : (NUM_TEST_IMAGES - i);

    const int8_t *in_ptrs[4];
    for (int b = 0; b < 4; b++) {
      if (b < batch)
        in_ptrs[b] = test_images[i + b];
      else
        in_ptrs[b] = test_images[i];
    }

    // Layer 1
    pack_input_batch(in_ptrs, input_batch_hw, INPUT_SIZE);
    layer_dense_4x4(fc1_W_hw, input_batch_hw, ptr_l1_raw, HIDDEN_SIZE,
                    INPUT_SIZE);

    if (batch == 4) {
      fused_bias_relu_rescale_4(l1_raw[0], l1_raw[1], l1_raw[2], l1_raw[3],
                                fc1_bias, l1_q[0], l1_q[1], l1_q[2], l1_q[3],
                                HIDDEN_SIZE);
    } else {
      for (int b = 0; b < batch; b++) {
        fused_bias_relu_rescale(l1_raw[b], fc1_bias, l1_q[b], HIDDEN_SIZE);
      }
    }

    // Layer 2
    const int8_t *l1_ptrs[4];
    for (int b = 0; b < 4; b++)
      l1_ptrs[b] = l1_q[b];
    pack_input_batch(l1_ptrs, input_batch_hw, HIDDEN_SIZE);

    // M padded to 12 so blocks align; bias/argmax uses OUTPUT_SIZE=10
    layer_dense_4x4(fc2_W_hw, input_batch_hw, ptr_l2_raw, 12, HIDDEN_SIZE);

    if (batch == 4) {
      int pred0, pred1, pred2, pred3;
      add_bias_and_argmax_4(l2_raw[0], l2_raw[1], l2_raw[2], l2_raw[3],
                            fc2_bias, OUTPUT_SIZE, &pred0, &pred1, &pred2,
                            &pred3);
      if (pred0 == expected_labels[i + 0])
        correct++;
      if (pred1 == expected_labels[i + 1])
        correct++;
      if (pred2 == expected_labels[i + 2])
        correct++;
      if (pred3 == expected_labels[i + 3])
        correct++;
    } else {
      for (int b = 0; b < batch; b++) {
        add_bias(l2_raw[b], fc2_bias, OUTPUT_SIZE);
        int pred = argmax(l2_raw[b], OUTPUT_SIZE);
        if (pred == expected_labels[i + b])
          correct++;
      }
    }
  }

  int num_wrong = NUM_TEST_IMAGES - correct;
  if (num_wrong == 0)
    csr_tohost(1);
  else
    csr_tohost(num_wrong + 1);

  for (;;)
    asm volatile("nop");
}
