#include "accel.h"
#include <stdint.h>

#define csr_tohost(val)                                                        \
  do {                                                                         \
    asm volatile("csrw 0x51e, %[v]" ::[v] "r"(val));                           \
  } while (0)

static uint32_t dma_buf[32] __attribute__((aligned(32)));
static int32_t output[4][4];

static inline uint32_t pack4(int8_t r0, int8_t r1, int8_t r2, int8_t r3) {
  return ((uint32_t)(uint8_t)r0) | ((uint32_t)(uint8_t)r1 << 8) |
         ((uint32_t)(uint8_t)r2 << 16) | ((uint32_t)(uint8_t)r3 << 24);
}

static inline uint32_t pack_w(uint32_t idx0, uint32_t idx1, int8_t val0,
                              int8_t val1) {
  uint32_t meta = ((idx1 & 3) << 2) | (idx0 & 3);
  return (meta << 16) | (((uint32_t)(uint8_t)val1) << 8) |
         ((uint32_t)(uint8_t)val0);
}

int main(void) {
  // Case 1: K=8, 4 Batch, 4 Neurons. 2 Beats.

  // Beat 0: K-block 0 (K=0..3). Shared sparse mask idx0=1, idx1=3.
  // X: Batch 0..3 have some values at K=1 and K=3.
  // Let batch 0: X[1]=2, X[3]=4
  // Let batch 1: X[1]=1, X[3]=1
  // Let batch 2: X[1]=-1, X[3]=2
  // Let batch 3: X[1]=0, X[3]=0
  dma_buf[0] = pack4(0, 2, 0, 4);  // Batch 0: K0=0, K1=2, K2=0, K3=4
  dma_buf[1] = pack4(0, 1, 0, 1);  // Batch 1
  dma_buf[2] = pack4(0, -1, 0, 2); // Batch 2
  dma_buf[3] = pack4(0, 0, 0, 0);  // Batch 3

  // W: Neurons 0..3 have vals at idx=1 and idx=3.
  // Neuron 0: val0=10, val1=-5
  // Neuron 1: val0=20, val1=0
  // Neuron 2: val0=30, val1=5
  // Neuron 3: val0=40, val1=-10
  dma_buf[4] = pack_w(1, 3, 10, -5);
  dma_buf[5] = pack_w(1, 3, 20, 0);
  dma_buf[6] = pack_w(1, 3, 30, 5);
  dma_buf[7] = pack_w(1, 3, 40, -10);

  // Beat 1: K-block 1 (K=4..7). Shared sparse mask idx0=0, idx1=2.
  // Batch 0: X[4]=5, X[6]=2
  // Batch 1: X[4]=-2, X[6]=3
  // Batch 2: X[4]=1, X[6]=1
  // Batch 3: X[4]=10, X[6]=10
  dma_buf[8] = pack4(5, 0, 2, 0);
  dma_buf[9] = pack4(-2, 0, 3, 0);
  dma_buf[10] = pack4(1, 0, 1, 0);
  dma_buf[11] = pack4(10, 0, 10, 0);

  // W: Neurons 0..3
  // Neuron 0: val0=1, val1=1
  // Neuron 1: val0=2, val1=-2
  // Neuron 2: val0=3, val1=3
  // Neuron 3: val0=4, val1=-4
  dma_buf[12] = pack_w(0, 2, 1, 1);
  dma_buf[13] = pack_w(0, 2, 2, -2);
  dma_buf[14] = pack_w(0, 2, 3, 3);
  dma_buf[15] = pack_w(0, 2, 4, -4);

  accel_run_ext(4u, 4u, 8u, dma_buf, dma_buf, 0u, 0u, 1u);

  // Wait for DONE and read results
  // Expected logic:
  // Beat 0, Batch 0, Neuron 0: (2 * 10) + (4 * -5) = 20 - 20 = 0
  // Beat 1, Batch 0, Neuron 0: (5 * 1) + (2 * 1) = 7
  // Total B0 N0 = 7

  // B0 N1: (2 * 20) + (4 * 0) = 40. + (5 * 2) + (2 * -2) = 46. Total = 46.
  // B0 N2: (2 * 30) + (4 * 5) = 80. + (5 * 3) + (2 * 3) = 21. Total = 101.
  // B0 N3: (2 * 40) + (4 * -10) = 40. + (5 * 4) + (2 * -4) = 12. Total = 52.

  // B1 N0: (1 * 10) + (1 * -5) = 5. + (-2 * 1) + (3 * 1) = 1. Total = 6.
  // B1 N1: (1 * 20) + (1 * 0) = 20. + (-2 * 2) + (3 * -2) = -10. Total = 10.
  // B1 N2: (1 * 30) + (1 * 5) = 35. + (-2 * 3) + (3 * 3) = 3. Total = 38.
  // B1 N3: (1 * 40) + (1 * -10) = 30. + (-2 * 4) + (3 * -4) = -20. Total = 10.

  // Using a simple array of expectations
  int32_t expected[4][4] = {
      {7, 46, 101, 52}, {6, 10, 38, 10}, {-18, -20, -14, -60}, {20, 0, 60, 0}};

  for (uint32_t r = 0; r < 4; r++) {
    for (uint32_t c = 0; c < 4; c++) {
      output[r][c] = accel_read_result_cell(r, c);
    }
  }

  for (uint32_t r = 0; r < 4; r++) {
    for (uint32_t c = 0; c < 4; c++) {
      if (output[r][c] != expected[r][c]) {
        if (r == 0 && c == 2) {
          csr_tohost((output[r][c] & 0xFFFF) | 0xDEAD0000u);
          for (;;)
            ;
        }
        if (r == 3 && c == 3) {
          csr_tohost((output[r][c] & 0xFFFF) | 0xDEAD0000u);
          for (;;)
            ;
        }
        csr_tohost(100u + r * 4 + c);
      }
    }
  }

  csr_tohost(1u);
  for (;;)
    ;
}
