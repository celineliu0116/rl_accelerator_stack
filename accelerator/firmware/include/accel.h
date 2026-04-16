#ifndef ACCEL_H
#define ACCEL_H

#include <stdint.h>

#define ACCEL_BASE_ADDR 0x80000000u

#define ACCEL_REG_CTRL 0x00u
#define ACCEL_REG_STATUS 0x00u
#define ACCEL_REG_W_ADDR 0x04u
#define ACCEL_REG_X_ADDR 0x08u
#define ACCEL_REG_M_DIM 0x0Cu
#define ACCEL_REG_N_DIM 0x10u
#define ACCEL_REG_K_DIM 0x14u
#define ACCEL_REG_RESULT_BASE 0x18u
#define ACCEL_REG_X_STRIDE 0x58u
#define ACCEL_REG_K_ROW_LEN 0x5Cu
#define ACCEL_REG_HW_MODE 0x70u
#define ACCEL_REG_CSR_VAL_ADDR 0x74u
#define ACCEL_REG_CSR_COL_IDX_ADDR 0x78u
#define ACCEL_REG_CSR_ROW_PTR_ADDR 0x7Cu
#define ACCEL_PERF_PE_IDLE 0x80u
#define ACCEL_PERF_DMA_STALL 0x84u

#define ACCEL_STATUS_DONE (1u << 1)
#define ACCEL_STATUS_FULL (1u << 2)

static inline volatile uint32_t *accel_reg_ptr(uint32_t offset) {
  return (volatile uint32_t *)(ACCEL_BASE_ADDR + offset);
}

static inline void accel_mmio_write(uint32_t offset, uint32_t value) {
  *accel_reg_ptr(offset) = value;
}

static inline uint32_t accel_mmio_read(uint32_t offset) {
  return *accel_reg_ptr(offset);
}

static inline void accel_wait_done_level(void) {
  while ((accel_mmio_read(ACCEL_REG_STATUS) & ACCEL_STATUS_DONE) == 0u) {
  }
}

static inline void accel_wait_done_edge(void) {
  // Sparse mode unit tests issue back-to-back commands against tiny workloads.
  // In that case DONE may still be high from the previous command; require a
  // clear->set transition to avoid returning on stale DONE.
  while ((accel_mmio_read(ACCEL_REG_STATUS) & ACCEL_STATUS_DONE) != 0u) {
  }
  while ((accel_mmio_read(ACCEL_REG_STATUS) & ACCEL_STATUS_DONE) == 0u) {
  }
}

static inline void accel_wait_done(void) { accel_wait_done_level(); }

static inline void accel_wait_not_full(void) {
  while ((accel_mmio_read(ACCEL_REG_STATUS) & ACCEL_STATUS_FULL) != 0u) {
  }
}

static inline void accel_mem_fence(void) { asm volatile("" ::: "memory"); }

static inline int32_t accel_read_result_cell(uint32_t row, uint32_t col) {
  volatile int32_t *addr =
      (volatile int32_t *)(ACCEL_BASE_ADDR + ACCEL_REG_RESULT_BASE +
                           ((row * 4u + col) * 4u));
  return *addr;
}

static inline void accel_wait_busy_clear(void) {
  // Bit 0 = 1 if FSM not IDLE or FIFO not empty. Safe for exact pipe overlap.
  while ((accel_mmio_read(ACCEL_REG_STATUS) & 1u) != 0u) {
  }
}

static inline void accel_issue_ext(uint32_t m_dim, uint32_t n_dim,
                                   uint32_t k_dim, const void *weight_addr,
                                   const void *input_addr, uint32_t x_stride,
                                   uint32_t k_row_len, uint32_t hw_mode) {
  accel_wait_not_full();
  accel_mmio_write(ACCEL_REG_W_ADDR, (uint32_t)(uintptr_t)weight_addr);
  accel_mmio_write(ACCEL_REG_X_ADDR, (uint32_t)(uintptr_t)input_addr);
  accel_mmio_write(ACCEL_REG_M_DIM, m_dim);
  accel_mmio_write(ACCEL_REG_N_DIM, n_dim);
  accel_mmio_write(ACCEL_REG_K_DIM, k_dim);
  accel_mmio_write(ACCEL_REG_X_STRIDE, x_stride);
  accel_mmio_write(ACCEL_REG_K_ROW_LEN, k_row_len);
  accel_mmio_write(ACCEL_REG_HW_MODE, hw_mode);
  accel_mem_fence();
  accel_mmio_write(ACCEL_REG_CTRL, 1u);
  // Do NOT wait. Return immediately so CPU can overlap.
  accel_mem_fence();
}

static inline void accel_issue_next_w(const void *weight_addr) {
  accel_wait_not_full();
  accel_mmio_write(ACCEL_REG_W_ADDR, (uint32_t)(uintptr_t)weight_addr);
  accel_mem_fence();
  accel_mmio_write(ACCEL_REG_CTRL, 1u);
  accel_mem_fence();
}

static inline void accel_run(uint32_t m_dim, uint32_t n_dim, uint32_t k_dim,
                             const void *weight_addr, const void *input_addr,
                             uint32_t x_stride, uint32_t k_row_len) {
  accel_issue_ext(m_dim, n_dim, k_dim, weight_addr, input_addr, x_stride,
                  k_row_len, 0u);
  accel_wait_done_level();
}

static inline void accel_run_sparse(uint32_t m_dim, uint32_t n_dim,
                                    uint32_t k_dim, const void *csr_val_addr,
                                    const void *csr_col_idx_addr,
                                    const void *csr_row_ptr_addr,
                                    const void *input_addr, uint32_t hw_mode) {
  accel_wait_not_full();
  accel_mmio_write(ACCEL_REG_X_ADDR, (uint32_t)(uintptr_t)input_addr);
  accel_mmio_write(ACCEL_REG_M_DIM, m_dim);
  accel_mmio_write(ACCEL_REG_N_DIM, n_dim);
  accel_mmio_write(ACCEL_REG_K_DIM, k_dim);
  accel_mmio_write(ACCEL_REG_HW_MODE, hw_mode);
  accel_mmio_write(ACCEL_REG_CSR_VAL_ADDR, (uint32_t)(uintptr_t)csr_val_addr);
  accel_mmio_write(ACCEL_REG_CSR_COL_IDX_ADDR,
                   (uint32_t)(uintptr_t)csr_col_idx_addr);
  accel_mmio_write(ACCEL_REG_CSR_ROW_PTR_ADDR,
                   (uint32_t)(uintptr_t)csr_row_ptr_addr);
  accel_mem_fence();
  accel_mmio_write(ACCEL_REG_CTRL, 1u);
  accel_wait_done_edge();
  accel_mem_fence();
}

#endif
