// inference.c — Fixed-point MLP inference for RISC-V
// Performs MNIST digit classification using INT8 weights
//
// Math:
//   hidden = ReLU(W1 * input + b1)
//   output = W2 * hidden + b2
//   prediction = argmax(output)

#include "test_images.h"
#include "weights.h"
#include <stdint.h>

// =============================================================
// Print helpers (using RISC-V syscalls via proxy kernel)
// =============================================================

// Minimal putchar — works with Spike + pk
extern int putchar(int c);

void print_str(const char *s) {
  while (*s)
    putchar(*s++);
}

void print_int(int32_t n) {
  if (n < 0) {
    putchar('-');
    n = -n;
  }
  if (n == 0) {
    putchar('0');
    return;
  }
  char buf[12];
  int i = 0;
  while (n > 0) {
    buf[i++] = '0' + (n % 10);
    n /= 10;
  }
  while (i > 0)
    putchar(buf[--i]);
}

// Read cycle counter (mcycle CSR)
static inline uint64_t read_cycles(void) {
  uint64_t cycles;
  asm volatile("rdcycle %0" : "=r"(cycles));
  return cycles;
}

// =============================================================
// Fixed-point neural network operations
// =============================================================

// Matrix-vector multiply: out[M] = W[M][N] * x[N]
// W is stored row-major as a flat array
// Uses int32 accumulation to avoid overflow
// 4x unrolled inner loop to reduce loop overhead
static inline void matmul(const int8_t *W, const int8_t *x, int32_t *out, int M,
                          int N) {
  for (int i = 0; i < M; i++) {
    int32_t acc = 0;
    const int8_t *row = &W[i * N];
    int j = 0;
    // Process 4 elements per iteration
    for (; j <= N - 4; j += 4) {
      acc += (int32_t)row[j] * (int32_t)x[j];
      acc += (int32_t)row[j + 1] * (int32_t)x[j + 1];
      acc += (int32_t)row[j + 2] * (int32_t)x[j + 2];
      acc += (int32_t)row[j + 3] * (int32_t)x[j + 3];
    }
    // Handle remainder
    for (; j < N; j++) {
      acc += (int32_t)row[j] * (int32_t)x[j];
    }
    out[i] = acc;
  }
}

// Add pre-scaled int32 bias directly to accumulator
static inline void add_bias(int32_t *out, const int32_t *bias, int size) {
  for (int i = 0; i < size; i++) {
    out[i] += bias[i];
  }
}

// ReLU: clamp negatives to zero (branchless)
static inline void relu(int32_t *x, int size) {
  for (int i = 0; i < size; i++) {
    x[i] = x[i] & ~(x[i] >> 31); // zero if negative, keep if positive
  }
}

// Rescale int32 accumulator back to int8 range for next layer
static inline void rescale_to_int8(int32_t *in, int8_t *out, int size) {
  // Find max absolute value
  int32_t max_val = 0;
  for (int i = 0; i < size; i++) {
    int32_t abs_val = in[i] < 0 ? -in[i] : in[i];
    if (abs_val > max_val)
      max_val = abs_val;
  }

  if (max_val == 0) {
    for (int i = 0; i < size; i++)
      out[i] = 0;
    return;
  }

  // Scale down to [-127, 127] using exact division
  for (int i = 0; i < size; i++) {
    out[i] = (int8_t)((in[i] * 127) / max_val);
  }
}

// Argmax: find index of largest value
static inline int argmax(int32_t *x, int size) {
  int max_idx = 0;
  int32_t max_val = x[0];
  for (int i = 1; i < size; i++) {
    if (x[i] > max_val) {
      max_val = x[i];
      max_idx = i;
    }
  }
  return max_idx;
}

// =============================================================
// Full forward pass
// =============================================================
int predict(const int8_t *image) {
  int32_t hidden_raw[HIDDEN_SIZE]; // int32 accumulator after layer 1
  int8_t hidden_q[HIDDEN_SIZE];    // rescaled to int8 for layer 2
  int32_t output[OUTPUT_SIZE];     // int32 accumulator after layer 2

  // Layer 1: hidden = W1 * image + b1
  matmul(fc1_weight, image, hidden_raw, HIDDEN_SIZE, INPUT_SIZE);
  // Biases are pre-scaled to accumulator scale during export
  add_bias(hidden_raw, fc1_bias, HIDDEN_SIZE);

  // ReLU
  relu(hidden_raw, HIDDEN_SIZE);

  // Rescale to int8 for layer 2 input
  rescale_to_int8(hidden_raw, hidden_q, HIDDEN_SIZE);

  // Layer 2: output = W2 * hidden + b2
  matmul(fc2_weight, hidden_q, output, OUTPUT_SIZE, HIDDEN_SIZE);
  add_bias(output, fc2_bias, OUTPUT_SIZE);

  // Argmax
  return argmax(output, OUTPUT_SIZE);
}

// =============================================================
// Main — run inference on test images
// =============================================================
int main() {
  print_str("===================================\n");
  print_str("RISC-V Neural Network Inference\n");
  print_str("Model: 2-layer MLP (784->128->10)\n");
  print_str("Dataset: MNIST handwritten digits\n");
  print_str("===================================\n\n");

  int correct = 0;
  uint64_t total_cycles = 0;

  for (int i = 0; i < NUM_TEST_IMAGES; i++) {
    uint64_t start = read_cycles();
    int prediction = predict(test_images[i]);
    uint64_t end = read_cycles();
    uint64_t elapsed = end - start;
    total_cycles += elapsed;

    int expected = expected_labels[i];

    print_str("Image ");
    print_int(i);
    print_str(": predicted=");
    print_int(prediction);
    print_str(" expected=");
    print_int(expected);

    if (prediction == expected) {
      print_str(" [CORRECT] ");
      correct++;
    } else {
      print_str(" [WRONG]   ");
    }

    print_str("cycles=");
    print_int((int32_t)elapsed);
    print_str("\n");
  }

  print_str("\nResults: ");
  print_int(correct);
  print_str("/");
  print_int(NUM_TEST_IMAGES);
  print_str(" correct\n");

  print_str("Total cycles: ");
  print_int((int32_t)total_cycles);
  print_str("\n");

  print_str("Avg cycles/inference: ");
  print_int((int32_t)(total_cycles / NUM_TEST_IMAGES));
  print_str("\n");

  return 0;
}