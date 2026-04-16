#include "Vriscv_top.h"
#include "Vriscv_top___024root.h"
#include "verilated.h"
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Verilator timing is event-driven; the testbench toggles clk manually.
// Cycle count is tracked directly.

uint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static bool parse_hex_line(const char *line, uint32_t words[4]) {
  const size_t len = strlen(line);
  if (len == 0 || len > 32) {
    return false;
  }

  for (int i = 0; i < 4; i++) {
    words[i] = 0;
  }

  // Input is big-endian hex text. The parser reconstructs 4 little-endian
  // 32-bit words.
  for (int i = 0; i < static_cast<int>(len); i++) {
    const char c = line[len - 1 - i];
    if (!std::isxdigit(static_cast<unsigned char>(c))) {
      return false;
    }

    int nibble = 0;
    if (c >= '0' && c <= '9') {
      nibble = c - '0';
    } else if (c >= 'a' && c <= 'f') {
      nibble = c - 'a' + 10;
    } else {
      nibble = c - 'A' + 10;
    }

    const int word_idx = i / 8;
    const int bit_pos = (i % 8) * 4;
    words[word_idx] |= (nibble << bit_pos);
  }

  return true;
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);

  // Parse arguments
  const char *hex_file = nullptr;
  uint64_t max_cycles = 50000000; // 50M cycles default
  bool loop_trace = false;
  uint32_t loop_start = 0x00000000u;
  uint32_t loop_end = 0xffffffffu;
  uint64_t loop_max = 0; // 0 = unlimited
  for (int i = 1; i < argc; i++) {
    if (strncmp(argv[i], "+loadmem=", 9) == 0) {
      hex_file = argv[i] + 9;
    } else if (strncmp(argv[i], "+max-cycles=", 12) == 0) {
      max_cycles = atoll(argv[i] + 12);
    } else if (strcmp(argv[i], "+loop-trace") == 0) {
      loop_trace = true;
    } else if (strncmp(argv[i], "+loop-start=", 12) == 0) {
      loop_start = static_cast<uint32_t>(strtoull(argv[i] + 12, nullptr, 0));
    } else if (strncmp(argv[i], "+loop-end=", 10) == 0) {
      loop_end = static_cast<uint32_t>(strtoull(argv[i] + 10, nullptr, 0));
    } else if (strncmp(argv[i], "+loop-max=", 10) == 0) {
      loop_max = strtoull(argv[i] + 10, nullptr, 0);
    }
  }

  if (!hex_file) {
    fprintf(stderr, "Usage: %s +loadmem=<hexfile> [+max-cycles=N]\n", argv[0]);
    return 1;
  }

  // Create model
  Vriscv_top *top = new Vriscv_top;

  // ---------------------------------------------------------------
  // Reset first to trigger Verilator `initial` blocks.
  // no_cache_mem initialization clears RAM, so program loading
  // is performed after this reset sequence.
  // ---------------------------------------------------------------
  top->reset = 1;
  top->clk = 0;
  for (int i = 0; i < 100; i++) {
    top->clk = !top->clk;
    top->eval();
    main_time++;
  }
  top->reset = 0;

  // ---------------------------------------------------------------
  // NOW load hex file into memory (after initial blocks have run)
  // ---------------------------------------------------------------
  FILE *f = fopen(hex_file, "r");
  if (!f) {
    fprintf(stderr, "ERROR: Cannot open hex file: %s\n", hex_file);
    return 1;
  }

  // objcopy -O binary strips the base address. The binary starts at VMA 0x2000
  // (PC_RESET). no_cache_mem uses 128-bit (16-byte) indexed RAM, so
  // 0x2000 / 16 = 512. Loading starts at that offset to match the CPU reset PC.
  const int mem_depth =
      sizeof(top->rootp->riscv_top__DOT__mem__DOT__icache__DOT__ram) /
      sizeof(top->rootp->riscv_top__DOT__mem__DOT__icache__DOT__ram[0]);

  char line[256];
  int addr = 0x2000 / 16; // must match PC_RESET / (MEM_DATA_BITS/8)
  int line_no = 0;
  while (fgets(line, sizeof(line), f)) {
    line_no++;
    line[strcspn(line, "\r\n")] = 0;
    if (strlen(line) == 0)
      continue;

    uint32_t words[4] = {0, 0, 0, 0};
    if (!parse_hex_line(line, words)) {
      fprintf(stderr, "ERROR: Invalid hex line %d in %s\n", line_no, hex_file);
      fclose(f);
      delete top;
      return 1;
    }
    if (addr >= mem_depth) {
      fprintf(stderr,
              "ERROR: Hex image exceeds memory depth at line %d (addr=%d)\n",
              line_no, addr);
      fclose(f);
      delete top;
      return 1;
    }

    // Store as 128-bit value in memory model
    // Verilator stores wide signals as arrays of uint32_t
    // For WData (wide data), index [0] is LSB
    top->rootp->riscv_top__DOT__mem__DOT__icache__DOT__ram[addr][0] = words[0];
    top->rootp->riscv_top__DOT__mem__DOT__icache__DOT__ram[addr][1] = words[1];
    top->rootp->riscv_top__DOT__mem__DOT__icache__DOT__ram[addr][2] = words[2];
    top->rootp->riscv_top__DOT__mem__DOT__icache__DOT__ram[addr][3] = words[3];

    top->rootp->riscv_top__DOT__mem__DOT__dcache__DOT__ram[addr][0] = words[0];
    top->rootp->riscv_top__DOT__mem__DOT__dcache__DOT__ram[addr][1] = words[1];
    top->rootp->riscv_top__DOT__mem__DOT__dcache__DOT__ram[addr][2] = words[2];
    top->rootp->riscv_top__DOT__mem__DOT__dcache__DOT__ram[addr][3] = words[3];

    addr++;
  }
  fclose(f);
  fprintf(stderr, "Loaded %d lines from %s\n", addr - 0x2000 / 16, hex_file);

  // Run simulation
  uint64_t cycle_count = 0;
  uint64_t cnt_lane1_issue = 0;
  uint64_t cnt_dual_issued = 0;
  uint64_t cnt_flush_id = 0;
  uint64_t cnt_front_freeze = 0;
  uint64_t cnt_retired0 = 0;
  uint64_t cnt_retired1 = 0;
  uint64_t loop_lines = 0;

  while (cycle_count < max_cycles) {
    // Rising edge
    top->clk = 1;
    top->eval();
    main_time++;
    cycle_count++;

#ifdef SIM_INTERNAL_PROBES
    const uint32_t pc_f =
        top->rootp->riscv_top__DOT__cpu__DOT__u_fetch__DOT__pc_F;
    const uint32_t pc_id = top->rootp->riscv_top__DOT__cpu__DOT__pc_id;
    const uint32_t i0 = top->rootp->riscv_top__DOT__cpu__DOT__inst_id;
    const uint32_t i1 = top->rootp->riscv_top__DOT__cpu__DOT__inst_id_1;
    const int flush_id = top->rootp->riscv_top__DOT__cpu__DOT__flush_id ? 1 : 0;
    const int front_freeze =
        top->rootp->riscv_top__DOT__cpu__DOT__load_use_hazard ? 1 : 0;
    const int lane1_issue_en =
        top->rootp->riscv_top__DOT__cpu__DOT__issue_ex_1_r ? 1 : 0;
    const int dual_issued = lane1_issue_en && !flush_id && !front_freeze;
    const uint32_t fwd_a0 = top->rootp->riscv_top__DOT__cpu__DOT__fwd_a_0_sel;
    const uint32_t fwd_b0 = top->rootp->riscv_top__DOT__cpu__DOT__fwd_b_0_sel;
    const uint32_t inst_wb_0 = top->rootp->riscv_top__DOT__cpu__DOT__inst_wb;
    const uint32_t inst_wb_1 = top->rootp->riscv_top__DOT__cpu__DOT__inst_wb_1;
    const int valid_wb_1 =
        top->rootp->riscv_top__DOT__cpu__DOT__valid_wb_1 ? 1 : 0;
    const uint32_t rd_wb0 = (inst_wb_0 >> 7) & 0x1f;
    const uint32_t rd_wb1 = (inst_wb_1 >> 7) & 0x1f;
    const uint32_t alu_ex1 = top->rootp->riscv_top__DOT__cpu__DOT__alu_out_ex_1;
    const uint32_t inst_ex0 = top->rootp->riscv_top__DOT__cpu__DOT__inst_ex_r;
    const uint32_t inst_ex1 = top->rootp->riscv_top__DOT__cpu__DOT__inst_ex_1_r;
    const uint32_t rd_ex0 = (inst_ex0 >> 7) & 0x1f;
    const uint32_t rd_ex1 = (inst_ex1 >> 7) & 0x1f;

    cnt_lane1_issue += lane1_issue_en;
    cnt_dual_issued += dual_issued;
    cnt_flush_id += flush_id;
    cnt_front_freeze += front_freeze;
    if (inst_wb_0 != 0 && inst_wb_0 != 0x00000013u) {
      cnt_retired0++;
    }
    if (valid_wb_1 && inst_wb_1 != 0 && inst_wb_1 != 0x00000013u) {
      cnt_retired1++;
    }

    if (cycle_count <= 20) {
      fprintf(stderr,
              "[C%llu] PC_F=%08x PC_ID=%08x I0=%08x I1=%08x flush_id=%d "
              "freeze=%d dual_issued=%d lane1_issue_en=%d\n",
              cycle_count, pc_f, pc_id, i0, i1, flush_id, front_freeze,
              dual_issued, lane1_issue_en);
    }

    if (loop_trace && (pc_f >= loop_start) && (pc_f <= loop_end) &&
        main_time > 6366800 && ((loop_max == 0) || (loop_lines < loop_max))) {
      fprintf(stderr,
              "[loop] C=%llu PC_F=%08x PC_ID=%08x I0=%08x I1=%08x flush_id=%d "
              "freeze=%d dual_issued=%d lane1_issue_en=%d fwd_a0=%u fwd_b0=%u "
              "rd_wb0=%u rd_wb1=%u alu_ex1=%08x rd_ex0=%u rd_ex1=%u\n",
              cycle_count, pc_f, pc_id, i0, i1, flush_id, front_freeze,
              dual_issued, lane1_issue_en, fwd_a0, fwd_b0, rd_wb0, rd_wb1,
              alu_ex1, rd_ex0, rd_ex1);
      loop_lines++;
    }
#endif

    // Check CSR (tohost)
    uint32_t csr_val = top->csr;

    if (csr_val == 1 && cycle_count > 10) {
      fprintf(stderr, "*** PASSED *** after %llu simulation cycles\n",
              cycle_count);
      break;
    }
    if (csr_val > 1 && cycle_count > 10) {
      fprintf(stderr,
              "*** FAILED *** (tohost = %d) after %llu simulation cycles\n",
              csr_val, cycle_count);
      break;
    }

    // Falling edge
    top->clk = 0;
    top->eval();
    main_time++;
  }

  if (cycle_count >= max_cycles) {
    fprintf(stderr, "*** TIMEOUT *** after %llu simulation cycles\n",
            cycle_count);
  }

  fprintf(stderr, "Total cycles: %llu\n", cycle_count);
  fprintf(stderr,
          "Counter: dual_issued=%llu lane1_issue_en=%llu flush_id=%llu "
          "freeze=%llu\n",
          cnt_dual_issued, cnt_lane1_issue, cnt_flush_id, cnt_front_freeze);
  const uint64_t retired_total = cnt_retired0 + cnt_retired1;
  const double ipc =
      (cycle_count == 0) ? 0.0 : ((double)retired_total / (double)cycle_count);
  fprintf(stderr, "Retired: lane0=%llu lane1=%llu total=%llu IPC=%.6f\n",
          cnt_retired0, cnt_retired1, retired_total, ipc);

  top->final();
  delete top;
  return 0;
}
