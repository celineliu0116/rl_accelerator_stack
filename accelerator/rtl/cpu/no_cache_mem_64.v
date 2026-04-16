`include "util.vh"
`include "const.vh"

// 64-bit instruction cache memory for dual-issue
// Fetches two consecutive 32-bit instructions at once
module no_cache_mem_64 #(
  parameter CPU_WIDTH      = 64,  // 64-bit output
  parameter WORD_ADDR_BITS = `CPU_ADDR_BITS - 2  // Word address (32-bit aligned)
) (
  input clk,
  input reset,

  input                       cpu_req_valid,
  output                      cpu_req_ready,
  input [WORD_ADDR_BITS-1:0]  cpu_req_addr,  // 32-bit word address

  output reg                  cpu_resp_valid,
  output reg [CPU_WIDTH-1:0]  cpu_resp_data,   // Two instructions
  output reg [WORD_ADDR_BITS-1:0] cpu_resp_addr
);

  localparam DEPTH = 2*1024*1024;
  localparam MEM_WIDTH = `MEM_DATA_BITS;  // 128 bits

  reg [MEM_WIDTH-1:0] ram [DEPTH-1:0];

  // Fetch two consecutive instructions, even when crossing a 128-bit line.
  wire [WORD_ADDR_BITS-1:0] cpu_req_addr_next = cpu_req_addr + {{(WORD_ADDR_BITS-1){1'b0}}, 1'b1};

  wire [WORD_ADDR_BITS-3:0] line_addr_0_w = cpu_req_addr[WORD_ADDR_BITS-1:2];
  wire [WORD_ADDR_BITS-3:0] line_addr_1_w = cpu_req_addr_next[WORD_ADDR_BITS-1:2];

  wire [`ceilLog2(DEPTH)-1:0] ram_addr_0 = line_addr_0_w[`ceilLog2(DEPTH)-1:0];
  wire [`ceilLog2(DEPTH)-1:0] ram_addr_1 = line_addr_1_w[`ceilLog2(DEPTH)-1:0];

  wire [MEM_WIDTH-1:0] read_line_0 = ram[ram_addr_0];
  wire [MEM_WIDTH-1:0] read_line_1 = ram[ram_addr_1];

  wire [1:0] word_sel_0 = cpu_req_addr[1:0];
  wire [1:0] word_sel_1 = cpu_req_addr_next[1:0];

  wire [MEM_WIDTH-1:0] shifted_line_0 = read_line_0 >> (word_sel_0 * 32);
  wire [MEM_WIDTH-1:0] shifted_line_1 = read_line_1 >> (word_sel_1 * 32);
  wire [31:0] inst0 = shifted_line_0[31:0];
  wire [31:0] inst1 = shifted_line_1[31:0];
  wire [CPU_WIDTH-1:0] read_data = {inst1, inst0};

  assign cpu_req_ready = 1'b1;

  always @(posedge clk) begin
    if (reset) begin
      cpu_resp_valid <= 1'b0;
      cpu_resp_data <= {CPU_WIDTH{1'b0}};
      cpu_resp_addr <= {WORD_ADDR_BITS{1'b0}};
    end else if (cpu_req_valid && cpu_req_ready) begin
      cpu_resp_valid <= 1'b1;
      cpu_resp_data <= read_data;
      cpu_resp_addr <= cpu_req_addr;
    end else
      cpu_resp_valid <= 1'b0;
  end

  initial
  begin : zero
    integer i;
    for (i = 0; i < DEPTH; i = i + 1)
      ram[i] = 0;
  end

endmodule
