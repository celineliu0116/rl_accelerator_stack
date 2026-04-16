// GShare-style 2-bit BHT (acts as PHT)
module BHT2bc #(
    parameter ENTRIES  = 128,
    parameter IDX_BITS = $clog2(ENTRIES)
)(
  input  wire        clk,
  input  wire        reset,
  input  wire [31:0] pc_r,
  output wire        pred_taken,
  input  wire        upd_valid,
  input  wire [31:0] upd_pc,
  input  wire        upd_taken
);

  // global history register
  reg [IDX_BITS-1:0] ghr;

  reg [1:0] counter_table [ENTRIES-1:0];

  // XOR PC bits with GHR
  wire [IDX_BITS-1:0] r_idx = (pc_r[IDX_BITS+1:2] ^ ghr);
  wire [IDX_BITS-1:0] w_idx = (upd_pc[IDX_BITS+1:2] ^ ghr);

  assign pred_taken = counter_table[r_idx][1];

  // 2-bit counter helpers
  function [1:0] inc2(input [1:0] x); inc2 = (x==2'b11) ? 2'b11 : x+1; endfunction
  function [1:0] dec2(input [1:0] x); dec2 = (x==2'b00) ? 2'b00 : x-1; endfunction

  integer i;
  initial begin
    for (i=0; i<ENTRIES; i=i+1) counter_table[i] = 2'b01;
  end

  always @(posedge clk) begin
    if (reset) begin
      ghr <= {IDX_BITS{1'b0}};
      // Do not reset counter_table for synthesis (maps to RAM)
    end else if (upd_valid) begin
      counter_table[w_idx] <= upd_taken ? inc2(counter_table[w_idx]) : dec2(counter_table[w_idx]);
      // shift in newest outcome bit (global history update)
      ghr <= {ghr[IDX_BITS-2:0], upd_taken};
    end
  end
endmodule

  
// DIRECT-MAPPED BTB - stores valid/tag/target arrays, checks tags to drive hit/target_r
// updates entries on taken branches (only when upd_valid and branch taken)
module BTB #(
  parameter ENTRIES  = 128,
  parameter IDX_BITS = $clog2(ENTRIES)  // 7
)(
  input  wire        clk,
  input  wire        reset,
  // combinational read
  input  wire [31:0] pc_r,
  output wire        hit,
  output wire [31:0] target_r,
  // synchronous update (from DX stage, only if taken)
  input  wire        upd_valid,
  input  wire [31:0] upd_pc,
  input  wire [31:0] upd_target
);
  localparam TAG_BITS = 32 - (IDX_BITS + 2); // 23
  
  reg [ENTRIES-1:0]   valid; // Bit vector for easy reset
  reg [TAG_BITS-1:0]  tag     [ENTRIES-1:0];
  reg [31:0]          target  [ENTRIES-1:0];
  
  wire [IDX_BITS-1:0] r_idx = pc_r[IDX_BITS+1:2];
  wire [TAG_BITS-1:0] r_tag = pc_r[31:32-TAG_BITS];
  wire [IDX_BITS-1:0] w_idx = upd_pc[IDX_BITS+1:2];
  wire [TAG_BITS-1:0] w_tag = upd_pc[31:32-TAG_BITS];
  
  // Combinational read
  assign hit      = valid[r_idx] && (tag[r_idx] == r_tag);
  assign target_r = target[r_idx];
  
  always @(posedge clk) begin
    if (reset) begin
      valid <= {ENTRIES{1'b0}};
      // Do not reset tag/target RAMs
    end else if (upd_valid) begin
      valid[w_idx]  <= 1'b1;
      tag[w_idx]    <= w_tag;
      target[w_idx] <= {upd_target[31:2], 2'b00};
    end
  end
endmodule