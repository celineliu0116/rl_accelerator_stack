// RegFile - hardwires x0 = 0, and does synchronous write on the rising edge
// includes read-during-write bypass so a WB in the same cycle is visible to decode reads.
module RegFile(
    input clk,
    input we0,               // lane 0 write enable
    input  [31:0] wdata0,    // lane 0 write data
    input  [4:0] rd0,        // lane 0 destination register
    input we1,               // lane 1 write enable
    input  [31:0] wdata1,    // lane 1 write data
    input  [4:0] rd1,        // lane 1 destination register
    input  [4:0] rs1_0,      // lane 0 source register 1
    input  [4:0] rs2_0,      // lane 0 source register 2
    output [31:0] rdata1_0,  // lane 0 read data 1
    output [31:0] rdata2_0,  // lane 0 read data 2
    input  [4:0] rs1_1,      // lane 1 source register 1
    input  [4:0] rs2_1,      // lane 1 source register 2
    output [31:0] rdata1_1,  // lane 1 read data 1
    output [31:0] rdata2_1   // lane 1 read data 2
  );
  reg [31:0] regfile [31:0];

  wire we0_eff = we0 && (rd0 != 5'd0);
  wire we1_eff = we1 && (rd1 != 5'd0) && !(we0_eff && (rd0 == rd1));

  // synchronous write on posedge
  initial begin
    for (integer i = 0; i < 32; i = i + 1) begin
      regfile[i] = 32'd0;
    end
  end
  always @(posedge clk) begin
    if (we0_eff)
      regfile[rd0] <= wdata0;
    if (we1_eff)
      regfile[rd1] <= wdata1;
    regfile[0] <= 32'd0;
  end
  
  // combinational reads
  wire [31:0] r1_0_raw = (rs1_0 == 5'd0) ? 32'd0 : regfile[rs1_0];
  wire [31:0] r2_0_raw = (rs2_0 == 5'd0) ? 32'd0 : regfile[rs2_0];
  wire [31:0] r1_1_raw = (rs1_1 == 5'd0) ? 32'd0 : regfile[rs1_1];
  wire [31:0] r2_1_raw = (rs2_1 == 5'd0) ? 32'd0 : regfile[rs2_1];

  // same-cycle read-during-write bypass; lane 0 write has priority on ties
  wire hit_w0_rs1_0 = we0_eff && (rd0 == rs1_0);
  wire hit_w1_rs1_0 = we1_eff && (rd1 == rs1_0) && !hit_w0_rs1_0;
  wire hit_w0_rs2_0 = we0_eff && (rd0 == rs2_0);
  wire hit_w1_rs2_0 = we1_eff && (rd1 == rs2_0) && !hit_w0_rs2_0;
  wire hit_w0_rs1_1 = we0_eff && (rd0 == rs1_1);
  wire hit_w1_rs1_1 = we1_eff && (rd1 == rs1_1) && !hit_w0_rs1_1;
  wire hit_w0_rs2_1 = we0_eff && (rd0 == rs2_1);
  wire hit_w1_rs2_1 = we1_eff && (rd1 == rs2_1) && !hit_w0_rs2_1;

  assign rdata1_0 = hit_w0_rs1_0 ? wdata0 : (hit_w1_rs1_0 ? wdata1 : r1_0_raw);
  assign rdata2_0 = hit_w0_rs2_0 ? wdata0 : (hit_w1_rs2_0 ? wdata1 : r2_0_raw);
  assign rdata1_1 = hit_w0_rs1_1 ? wdata0 : (hit_w1_rs1_1 ? wdata1 : r1_1_raw);
  assign rdata2_1 = hit_w0_rs2_1 ? wdata0 : (hit_w1_rs2_1 ? wdata1 : r2_1_raw);

  // SystemVerilog assertions: x0 must always read as zero and regfile[0] must never be written to non-zero.
  // Immediate assertion sampled on clock edge to catch synchronous write violations.

  // initial regfile[0] = 32'd0; // or initialize entire array
  `ifndef SYNTHESIS
  // x0 must always be 0 (no reset gating)
  assert property (@(posedge clk) regfile[0] == 32'd0)
    else $fatal("SVA: x0 violated (regfile[0]=%h)", regfile[0]);

  // never write x0
  assert property (@(posedge clk) !(we0 && rd0 == 5'd0))
    else $fatal("SVA: lane0 attempted write to x0");
  assert property (@(posedge clk) !(we1 && rd1 == 5'd0))
    else $fatal("SVA: lane1 attempted write to x0");
`endif

endmodule
