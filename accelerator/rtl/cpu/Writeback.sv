module Writeback(
    input [31:0] dcache_dout,
    input [31:0] alu_wb,
    input [31:0] pc_add_4_wb,
    input [31:0] inst_wb,
    input [1:0] wb_sel,
    output [31:0] wb_val
);

reg [31:0] rdata;

// funct3 for loads: LB=000, LH=001, LW=010, LBU=100, LHU=101
wire [2:0] f3 = inst_wb[14:12];

// addr[1:0] tells which byte in the 32-bit word (00=lowest, 11=highest)
// addr[1] alone tells which halfword (0=lower 16 bits, 1=upper 16 bits)
wire [1:0] byte_off = alu_wb[1:0];
wire       half_sel = alu_wb[1];

wire [31:0] dout_sh = dcache_dout >> (8*byte_off);
wire [7:0]  byte_val    = dout_sh[7:0];
wire [15:0] half    = half_sel ? dcache_dout[31:16] : dcache_dout[15:0];

always @(*) begin
  if(f3[1]) rdata = dcache_dout;
  else begin
    if(f3[2]) begin
      if(f3[0]) rdata = {16'b0, half};             // LHU
      else rdata = {24'b0, byte_val};              // LBU
    end
    else begin
      if(f3[0]) rdata = {{16{half[15]}}, half};    // LH
      else rdata = {{24{byte_val[7]}}, byte_val};  // LB
    end
  end
end

assign wb_val = wb_sel[1] ? 
                (wb_sel[0] ? rdata : alu_wb) : pc_add_4_wb ;

// Validate LB/LH extension properties when memory result is selected
`ifndef SYNTHESIS
  always_comb begin
    // Only check when memory path is selected
    if (wb_sel == 2'b11) begin
      // LB: sign-extend from bit 7
      if (f3 == 3'b000) begin
        assert (rdata[31:8] == {24{rdata[7]}})
          else $fatal(1, "LB sign-extension wrong: rdata=%0h", rdata);
      end
      // LH: sign-extend from bit 15
      if (f3 == 3'b001) begin
        assert (rdata[31:16] == {16{rdata[15]}})
          else $fatal(1, "LH sign-extension wrong: rdata=%0h", rdata);
      end
    end
  end
`endif

endmodule