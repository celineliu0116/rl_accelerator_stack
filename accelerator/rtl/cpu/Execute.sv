// Execute stage for the 4-stage pipeline (IF → ID → EX → WB)
// Receives pre-computed ALU operands from ID stage.
// Performs: ALU, Multiplier, Branch compare, Branch resolve, Store align, CSR, Wmask

module execute(
  input         clk,
  input         reset,

  // From ID/EX pipeline register
  input [31:0]  pc_ex,
  input [31:0]  alu_a_ex,        // pre-computed ALU operand A
  input [31:0]  alu_b_ex,        // pre-computed ALU operand B
  input [3:0]   alu_op_ex,
  input [31:0]  inst_ex,
  input [31:0]  reg1_ex,         // forwarded rs1 (for branch compare + store)
  input [31:0]  reg2_ex,         // forwarded rs2 (for branch compare + store)
  input [31:0]  immediate_ex,    // immediate (for branch/jump target)
  input         pred_ex,         // prediction bit from BHT/BTB
  input [31:0]  btb_target_ex,   // BTB target address

  // Control signals (from ID, piped)
  input         is_branch_ex,
  input         is_j_ex,
  input         is_jal_ex,
  input         is_jalr_ex,
  input         is_Store_ex,
  input         is_M_extension_ex,
  input         csr_op_ex,
  input         br_un_ex,

  // Outputs
  output [31:0] alu_out,
  output reg [31:0] data_w,
  output reg [31:0] csr,
  output [31:0] pc_add_4,
  output        taken,
  output [31:0] target_addr,
  output        mispredict,
  output [3:0]  wmask
);

assign pc_add_4 = pc_ex + 32'd4;

// ---- Branch comparator ----
wire br_eq, br_lt;
branch_comp bc(
  .reg1(reg1_ex), .reg2(reg2_ex),
  .br_un(br_un_ex), .br_eq(br_eq), .br_lt(br_lt)
);

// ---- ALU ----
wire [31:0] alu_res_simple;
ALU alu_inst(.A(alu_a_ex), .B(alu_b_ex), .ALUop(alu_op_ex), .Out(alu_res_simple));

// ---- Multiplier ----
wire [31:0] mul_res;
Multiplier mul_inst(
  .m_valid(is_M_extension_ex),
  .A(alu_a_ex), .B(alu_b_ex),
  .funct3(inst_ex[14:12]),
  .result(mul_res)
);

assign alu_out = is_M_extension_ex ? mul_res : alu_res_simple;

// ---- CSR ----
wire csr_is_imm = csr_op_ex && inst_ex[14];
wire [31:0] csr_wdata = csr_is_imm ? {27'b0, inst_ex[19:15]} : reg1_ex;
always @(posedge clk) begin
    if (reset)      csr <= 32'b0;
    else if(csr_op_ex) csr <= csr_wdata;
end

// ---- Branch / jump target ----
wire [31:0] branch_target = pc_ex + immediate_ex;
wire [31:0] temp_j_target = (is_jal_ex ? pc_ex : reg1_ex) + immediate_ex;
wire [31:0] j_target = {temp_j_target[31:1], 1'b0};

assign target_addr = is_branch_ex ? branch_target :
                     is_j_ex      ? j_target      : 32'b0;

// ---- Branch resolution ----
wire [2:0] f3_ex = inst_ex[14:12];

wire beq  = is_branch_ex && (f3_ex == 3'b000);
wire bne  = is_branch_ex && (f3_ex == 3'b001);
wire blt  = is_branch_ex && (f3_ex == 3'b100);
wire bge  = is_branch_ex && (f3_ex == 3'b101);
wire bgeu = is_branch_ex && (f3_ex == 3'b111);
wire bltu = is_branch_ex && (f3_ex == 3'b110);

wire branch_taken = (beq && br_eq) || (bne && !br_eq) ||
                    (blt && br_lt) || (bge && !br_lt) ||
                    (bgeu && !br_lt) || (bltu && br_lt);

assign taken = is_j_ex | branch_taken;

wire jump_mispredict = is_j_ex && (!pred_ex || (btb_target_ex != target_addr));
assign mispredict = (is_branch_ex && ((taken ^ pred_ex) ||
                     (pred_ex && (btb_target_ex != target_addr)))) |
                    jump_mispredict;

// ---- Store data alignment ----
always @(*) begin
    if (is_Store_ex) begin
        if (f3_ex[1])      data_w = reg2_ex;
        else if (f3_ex[0]) data_w = alu_out[1] ? {reg2_ex[15:0], 16'd0} : {16'd0, reg2_ex[15:0]};
        else begin
            case (alu_out[1:0])
                2'b00:   data_w = {24'd0, reg2_ex[7:0]};
                2'b01:   data_w = {16'd0, reg2_ex[7:0], 8'd0};
                2'b10:   data_w = {8'd0,  reg2_ex[7:0], 16'd0};
                2'b11:   data_w = {reg2_ex[7:0], 24'd0};
                default: data_w = 32'b0;
            endcase
        end
    end else data_w = 32'b0;
end

// ---- Store write mask ----
assign wmask = ~is_Store_ex ? 4'd0 :
               (f3_ex == 3'b000) ?
                   ((alu_out[1:0] == 2'b00) ? 4'b0001 :
                    (alu_out[1:0] == 2'b01) ? 4'b0010 :
                    (alu_out[1:0] == 2'b10) ? 4'b0100 :
                    (alu_out[1:0] == 2'b11) ? 4'b1000 : 4'b0000) :
               (f3_ex == 3'b001) ? ((alu_out[1]) ? 4'b1100 : 4'b0011) :
               (f3_ex == 3'b010) ? 4'b1111 : 4'b0000;

endmodule

module imm_gen(
    output [31:0] imm,
    input [31:0] inst,
    input is_I_x, is_Store_x, is_branch, is_jal, is_U_x, csr_op
);

assign imm = is_I_x ? {{20{inst[31]}}, inst[31:20]} :  // I-type (Load)
             is_Store_x ? {{20{inst[31]}}, inst[31:25], inst[11:7]} :  // S-type
             is_branch ? {{19{inst[31]}}, inst[31], inst[7], inst[30:25], inst[11:8], 1'b0} :  // B-type
             is_jal ? {{11{inst[31]}}, inst[31], inst[19:12], inst[20], inst[30:21], 1'b0} :  // J-type
             is_U_x  ? {inst[31:12], 12'b0} :  // U-type
             csr_op ? {27'b0, inst[19:15]} : 32'b0; // CSR-type

endmodule
