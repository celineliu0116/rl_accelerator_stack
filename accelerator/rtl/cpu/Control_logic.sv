module control_logic(
    input [31:0] inst_x, // EX-stage instr (for EX controls) 
    input [31:0] inst_m, // M/WB stage instr (for WB controls)
    output br_un,
    output a_sel, b_sel,
    output [3:0] alu_op,
    output csr_op,
    output [1:0] wb_sel,
    output is_branch,
    output is_j,
    output reg_wen,
    output is_Store_x,
    output is_Load_x, is_JAL_x, is_JALR_x, is_U_x, is_I_x,
    output is_M_extension
);

// assign is_j = (inst_x[6:0] == 7'b110_1111) || (inst_x[6:0] == 7'b110_0111 && inst_x[14:12] == 3'b000);
// assign is_branch = (inst_x[6:0] == 7'b110_0011);
// assign br_un = inst_x[13];

wire [6:0] op_x = inst_x[6:0];
wire [2:0] f3_x = inst_x[14:12];

wire [6:0] op_m = inst_m[6:0];
wire [4:0] rd_m = inst_m[11:7];

wire [6:0] op_x_neg = ~op_x[6:0];
wire [6:0] op_m_neg = ~op_m[6:0];

wire is_R_x       = op_x_neg[6] & op_x[5] & op_x[4] & op_x_neg[2];
wire is_Ialu_x    = op_x_neg[6] & op_x_neg[5] & op_x[4] & op_x_neg[2];
assign is_Load_x  = op_x_neg[5] & op_x_neg[4] & op_x[1];
assign is_Store_x = op_x_neg[6] & op_x[5] & op_x_neg[4];
assign is_branch  = op_x[6] & op_x_neg[4] & op_x_neg[2];
assign is_JAL_x   = op_x[3];
assign is_JALR_x  = op_x[6] & op_x_neg[3] & op_x[2];
wire is_LUI_x     = op_x[5] & op_x[4] & op_x[2];
assign csr_op     = op_x[6] & op_x[4] ; // SYSTEM (CSR*/ECALL/…)
assign is_U_x     = op_x[4] | op_x[2];
assign is_I_x     = (op_x_neg[5] & op_x_neg[2]) | is_JALR_x;

// M-Extension: Opcode 0110011 (same as R-type) but funct7 is 0000001
// Normal R-type has funct7 0000000 or 0100000
// So we check for R-type opcode AND funct7[0] (bit 25 of inst_x) because 0000001 has LSB set.
// Actually, bit 25 is the 0-th bit of funct7.
assign is_M_extension = is_R_x & inst_x[25];

// Same for inst_m (WB controls)

// J/Branch flags (EX)
assign is_j       = (op_x[6] & op_x[2]) | op_x[3];

// Unsigned compare for BLTU/BGEU
assign br_un = f3_x[1];

assign a_sel = (op_x_neg[6] & op_x_neg[2]) | (op_x[5] & op_x[4]);

assign b_sel = is_R_x;

// ALU op comes from your decoder (good)
ALUdec ad(
    .funct(f3_x),
    .add_rshift_type(inst_x[30]),
    .is_R_x(is_R_x),
    .is_Ialu_x(is_Ialu_x),
    .alu_is_copy(is_LUI_x),
    .ALUop(alu_op)
);

// MEM/WB controls (use inst_m)
// Write-back mux
// 01: PC+4 (JAL/JALR)
// 10: ALU/CSR result (R, I-ALU, LUI, AUIPC, CSR)
// 11: Load data
// 00: none

assign wb_sel[1] = ~((op_m[6] & op_m[2]) | op_m[3]);
assign wb_sel[0] = op_m_neg[5] & op_m_neg[4]; // is_Load_m

// Register write enable (rd != x0)
assign reg_wen = (rd_m != 5'd0) & (~(op_m[5] & op_m_neg[4] & op_m_neg[2]));


endmodule