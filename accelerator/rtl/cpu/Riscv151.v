`include "const.vh"

// 4-stage pipeline: IF → ID → EX → WB
module Riscv151(
    input clk,
    input reset,

    // Memory system ports
    output [31:0] dcache_addr,
    output [31:0] icache_addr,
    output [3:0]  dcache_we,
    output        dcache_re,
    output        icache_re,
    output [31:0] dcache_din,
    input [31:0]  dcache_dout,
    input [63:0]  icache_dout,
    input         icache_valid,
    input [31:0]  icache_resp_addr,
    input         stall,
    output [31:0] csr
);

// ========================================================================
//  STAGE 1: INSTRUCTION FETCH (IF)
// ========================================================================

wire [31:0] pc_if;           // current PC (from Fetch)
wire        pred_if;         // BHT/BTB prediction for this cycle
wire [31:0] btb_target_if;   // BTB target for this cycle

// Redirect signals from EX stage
wire        mispredict;
wire [31:0] actual_target;
wire [31:0] pcx_plus4;
wire        taken;

// Predictor update from EX stage
wire        cf_upd_valid;
wire [31:0] cf_upd_pc;
wire        dual_issued;
wire        flush_id;
wire        lane1_issue_en;

assign icache_addr = pc_if;

// Combined stall: memory stall OR load-use hazard
wire load_use_hazard;
wire freeze_ext = stall | load_use_hazard;
wire ic_resp_valid = icache_valid;
wire ic_launch;

// Metadata for the request that will return on the next cycle.
reg [31:0] pc_launch_q, btb_launch_q;
reg        pred_launch_q;

reg  half_fetched;
wire have_bundle;
wire slip_stall;
wire stall_fetch = freeze_ext | slip_stall;
wire stall_front = freeze_ext | slip_stall | !have_bundle;
wire front_freeze = stall_front;

always @(posedge clk) begin
  if (reset | flush_id) begin
    half_fetched <= 1'b0;
  end else if (!freeze_ext && have_bundle) begin
    if (half_fetched)
      half_fetched <= 1'b0;
    else if (slip_stall)
      half_fetched <= 1'b1;
  end
end

assign icache_re = ~stall_fetch;
assign ic_launch = icache_re;

Fetch u_fetch (
  .clk            (clk),
  .reset          (reset),
  .dual_issued    (dual_issued),
  .taken          (taken),
  .stall          (stall_fetch),
  .mispredict     (mispredict),
  .actual_target  (actual_target),
  .pcx_plus4      (pcx_plus4),
  .cf_upd_valid   (cf_upd_valid),
  .cf_upd_pc      (cf_upd_pc),
  .cf_upd_taken   (taken),
  .cf_upd_target  (actual_target),
  .pc_req         (pc_if),
  .pred_req       (pred_if),
  .btb_target_out (btb_target_if)
);

always @(posedge clk) begin
  if (reset) begin
    pc_launch_q <= `PC_RESET;
    pred_launch_q <= 1'b0;
    btb_launch_q <= 32'b0;
  end else if (ic_launch) begin
    pc_launch_q <= pc_if;
    pred_launch_q <= pred_if;
    btb_launch_q <= btb_target_if;
  end
end

// Buffer response bundle only when front-end cannot consume it.
reg [63:0] held_bundle;
reg [31:0] held_pc_base, held_btb;
reg        held_pred;
reg        holding;
wire [31:0] resp_pc_base = icache_resp_addr;
wire        resp_pred = pred_launch_q;
wire [31:0] resp_btb = btb_launch_q;

always @(posedge clk) begin
  if (reset || flush_id) begin
    held_bundle <= {`INSTR_NOP, `INSTR_NOP};
    held_pc_base <= `PC_RESET;
    held_pred <= 1'b0;
    held_btb <= 32'b0;
    holding <= 1'b0;
  end else if (ic_resp_valid && stall_front && ~holding) begin
    held_bundle <= icache_dout;
    held_pc_base <= resp_pc_base;
    held_pred <= resp_pred;
    held_btb <= resp_btb;
    holding <= 1'b1;
  end else if (~stall_front) begin
    holding <= 1'b0;
  end
end

assign have_bundle = holding | ic_resp_valid;
wire [31:0] inst_fetched = holding ? held_bundle[31:0] : icache_dout[31:0];
wire [31:0] inst_fetched_1 = holding ? held_bundle[63:32] : icache_dout[63:32];
wire [31:0] pc_bundle_base = holding ? held_pc_base : resp_pc_base;
wire        pred_bundle = holding ? held_pred : resp_pred;
wire [31:0] btb_bundle = holding ? held_btb : resp_btb;

// ========================================================================
//  IF/ID PIPELINE REGISTER
// ========================================================================

// Flush: mispredict needs 2 cycles of NOPs (branch resolved in EX, 2 stages from IF)
reg flush_delay;
always @(posedge clk) begin
  if (reset)        flush_delay <= 1'b0;
  else if (!stall_front)  flush_delay <= mispredict;
end

assign flush_id = mispredict | flush_delay;

wire [31:0] inst_id = (flush_id || !have_bundle) ? `INSTR_NOP :
                      (half_fetched ? inst_fetched_1 : inst_fetched);
wire [31:0] inst_id_1 = (flush_id || !have_bundle) ? `INSTR_NOP :
                        (half_fetched ? `INSTR_NOP : inst_fetched_1);

wire [31:0] pc_id_base;
wire [31:0] pc_id;
wire [31:0] pc_id_1;
wire        pred_id;
wire [31:0] btb_target_id;
wire        squash_lane1;

assign pc_id_base = pc_bundle_base;
assign pc_id = half_fetched ? (pc_bundle_base + 32'd4) : pc_bundle_base;
assign pc_id_1 = pc_id + 32'd4;
assign pred_id = pred_bundle;
assign btb_target_id = btb_bundle;

// ========================================================================
//  STAGE 2: INSTRUCTION DECODE (ID)
// ========================================================================

// --- Control Logic (decode from inst_id) ---
wire        a_sel, b_sel;
wire [3:0]  alu_op;
wire        csr_op_id;
wire [1:0]  wb_sel;
wire        is_branch_id, is_j_id;
wire        is_Store_id, is_Load_id;
wire        is_JAL_id, is_JALR_id, is_U_id, is_I_id;
wire        is_M_extension_id;
wire        br_un_id;
wire        reg_wen_id_raw;  // raw decode, not used for hazard (from inst_id)

control_logic u_cl(
    .inst_x    (inst_id),
    .inst_m    (inst_wb),       // WB-stage instruction for wb_sel/reg_wen
    .br_un     (br_un_id),
    .a_sel     (a_sel),
    .b_sel     (b_sel),
    .alu_op    (alu_op),
    .csr_op    (csr_op_id),
    .wb_sel    (wb_sel),
    .is_branch (is_branch_id),
    .is_j      (is_j_id),
    .reg_wen   (reg_wen_wb),    // write enable for WB stage (from inst_m/inst_wb)
    .is_Store_x(is_Store_id),
    .is_Load_x (is_Load_id),
    .is_JAL_x  (is_JAL_id),
    .is_JALR_x (is_JALR_id),
    .is_U_x    (is_U_id),
    .is_I_x    (is_I_id),
    .is_M_extension(is_M_extension_id)
);

// --- Register File ---
wire [31:0] reg_read_data1_0, reg_read_data2_0;
wire [31:0] reg_read_data1_1, reg_read_data2_1;
wire [31:0] wb_val;
wire [31:0] wb_val_1;
wire        reg_wen_wb;
wire        reg_wen_wb_1;
wire [31:0] inst_wb;
wire [31:0] inst_wb_1;

RegFile u_regfile (
  .clk     (clk),
  .we0     (reg_wen_wb),
  .wdata0  (wb_val),
  .rd0     (inst_wb[11:7]),
  .we1     (reg_wen_wb_1),
  .wdata1  (wb_val_1),
  .rd1     (inst_wb_1[11:7]),
  .rs1_0   (inst_id[19:15]),
  .rs2_0   (inst_id[24:20]),
  .rdata1_0(reg_read_data1_0),
  .rdata2_0(reg_read_data2_0),
  .rs1_1   (inst_id_1[19:15]),
  .rs2_1   (inst_id_1[24:20]),
  .rdata1_1(reg_read_data1_1),
  .rdata2_1(reg_read_data2_1)
);

// --- Immediate Generation ---
wire [31:0] immediate_id;
imm_gen u_immgen(
    .imm       (immediate_id),
    .inst      (inst_id),
    .is_I_x    (is_I_id),
    .is_Store_x(is_Store_id),
    .is_branch (is_branch_id),
    .is_jal    (is_JAL_id),
    .is_U_x    (is_U_id),
    .csr_op    (csr_op_id)
);

// --- Hazard Unit ---
wire [1:0] fwd_a_0_sel, fwd_b_0_sel;
wire [1:0] fwd_a_1_sel, fwd_b_1_sel;
wire [31:0] alu_out_ex;  // from EX stage for forwarding
wire [31:0] alu_out_ex_1;
wire [31:0] inst_ex;
wire [31:0] inst_ex_1;
wire        is_Load_ex;
wire        is_M_extension_ex;

// reg_wen for EX-stage instruction (simple decode: not a store, not a branch)
wire        reg_wen_ex_raw;
wire [4:0]  rd_ex = inst_ex[11:7];
wire        lane1_simple_alu_hz = ((inst_id_1[6:0] == `OPC_ARI_ITYPE) ||
                                  ((inst_id_1[6:0] == `OPC_ARI_RTYPE) &&
                                   (inst_id_1[31:25] != 7'b0000001)));
wire        lane0_pairable_hz = ~is_branch_id & ~is_j_id & ~is_Load_id &
                                ~is_Store_id & ~csr_op_id & ~is_M_extension_id;
`ifdef ENABLE_DUAL_ISSUE
wire        lane1_eligible_hz = lane1_simple_alu_hz & lane0_pairable_hz;
`else
wire        lane1_eligible_hz = 1'b0;
`endif
wire [4:0]  rs1_id_1_hz = lane1_eligible_hz ? inst_id_1[19:15] : 5'd0;
wire [4:0]  rs2_id_1_hz = lane1_eligible_hz ? inst_id_1[24:20] : 5'd0;
wire [4:0]  rd_id_0_hz  = inst_id[11:7];
wire [4:0]  rd_id_1_hz  = lane1_eligible_hz ? inst_id_1[11:7] : 5'd0;
wire        reg_wen_id_0_hz = ~is_Store_id & ~is_branch_id & (rd_id_0_hz != 5'd0);
wire        reg_wen_id_1_hz = lane1_eligible_hz & (rd_id_1_hz != 5'd0);
wire        is_Load_id_0_hz = is_Load_id;
wire        ex0_no_fwd_hazard = is_Load_ex | is_M_extension_ex;
wire [4:0]  rd_ex_1 = inst_ex_1[11:7];
wire        reg_wen_ex_1_raw;

DualHazardUnit u_dual_hazard(
    .rs1_id_0       (inst_id[19:15]),
    .rs2_id_0       (inst_id[24:20]),
    .rd_id_0        (rd_id_0_hz),
    .reg_wen_id_0   (reg_wen_id_0_hz),
    .is_Load_id_0   (is_Load_id_0_hz),
    .rs1_id_1       (rs1_id_1_hz),
    .rs2_id_1       (rs2_id_1_hz),
    .rd_id_1        (rd_id_1_hz),
    .reg_wen_id_1   (reg_wen_id_1_hz),
    .lane1_eligible (lane1_eligible_hz),
    .rd_ex_0        (rd_ex),
    .reg_wen_ex_0   (reg_wen_ex_raw),
    // Treat M-extension like load-use for hazard purposes: no EX->ID forward.
    .is_Load_ex_0   (ex0_no_fwd_hazard),
    .rd_ex_1        (rd_ex_1),
    .reg_wen_ex_1   (reg_wen_ex_1_raw),
    .fwd_a_0        (fwd_a_0_sel),
    .fwd_b_0        (fwd_b_0_sel),
    .fwd_a_1        (fwd_a_1_sel),
    .fwd_b_1        (fwd_b_1_sel),
    .load_use_hazard(load_use_hazard),
    .squash_lane1   (squash_lane1)
);

assign lane1_issue_en = lane1_eligible_hz & ~squash_lane1;
// Never "slip" lane1 when lane0 is control-flow predicted taken; lane1 is wrong-path.
assign slip_stall = have_bundle && !half_fetched && !flush_id &&
                    !freeze_ext && !lane1_issue_en &&
                    !((is_branch_id | is_j_id) & pred_id);
assign dual_issued = lane1_issue_en & ~flush_id & ~front_freeze;

// --- Forwarding Mux ---
// EX forwarding takes priority; WB forwarding handled by RegFile bypass
wire [31:0] reg1_id = (fwd_a_0_sel == 2'b01) ? alu_out_ex :
                      (fwd_a_0_sel == 2'b10) ? alu_out_ex_1 :
                                                reg_read_data1_0;
wire [31:0] reg2_id = (fwd_b_0_sel == 2'b01) ? alu_out_ex :
                      (fwd_b_0_sel == 2'b10) ? alu_out_ex_1 :
                                                reg_read_data2_0;

wire [31:0] reg1_id_1 = (fwd_a_1_sel == 2'b01) ? alu_out_ex :
                        (fwd_a_1_sel == 2'b10) ? alu_out_ex_1 :
                                                  reg_read_data1_1;
wire [31:0] reg2_id_1 = (fwd_b_1_sel == 2'b01) ? alu_out_ex :
                        (fwd_b_1_sel == 2'b10) ? alu_out_ex_1 :
                                                  reg_read_data2_1;

// --- ALU Operand Selection ---
// a_sel: 0 = PC, 1 = reg1
// b_sel: 0 = immediate, 1 = reg2
wire [31:0] alu_input_a = a_sel ? reg1_id : pc_id;
wire [31:0] alu_input_b = b_sel ? reg2_id : immediate_id;
wire [3:0]  alu_op_id_1;
wire [31:0] immediate_id_1 = {{20{inst_id_1[31]}}, inst_id_1[31:20]};
wire [31:0] alu_input_a_1 = reg1_id_1;
wire [31:0] alu_input_b_1 = (inst_id_1[6:0] == `OPC_ARI_RTYPE) ? reg2_id_1 : immediate_id_1;

ALUdec u_aludec_lane1(
  .funct(inst_id_1[14:12]),
  .add_rshift_type(inst_id_1[30]),
  .is_R_x(inst_id_1[6:0] == `OPC_ARI_RTYPE),
  .is_Ialu_x(inst_id_1[6:0] == `OPC_ARI_ITYPE),
  .alu_is_copy(1'b0),
  .ALUop(alu_op_id_1)
);

// --- Predictor update wiring ---
assign cf_upd_pc = pc_ex_out;
wire        is_branch_ex, is_j_ex;
assign cf_upd_valid = (is_branch_ex | is_j_ex) & ~stall;

// ========================================================================
//  ID/EX PIPELINE REGISTER
// ========================================================================

// When load-use hazard, insert bubble into EX (NOP + clear controls)
wire bubble_ex = load_use_hazard & ~stall;

// When mispredict or bubble, flush the ID/EX register
wire flush_idex = mispredict | bubble_ex;

wire [31:0] pc_ex_out;
wire [31:0] alu_a_ex, alu_b_ex;
wire [31:0] alu_a_ex_1, alu_b_ex_1;
wire [3:0]  alu_op_ex;
wire [3:0]  alu_op_ex_1;
wire [31:0] reg1_ex, reg2_ex;
wire [31:0] immediate_ex;
wire        pred_ex;
wire [31:0] btb_target_ex;
wire        is_jal_ex, is_jalr_ex, is_Store_ex;
wire        csr_op_ex, br_un_ex;
reg         issue_ex_1_r;

// Instruction register with flush support
reg [31:0] inst_ex_r;
reg [31:0] inst_ex_1_r;
always @(posedge clk) begin
  if (reset | (flush_idex & ~stall)) begin
    inst_ex_r <= `INSTR_NOP;
    inst_ex_1_r <= `INSTR_NOP;
  end else if (!stall) begin
    inst_ex_r <= inst_id;
    inst_ex_1_r <= lane1_issue_en ? inst_id_1 : `INSTR_NOP;
  end
end
assign inst_ex = inst_ex_r;
assign inst_ex_1 = inst_ex_1_r;

// PC
pipeline_reg #(32) u_pc_idex (
  .clk(clk), .reset(reset), .stall(stall),
  .in(pc_id), .out(pc_ex_out)
);

// ALU operands
pipeline_reg #(32) u_alua_idex (
  .clk(clk), .reset(reset), .stall(stall),
  .in(alu_input_a), .out(alu_a_ex)
);

pipeline_reg #(32) u_alub_idex (
  .clk(clk), .reset(reset), .stall(stall),
  .in(alu_input_b), .out(alu_b_ex)
);

pipeline_reg #(32) u_alua1_idex (
  .clk(clk), .reset(reset), .stall(stall),
  .in(alu_input_a_1), .out(alu_a_ex_1)
);

pipeline_reg #(32) u_alub1_idex (
  .clk(clk), .reset(reset), .stall(stall),
  .in(alu_input_b_1), .out(alu_b_ex_1)
);

// ALU op
reg [3:0] alu_op_ex_r;
reg [3:0] alu_op_ex_1_r;
always @(posedge clk) begin
  if (reset | (flush_idex & ~stall)) begin
    alu_op_ex_r <= 4'd0;
    alu_op_ex_1_r <= 4'd0;
    issue_ex_1_r <= 1'b0;
  end else if (!stall) begin
    alu_op_ex_r <= alu_op;
    alu_op_ex_1_r <= alu_op_id_1;
    issue_ex_1_r <= lane1_issue_en;
  end
end
assign alu_op_ex = alu_op_ex_r;
assign alu_op_ex_1 = alu_op_ex_1_r;

// Register values (for branch compare + store data)
pipeline_reg #(32) u_reg1_idex (
  .clk(clk), .reset(reset), .stall(stall),
  .in(reg1_id), .out(reg1_ex)
);

pipeline_reg #(32) u_reg2_idex (
  .clk(clk), .reset(reset), .stall(stall),
  .in(reg2_id), .out(reg2_ex)
);

// Immediate (for branch/jump target computation in EX)
pipeline_reg #(32) u_imm_idex (
  .clk(clk), .reset(reset), .stall(stall),
  .in(immediate_id), .out(immediate_ex)
);

// Prediction
pipeline_reg #(1) u_pred_idex (
  .clk(clk), .reset(reset), .stall(stall),
  .in(pred_id), .out(pred_ex)
);

pipeline_reg #(32) u_btb_idex (
  .clk(clk), .reset(reset), .stall(stall),
  .in(btb_target_id), .out(btb_target_ex)
);

// Control signals (with flush support)
reg is_branch_ex_r, is_j_ex_r, is_jal_ex_r, is_jalr_ex_r;
reg is_Store_ex_r, is_Load_ex_r;
reg is_M_extension_ex_r, csr_op_ex_r, br_un_ex_r;
reg reg_wen_ex_r;

always @(posedge clk) begin
  if (reset | (flush_idex & ~stall)) begin
    is_branch_ex_r     <= 1'b0;
    is_j_ex_r          <= 1'b0;
    is_jal_ex_r        <= 1'b0;
    is_jalr_ex_r       <= 1'b0;
    is_Store_ex_r      <= 1'b0;
    is_Load_ex_r       <= 1'b0;
    is_M_extension_ex_r <= 1'b0;
    csr_op_ex_r        <= 1'b0;
    br_un_ex_r         <= 1'b0;
    reg_wen_ex_r       <= 1'b0;
  end else if (!stall) begin
    is_branch_ex_r     <= is_branch_id;
    is_j_ex_r          <= is_j_id;
    is_jal_ex_r        <= is_JAL_id;
    is_jalr_ex_r       <= is_JALR_id;
    is_Store_ex_r      <= is_Store_id;
    is_Load_ex_r       <= is_Load_id;
    is_M_extension_ex_r <= is_M_extension_id;
    csr_op_ex_r        <= csr_op_id;
    br_un_ex_r         <= br_un_id;
    // reg_wen for hazard detection: check that instruction is not store/branch and rd != x0
    reg_wen_ex_r       <= ~is_Store_id & ~is_branch_id & (inst_id[11:7] != 5'd0);
  end
end

assign is_branch_ex     = is_branch_ex_r;
assign is_j_ex          = is_j_ex_r;
assign is_jal_ex        = is_jal_ex_r;
assign is_jalr_ex       = is_jalr_ex_r;
assign is_Store_ex      = is_Store_ex_r;
assign is_Load_ex       = is_Load_ex_r;
assign is_M_extension_ex = is_M_extension_ex_r;
assign csr_op_ex        = csr_op_ex_r;
assign br_un_ex         = br_un_ex_r;
assign reg_wen_ex_raw   = reg_wen_ex_r;
assign reg_wen_ex_1_raw = issue_ex_1_r & (rd_ex_1 != 5'd0);

// ========================================================================
//  STAGE 3: EXECUTE (EX)
// ========================================================================

wire [31:0] data_w_ex;
wire [3:0]  wmask_ex;

execute u_ex(
    .clk              (clk),
    .reset            (reset),
    .pc_ex            (pc_ex_out),
    .alu_a_ex         (alu_a_ex),
    .alu_b_ex         (alu_b_ex),
    .alu_op_ex        (alu_op_ex),
    .inst_ex          (inst_ex),
    .reg1_ex          (reg1_ex),
    .reg2_ex          (reg2_ex),
    .immediate_ex     (immediate_ex),
    .pred_ex          (pred_ex),
    .btb_target_ex    (btb_target_ex),
    .is_branch_ex     (is_branch_ex),
    .is_j_ex          (is_j_ex),
    .is_jal_ex        (is_jal_ex),
    .is_jalr_ex       (is_jalr_ex),
    .is_Store_ex      (is_Store_ex),
    .is_M_extension_ex(is_M_extension_ex),
    .csr_op_ex        (csr_op_ex),
    .br_un_ex         (br_un_ex),
    .alu_out          (alu_out_ex),
    .data_w           (data_w_ex),
    .csr              (csr),
    .pc_add_4         (pcx_plus4),
    .taken            (taken),
    .target_addr      (actual_target),
    .mispredict       (mispredict),
    .wmask            (wmask_ex)
);

ALU u_alu_lane1(
  .A(alu_a_ex_1),
  .B(alu_b_ex_1),
  .ALUop(alu_op_ex_1),
  .Out(alu_out_ex_1)
);

// Dcache interface driven from EX
assign dcache_addr = alu_out_ex;
assign dcache_re   = is_Load_ex;
assign dcache_we   = wmask_ex;
assign dcache_din  = data_w_ex;

// ========================================================================
//  EX/WB PIPELINE REGISTER
// ========================================================================

wire [31:0] alu_wb;
wire [31:0] pc_add_4_wb;
wire [31:0] alu_wb_1;
wire        valid_wb_1;

pipeline_reg #(32) u_alu_exwb (
  .clk(clk), .reset(reset), .stall(stall),
  .in(alu_out_ex), .out(alu_wb)
);

pipeline_reg #(32) u_pc4_exwb (
  .clk(clk), .reset(reset), .stall(stall),
  .in(pcx_plus4), .out(pc_add_4_wb)
);

pipeline_reg #(32) u_inst_exwb (
  .clk(clk), .reset(reset), .stall(stall),
  .in(inst_ex), .out(inst_wb)
);

pipeline_reg #(32) u_alu1_exwb (
  .clk(clk), .reset(reset), .stall(stall),
  .in(alu_out_ex_1), .out(alu_wb_1)
);

pipeline_reg #(32) u_inst1_exwb (
  .clk(clk), .reset(reset), .stall(stall),
  .in(inst_ex_1), .out(inst_wb_1)
);

pipeline_reg #(1) u_valid1_exwb (
  .clk(clk), .reset(reset), .stall(stall),
  .in(issue_ex_1_r), .out(valid_wb_1)
);

// ========================================================================
//  STAGE 4: WRITEBACK (WB)
// ========================================================================

Writeback u_writeback (
  .dcache_dout   (dcache_dout),
  .alu_wb        (alu_wb),
  .pc_add_4_wb   (pc_add_4_wb),
  .inst_wb       (inst_wb),
  .wb_sel        (wb_sel),
  .wb_val        (wb_val)
);

assign wb_val_1 = alu_wb_1;
assign reg_wen_wb_1 = valid_wb_1 & (inst_wb_1[11:7] != 5'd0);

`ifndef SYNTHESIS
// ----------------------------
// Front-end / dual-issue tripwires (simulation only)
// ----------------------------

// Track lane1 wrong-path intent across ID->EX->WB.
reg lane1_wrongpath_ex_q;
reg lane1_wrongpath_wb_q;

always @(posedge clk) begin
  if (reset | (flush_idex & ~stall)) begin
    lane1_wrongpath_ex_q <= 1'b0;
  end else if (!stall) begin
    // If lane0 is control-flow, lane1 of this bundle must never commit.
    lane1_wrongpath_ex_q <= is_branch_id | is_j_id;
  end
end

always @(posedge clk) begin
  if (reset) begin
    lane1_wrongpath_wb_q <= 1'b0;
  end else if (!stall) begin
    lane1_wrongpath_wb_q <= lane1_wrongpath_ex_q;
  end
end

always @(posedge clk) begin
  if (!reset) begin
    // A) Never advance front-end without an available bundle.
    if (!stall_front && !have_bundle) begin
      $fatal(1, "IF/ID advanced without have_bundle");
    end

    // B) Lane1 must never issue when lane0 is control-flow.
    if (lane1_issue_en && (is_branch_id || is_j_id)) begin
      $fatal(1, "Lane1 issued alongside control-flow in lane0");
    end

    // C) Whenever decode is in a flush window and front-end advances, decode must see NOP.
    if (!stall_front && flush_id && (inst_id != `INSTR_NOP)) begin
      $fatal(1, "Flush failed: inst_id not NOP while flush_id is asserted");
    end

    // D) Wrong-path lane1 from control-flow bundle must never write back.
    if (lane1_wrongpath_wb_q && reg_wen_wb_1) begin
      $fatal(1, "Wrong-path lane1 committed a WB write");
    end
  end
end
`endif

endmodule

// ---- Generic pipeline register ----
module pipeline_reg #(parameter N = 32) (
  input wire         clk,
  input wire         reset,
  input wire         stall,
  input wire [N-1:0] in,
  output reg [N-1:0] out
);
  always @(posedge clk) begin
    if (reset)
      out <= {N{1'b0}};
    else if(stall)
      out <= out;
    else
      out <= in;
  end
endmodule
