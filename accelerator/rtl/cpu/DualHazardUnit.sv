// DualHazardUnit.sv — Hazard detection and forwarding for dual-issue pipeline
// Handles:
// - Inter-lane RAW/WAW hazards in ID stage
// - Dual-lane EX→ID forwarding (2-bit select: regfile/EX0/EX1)
// - Load-use stall detection for both lanes
// - Lane 1 squash decision

module DualHazardUnit(
    // ID-stage instruction fields (Lane 0)
    input [4:0] rs1_id_0,
    input [4:0] rs2_id_0,
    input [4:0] rd_id_0,
    input       reg_wen_id_0,
    input       is_Load_id_0,

    // ID-stage instruction fields (Lane 1)
    input [4:0] rs1_id_1,
    input [4:0] rs2_id_1,
    input [4:0] rd_id_1,
    input       reg_wen_id_1,
    input       lane1_eligible,  // Lane 1 is simple ALU instruction

    // EX-stage hazard info (Lane 0)
    input [4:0] rd_ex_0,
    input       reg_wen_ex_0,
    input       is_Load_ex_0,

    // EX-stage hazard info (Lane 1)
    input [4:0] rd_ex_1,
    input       reg_wen_ex_1,

    // Forwarding selects (2-bit: 00=regfile, 01=EX_lane0, 10=EX_lane1)
    output [1:0] fwd_a_0, fwd_b_0,  // Lane 0 forwarding
    output [1:0] fwd_a_1, fwd_b_1,  // Lane 1 forwarding

    // Stall / squash signals
    output load_use_hazard,    // stall both lanes (1-cycle bubble)
    output squash_lane1        // cannot dual-issue, squash lane 1
);

    // ========== INTER-LANE HAZARD DETECTION (ID stage) ==========
    // RAW: Lane 1 reads what Lane 0 writes (same cycle)
    wire raw_intra = reg_wen_id_0 && (rd_id_0 != 5'd0) &&
                     ((rd_id_0 == rs1_id_1) || (rd_id_0 == rs2_id_1));

    // WAW: Both lanes write same register
    wire waw_intra = reg_wen_id_0 && reg_wen_id_1 &&
                     (rd_id_0 == rd_id_1) && (rd_id_0 != 5'd0);

    // Structural: Lane 0 is load -> Lane 1 cannot issue (only one memory port)
    wire struct_hazard = is_Load_id_0;

    // ========== EX→ID FORWARDING ==========
    // EX Lane 0 can forward to ID (not if it's a load - data not ready)
    wire ex0_fwd_ok = reg_wen_ex_0 && (rd_ex_0 != 5'd0) && !is_Load_ex_0;

    // EX Lane 1 can forward to ID (Lane 1 is never a load)
    wire ex1_fwd_ok = reg_wen_ex_1 && (rd_ex_1 != 5'd0);

    // Lane 0 forwarding (EX0 has priority over EX1)
    wire fwd_ex0_to_a0 = ex0_fwd_ok && (rd_ex_0 == rs1_id_0);
    wire fwd_ex1_to_a0 = ex1_fwd_ok && (rd_ex_1 == rs1_id_0) && !fwd_ex0_to_a0;
    wire fwd_ex0_to_b0 = ex0_fwd_ok && (rd_ex_0 == rs2_id_0);
    wire fwd_ex1_to_b0 = ex1_fwd_ok && (rd_ex_1 == rs2_id_0) && !fwd_ex0_to_b0;

    // Lane 1 forwarding
    wire fwd_ex0_to_a1 = ex0_fwd_ok && (rd_ex_0 == rs1_id_1);
    wire fwd_ex1_to_a1 = ex1_fwd_ok && (rd_ex_1 == rs1_id_1) && !fwd_ex0_to_a1;
    wire fwd_ex0_to_b1 = ex0_fwd_ok && (rd_ex_0 == rs2_id_1);
    wire fwd_ex1_to_b1 = ex1_fwd_ok && (rd_ex_1 == rs2_id_1) && !fwd_ex0_to_b1;

    // 2-bit forwarding select: 00=regfile, 01=EX0, 10=EX1
    assign fwd_a_0 = fwd_ex0_to_a0 ? 2'b01 : (fwd_ex1_to_a0 ? 2'b10 : 2'b00);
    assign fwd_b_0 = fwd_ex0_to_b0 ? 2'b01 : (fwd_ex1_to_b0 ? 2'b10 : 2'b00);
    assign fwd_a_1 = fwd_ex0_to_a1 ? 2'b01 : (fwd_ex1_to_a1 ? 2'b10 : 2'b00);
    assign fwd_b_1 = fwd_ex0_to_b1 ? 2'b01 : (fwd_ex1_to_b1 ? 2'b10 : 2'b00);

    // ========== LOAD-USE STALL ==========
    // EX Lane 0 is a load whose rd is needed by ID (either lane)
    assign load_use_hazard = is_Load_ex_0 && reg_wen_ex_0 && (rd_ex_0 != 5'd0) &&
                             ((rd_ex_0 == rs1_id_0) || (rd_ex_0 == rs2_id_0) ||
                              (rd_ex_0 == rs1_id_1) || (rd_ex_0 == rs2_id_1));

    // ========== DUAL-ISSUE DECISION ==========
    // Squash Lane 1 if: not eligible, RAW/WAW hazard, or structural hazard
`ifdef ENABLE_DUAL_ISSUE
    assign squash_lane1 = !lane1_eligible || raw_intra || waw_intra || struct_hazard;
`else
    assign squash_lane1 = 1'b1;
`endif

endmodule
