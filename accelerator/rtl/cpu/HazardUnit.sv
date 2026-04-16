// HazardUnit.sv — Centralized hazard detection and forwarding control
// for the 4-stage IF → ID → EX → WB pipeline.
//
// Forwarding priority: EX > WB (RegFile bypass handles WB internally)
// Load-use stall: 1-cycle bubble when EX is a load and ID reads its rd

module HazardUnit(
    // ID-stage instruction fields
    input [4:0] rs1_id,
    input [4:0] rs2_id,

    // EX-stage hazard info
    input [4:0] rd_ex,
    input       reg_wen_ex,   // does EX-stage instruction write a register?
    input       is_Load_ex,   // is EX-stage instruction a load?

    // WB-stage hazard info
    input [4:0] rd_wb,
    input       reg_wen_wb,

    // Forwarding selects (to ID-stage operand muxes)
    //   0 = use register file, 1 = forward from EX (alu_out)
    output fwd_a,
    output fwd_b,

    // Stall / bubble
    output load_use_hazard    // stall IF+ID, bubble EX
);

    // EX→ID forwarding: EX has a non-load result ready
    wire ex_fwd_ok = reg_wen_ex && (rd_ex != 5'd0) && !is_Load_ex;

    assign fwd_a = ex_fwd_ok && (rd_ex == rs1_id);
    assign fwd_b = ex_fwd_ok && (rd_ex == rs2_id);

    // Load-use hazard: EX is a load whose rd is needed by ID
    assign load_use_hazard = is_Load_ex && reg_wen_ex && (rd_ex != 5'd0) &&
                             ((rd_ex == rs1_id) || (rd_ex == rs2_id));

endmodule
