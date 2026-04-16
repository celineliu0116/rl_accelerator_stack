
`include "Opcode.vh"

module Multiplier(
    input        m_valid,
    input [31:0] A,
    input [31:0] B,
    input [2:0] funct3,
    output reg [31:0] result
);

    wire [63:0] mul_res_u = $unsigned(A) * $unsigned(B);
    wire [63:0] mul_res_s = $signed(A) * $signed(B);
    wire [63:0] mul_res_su = $signed(A) * $unsigned(B);

    always @(*) begin
        case (funct3)
            3'b000: result = mul_res_s[31:0];   // MUL
            3'b001: result = mul_res_s[63:32];  // MULH
            3'b010: result = mul_res_su[63:32]; // MULHSU
            3'b011: result = mul_res_u[63:32];  // MULHU
            // DIV/REM are intentionally unsupported in this core configuration.
            3'b100: result = 32'd0; // DIV   (stubbed)
            3'b101: result = 32'd0; // DIVU  (stubbed)
            3'b110: result = 32'd0; // REM   (stubbed)
            3'b111: result = 32'd0; // REMU  (stubbed)
            default: result = 32'd0;
        endcase
    end

`ifndef SYNTHESIS
    always @(*) begin
        if (m_valid && funct3[2]) begin
            $error("Unsupported M-extension op (DIV/REM) funct3=%0b", funct3);
        end
    end
`endif

endmodule
