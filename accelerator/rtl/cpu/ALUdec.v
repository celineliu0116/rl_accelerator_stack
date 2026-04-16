// Module: ALUdecoder
// Desc:   Sets the ALU operation
// Inputs: opcode: the top 6 bits of the instruction
//         funct: the funct, in the case of r-type instructions
//         add_rshift_type: selects whether an ADD vs SUB, or an SRA vs SRL
// Outputs: ALUop: Selects the ALU's operation
//

`include "Opcode.vh"
`include "ALUop.vh"

module ALUdec(
  input [2:0]       funct,
  input             add_rshift_type,
  input             alu_is_copy,
  input             is_R_x,
  input             is_Ialu_x,
  output reg [3:0]  ALUop
);

// Implement your ALU decoder here, then delete this comment
always @(*) begin
    if(is_R_x) begin
      case (funct)
        3'b000: ALUop = add_rshift_type ? `ALU_SUB : `ALU_ADD;
        3'b001: ALUop = `ALU_SLL;
        3'b010: ALUop = `ALU_SLT;
        3'b011: ALUop = `ALU_SLTU;
        3'b100: ALUop = `ALU_XOR;
        3'b101: ALUop = add_rshift_type ? `ALU_SRA : `ALU_SRL;
        3'b110: ALUop = `ALU_OR;
        3'b111: ALUop = `ALU_AND;
      endcase
    end
    else if(is_Ialu_x) begin // I-type arithmetic
      case (funct)
        3'b000: ALUop = `ALU_ADD;  // ADDI
        3'b001: ALUop = `ALU_SLL;  // SLLI
        3'b010: ALUop = `ALU_SLT;  // SLTI
        3'b011: ALUop = `ALU_SLTU; // SLTIU
        3'b100: ALUop = `ALU_XOR;  // XORI
        3'b101: ALUop = add_rshift_type ? `ALU_SRA : `ALU_SRL; // SRAI/SRLI
        3'b110: ALUop = `ALU_OR;   // ORI
        3'b111: ALUop = `ALU_AND;  // ANDI
      endcase
    end
    else if(alu_is_copy) ALUop = `ALU_COPY_B;
    else ALUop = `ALU_ADD;
end


endmodule
