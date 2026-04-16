// Module: ALU.v
// Desc:   32-bit ALU with parallel functional units + 4:1 final mux
//         Replaces monolithic 11-case mux to shorten critical path.
//
// Groups:
//   00 - Add/Sub   (ALU_ADD, ALU_SUB)
//   01 - Logic     (ALU_AND, ALU_OR, ALU_XOR, ALU_COPY_B)
//   10 - Compare   (ALU_SLT, ALU_SLTU)
//   11 - Shift     (ALU_SLL, ALU_SRA, ALU_SRL)

`include "Opcode.vh"
`include "ALUop.vh"

module ALU(
    input [31:0] A,B,
    input [3:0] ALUop,
    output reg [31:0] Out
);

// ---- Category select (derived from ALUop) ----
wire [1:0] alu_cat;
assign alu_cat = (ALUop == `ALU_ADD  || ALUop == `ALU_SUB)  ? 2'b00 :
                 (ALUop == `ALU_AND  || ALUop == `ALU_OR  ||
                  ALUop == `ALU_XOR  || ALUop == `ALU_COPY_B) ? 2'b01 :
                 (ALUop == `ALU_SLT  || ALUop == `ALU_SLTU) ? 2'b10 :
                                                               2'b11;

// ---- Parallel functional units ----

// Group 0: Add / Sub
wire [31:0] addsub_out = ALUop[0] ? (A - B) : (A + B);

// Group 1: Logic
reg [31:0] logic_out;
always @(*) begin
    case (ALUop)
        `ALU_AND:    logic_out = A & B;
        `ALU_OR:     logic_out = A | B;
        `ALU_XOR:    logic_out = A ^ B;
        `ALU_COPY_B: logic_out = B;
        default:     logic_out = 32'b0;
    endcase
end

// Group 2: Compare
wire [31:0] cmp_out = (ALUop == `ALU_SLTU)
                        ? {31'b0, (A < B)}
                        : {31'b0, ($signed(A) < $signed(B))};

// Group 3: Shift
reg [31:0] shift_out;
always @(*) begin
    case (ALUop)
        `ALU_SLL: shift_out = A << B[4:0];
        `ALU_SRA: shift_out = $signed(A) >>> B[4:0];
        `ALU_SRL: shift_out = A >> B[4:0];
        default:  shift_out = 32'b0;
    endcase
end

// ---- Final 4:1 mux ----
always @(*) begin
    case (alu_cat)
        2'b00:   Out = addsub_out;
        2'b01:   Out = logic_out;
        2'b10:   Out = cmp_out;
        2'b11:   Out = shift_out;
        default: Out = 32'b0;
    endcase
end

endmodule