module branch_comp(
  input [31:0] reg1, reg2,
  input br_un,
  output br_eq,
  output br_lt
);
  assign br_eq = (reg1 == reg2);
  assign br_lt = br_un ? (reg1 < reg2)
                       : ($signed(reg1) < $signed(reg2));
endmodule