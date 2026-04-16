module SystolicPE #(
    parameter int DATA_W = 16,
    parameter int ACC_W  = 32
) (
    input  logic                         clk,
    input  logic                         rst,
    input  logic                         clear_acc,
    input  logic                         en,
    input  logic signed [DATA_W-1:0]     a_in,
    input  logic signed [DATA_W-1:0]     b_in,
    output logic signed [DATA_W-1:0]     a_out,
    output logic signed [DATA_W-1:0]     b_out,
    output logic signed [ACC_W-1:0]      acc_out
);

    logic signed [ACC_W-1:0] acc_r;

    always_ff @(posedge clk) begin
        if (rst) begin
            a_out <= '0;
            b_out <= '0;
            acc_r <= '0;
        end else begin
            a_out <= a_in;
            b_out <= b_in;
            if (clear_acc) begin
                acc_r <= (a_in * b_in);
            end else if (en) begin
                acc_r <= acc_r + (a_in * b_in);
            end
        end
    end

    assign acc_out = acc_r;

endmodule
