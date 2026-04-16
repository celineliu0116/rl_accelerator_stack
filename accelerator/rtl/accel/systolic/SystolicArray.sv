module SystolicArray #(
    parameter int N      = 4,
    parameter int DATA_W = 16,
    parameter int ACC_W  = 32
) (
    input  logic                           clk,
    input  logic                           rst,
    input  logic                           clear_acc,
    input  logic                           en,
    input  logic signed [DATA_W-1:0]       a_left [N],
    input  logic signed [DATA_W-1:0]       b_top  [N],
    output logic signed [ACC_W-1:0]        c_out  [N][N]
);

    logic signed [DATA_W-1:0] a_bus [N][N+1];
    logic signed [DATA_W-1:0] b_bus [N+1][N];

    genvar r, c;
    generate
        for (r = 0; r < N; r++) begin : gen_left
            assign a_bus[r][0] = a_left[r];
        end

        for (c = 0; c < N; c++) begin : gen_top
            assign b_bus[0][c] = b_top[c];
        end

        for (r = 0; r < N; r++) begin : gen_rows
            for (c = 0; c < N; c++) begin : gen_cols
                SystolicPE #(
                    .DATA_W(DATA_W),
                    .ACC_W(ACC_W)
                ) u_pe (
                    .clk      (clk),
                    .rst      (rst),
                    .clear_acc(clear_acc),
                    .en       (en),
                    .a_in     (a_bus[r][c]),
                    .b_in     (b_bus[r][c]),
                    .a_out    (a_bus[r][c+1]),
                    .b_out    (b_bus[r+1][c]),
                    .acc_out  (c_out[r][c])
                );
            end
        end
    endgenerate

endmodule
