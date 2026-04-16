/*
 * FIFO.sv
 *
 * Robust synchronous FIFO (1-clock) with parameterized width/depth.
 * - Safe semantics for read-during-write.
 * - Registered output on pop (stable data for consumers).
 * - Supports simultaneous push and pop.
 *
 * Notes:
 * - dout is only guaranteed valid on cycles where `pop && !empty` is true.
 *   (dout may be read on any cycle; validity is guaranteed only on pop.)
 * - count/full/empty reflect the FIFO state *after* the clock edge updates.
 */

module FIFO #(
    parameter int WIDTH = 256,
    parameter int DEPTH = 4
) (
    input  wire                  clk,
    input  wire                  reset,

    input  wire                  push,
    input  wire                  pop,
    input  wire [WIDTH-1:0]      din,

    output reg  [WIDTH-1:0]      dout,   // registered data-out (valid on pop)
    output wire                  full,
    output wire                  empty,
    output reg  [$clog2(DEPTH):0] count
);

    // -------------------------------------------------------------------------
    // Storage and pointers
    // -------------------------------------------------------------------------
    reg [WIDTH-1:0] mem [0:DEPTH-1];
    reg [$clog2(DEPTH)-1:0] w_ptr, r_ptr;

    assign full  = (count == DEPTH);
    assign empty = (count == 0);

    // -------------------------------------------------------------------------
    // Helpers for wrap-around increment
    // -------------------------------------------------------------------------
    function automatic [$clog2(DEPTH)-1:0] inc_ptr(
        input [$clog2(DEPTH)-1:0] ptr
    );
        if (ptr == DEPTH-1) inc_ptr = '0;
        else                inc_ptr = ptr + 1'b1;
    endfunction

    // -------------------------------------------------------------------------
    // Main sequential logic
    // -------------------------------------------------------------------------
    always @(posedge clk) begin
        if (reset) begin
            w_ptr <= '0;
            r_ptr <= '0;
            count <= '0;
            dout  <= '0;
        end else begin
            // Compute effective enables (don’t overflow/underflow)
            logic do_push, do_pop;
            do_push = push && !full;
            do_pop  = pop  && !empty;

            // -----------------------------
            // Read data (registered) on pop
            // -----------------------------
            // Capture the element being popped (at current r_ptr).
            // This avoids ambiguous combinational read-during-write behavior.
            if (do_pop) begin
                dout <= mem[r_ptr];
            end

            // -----------------------------
            // Write data on push
            // -----------------------------
            if (do_push) begin
                mem[w_ptr] <= din;
            end

            // -----------------------------
            // Advance pointers
            // -----------------------------
            if (do_push) begin
                w_ptr <= inc_ptr(w_ptr);
            end

            if (do_pop) begin
                r_ptr <= inc_ptr(r_ptr);
            end

            // -----------------------------
            // Update count (simultaneous push/pop leaves count unchanged)
            // -----------------------------
            unique case ({do_push, do_pop})
                2'b10: count <= count + 1'b1;
                2'b01: count <= count - 1'b1;
                default: count <= count; // 00 or 11
            endcase
        end
    end

endmodule
