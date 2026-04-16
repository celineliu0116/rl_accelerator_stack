`include "const.vh"

// =============================================================================
// MatmulAccelerator 4x4 Block Outer Product Engine
//
// MMIO register map
//   0x80000000  CTRL/STATUS
//       W: [0]=start [4]=err_clr
//       R: [0]=busy [1]=done [2]=full [3]=empty [4]=error
//   0x80000004  W_ADDR
//   0x80000008  X_ADDR
//   0x8000000C  M_DIM
//   0x80000010  N_DIM
//   0x80000014  K_DIM
//   0x80000018 - 0x80000054 RESULTS row-major c[0][0]..c[3][3]
//   0x80000058  X_STRIDE
//   0x8000005C  K_ROW_LEN
// =============================================================================

module MatmulAcceleratorLegacy (
    input  wire         clk,
    input  wire         reset,

    // MMIO
    input  wire [31:0]  mmio_addr,
    input  wire [31:0]  mmio_wdata,
    input  wire [3:0]   mmio_we,
    input  wire         mmio_re,
    output reg  [31:0]  mmio_rdata,

    // DMA read interface
    output reg  [31:0]  dma_addr,
    output reg          dma_re,
    input  wire         dma_req_ready,
    input  wire         dma_resp_valid,
    input  wire [255:0] dma_rdata,

    // Status
    output wire         accel_busy
);

    localparam integer BUF_DEPTH  = 6;
    localparam integer TAGQ_DEPTH = 8;
    localparam integer IDX_BITS   = $clog2(BUF_DEPTH);
    localparam integer TAG_BITS   = $clog2(TAGQ_DEPTH);

    localparam S_IDLE    = 4'd0;
    localparam S_POP     = 4'd1;
    localparam S_LATCH   = 4'd2;
    localparam S_COMPUTE = 4'd3;
    localparam S_DONE    = 4'd4;

    localparam F_IDLE   = 2'd0;
    localparam F_WAIT_A = 2'd1;
    localparam F_WAIT_B = 2'd2;

    // Shadow config
    reg [31:0] shadow_w_addr, shadow_x_addr;
    reg [31:0] shadow_m_dim, shadow_n_dim, shadow_k_dim;
    reg [31:0] shadow_x_stride, shadow_k_row_len;

    // Active command
    reg [31:0] active_ctrl;
    reg [31:0] active_w_addr, active_x_addr;
    reg [31:0] active_m_dim, active_n_dim, active_k_dim;
    reg [31:0] active_x_stride, active_k_row_len;

    // FIFO
    wire [255:0] fifo_din, fifo_dout;
    wire         fifo_full, fifo_empty;
    wire         fifo_push;
    reg          fifo_pop;
    reg          error_sticky;

    assign fifo_din = {
        shadow_k_row_len,
        shadow_k_dim,
        shadow_n_dim,
        shadow_m_dim,
        shadow_x_stride,
        shadow_x_addr,
        shadow_w_addr,
        mmio_wdata
    };

    wire       mmio_wr    = (|mmio_we);
    wire [7:0] reg_offset = mmio_addr[7:0];
    assign fifo_push = mmio_wr && (reg_offset == 8'h00) && mmio_wdata[0];

    FIFO #(.WIDTH(256), .DEPTH(4)) cmd_fifo (
        .clk   (clk),
        .reset (reset),
        .push  (fifo_push && !fifo_full),
        .pop   (fifo_pop),
        .din   (fifo_din),
        .dout  (fifo_dout),
        .full  (fifo_full),
        .empty (fifo_empty),
        .count ()
    );

    // Buffer/state
    reg [127:0] A_buf [0:BUF_DEPTH-1];
    reg [127:0] B_buf [0:BUF_DEPTH-1];
    reg [31:0]  slot_b_addr [0:BUF_DEPTH-1];
    reg         buf_valid [0:BUF_DEPTH-1];
    reg         slot_has_a [0:BUF_DEPTH-1];
    reg         slot_has_b [0:BUF_DEPTH-1];
    reg         slot_pending_a [0:BUF_DEPTH-1];
    reg         slot_pending_b [0:BUF_DEPTH-1];

    reg [IDX_BITS-1:0] use_idx, fill_idx;
    reg [IDX_BITS-1:0] bq_slot [0:BUF_DEPTH-1];
    reg [IDX_BITS-1:0] bq_head, bq_tail;
    reg [IDX_BITS:0]   bq_count;

    // Tag queue for response routing
    reg [IDX_BITS-1:0] tag_slot_q [0:TAGQ_DEPTH-1];
    reg                tag_is_b_q [0:TAGQ_DEPTH-1];
    reg [TAG_BITS-1:0] tag_head, tag_tail;
    reg [TAG_BITS:0]   tag_count;

    // Command/progress
    reg        cmd_active;
    reg [31:0] k_cnt, k_limit;
    reg [31:0] pref_k_cnt;
    reg [31:0] pref_addr_a, pref_addr_b;
    reg [31:0] pref_k_row_cnt;
    reg        startup_prefill_done;

    reg [3:0]  state;
    reg [1:0]  fill_state;

    // Compute accumulators
    reg signed [31:0] c [0:3][0:3];

    // Perf and checks
`ifndef SYNTHESIS
    reg [31:0] accel_busy_cycles;
    reg [31:0] compute_cycles;
    reg [31:0] compute_stall_cycles;
    reg [31:0] fill_cycles;
    reg [31:0] dma_req_count;
    reg [31:0] produced_blocks;
    reg [31:0] consumed_blocks;
    reg [31:0] occupancy0_cycles;
    reg [31:0] occupancy1_cycles;
    reg [31:0] occupancy2_cycles;
    reg [31:0] max_occupancy;
    reg [31:0] cmd_seq;
    reg [31:0] sched_want_cycles;
    reg [31:0] sched_issue_cycles;
    reg [31:0] dma_not_ready_block_cycles;
    reg [31:0] tagq_block_cycles;
    reg [31:0] slot_block_cycles;
`endif

    assign accel_busy = (state != S_IDLE) || !fifo_empty;
    // Registered done flag: cleared immediately on fifo_push (same cycle),
    // set only when state returns to S_IDLE with empty FIFO.
    reg done_r;
    always @(posedge clk) begin
        if (reset)
            done_r <= 1'b1;
        else if (fifo_push)
            done_r <= 1'b0;
        else if ((state == S_IDLE) && fifo_empty)
            done_r <= 1'b1;
        else if (state != S_IDLE)
            done_r <= 1'b0;
    end
    wire done_bit = done_r && !fifo_push;
    wire mode_interleaved = active_ctrl[8];

    function automatic [31:0] ceil_div4(input [31:0] K);
        begin
            ceil_div4 = (K + 32'd3) >> 2;
        end
    endfunction

    function automatic integer rem_k(input integer K, input integer blk);
        begin
            rem_k = K - (blk * 4);
        end
    endfunction

    function automatic [IDX_BITS-1:0] next_idx(input [IDX_BITS-1:0] idx);
        begin
            if (idx == BUF_DEPTH-1)
                next_idx = {IDX_BITS{1'b0}};
            else
                next_idx = idx + {{(IDX_BITS-1){1'b0}}, 1'b1};
        end
    endfunction

    function automatic [TAG_BITS-1:0] next_tag_ptr(input [TAG_BITS-1:0] idx);
        begin
            if (idx == TAGQ_DEPTH-1)
                next_tag_ptr = {TAG_BITS{1'b0}};
            else
                next_tag_ptr = idx + {{(TAG_BITS-1){1'b0}}, 1'b1};
        end
    endfunction

    function automatic signed [7:0] get_blk(input [127:0] blk, input integer step, input integer idx);
        integer lo;
        begin
            lo = step*32 + idx*8;
            get_blk = $signed(blk[lo +: 8]);
        end
    endfunction

    integer i, j, n;
    reg [127:0] curA, curB;
    integer rem;
    reg [IDX_BITS-1:0] rsp_slot;
    reg rsp_is_b;
    reg issue_fire;
    reg issue_is_b;
    reg [IDX_BITS-1:0] issue_slot;
    reg produce_evt;
    reg consume_evt;
    reg [31:0] valid_slots;

    // valid slot count for startup gating and occupancy accounting
    always @(*) begin
        valid_slots = 32'd0;
        for (n = 0; n < BUF_DEPTH; n = n + 1) begin
            if (buf_valid[n]) valid_slots = valid_slots + 32'd1;
        end
    end

    // Config writes
    always @(posedge clk) begin
        if (reset) begin
            shadow_w_addr <= 32'd0;
            shadow_x_addr <= 32'd0;
            shadow_m_dim  <= 32'd0;
            shadow_n_dim  <= 32'd0;
            shadow_k_dim  <= 32'd0;
            shadow_x_stride <= 32'd0;
            shadow_k_row_len <= 32'd0;
            error_sticky <= 1'b0;
        end else if (mmio_wr) begin
            case (reg_offset)
                8'h04: shadow_w_addr <= mmio_wdata;
                8'h08: shadow_x_addr <= mmio_wdata;
                8'h0C: shadow_m_dim  <= mmio_wdata;
                8'h10: shadow_n_dim  <= mmio_wdata;
                8'h14: shadow_k_dim  <= mmio_wdata;
                8'h58: shadow_x_stride <= mmio_wdata;
                8'h5C: shadow_k_row_len <= mmio_wdata;
                8'h00: begin
                    if (mmio_wdata[4]) error_sticky <= 1'b0;
                    if (mmio_wdata[0] && fifo_full) error_sticky <= 1'b1;
                end
            endcase
        end
    end

    always @(posedge clk) begin
        if (reset) begin
            state <= S_IDLE;
            fill_state <= F_IDLE;
            fifo_pop <= 1'b0;
            dma_re <= 1'b0;
            dma_addr <= 32'd0;

            active_ctrl <= 32'd0;
            active_w_addr <= 32'd0;
            active_x_addr <= 32'd0;
            active_m_dim <= 32'd0;
            active_n_dim <= 32'd0;
            active_k_dim <= 32'd0;
            active_x_stride <= 32'd0;
            active_k_row_len <= 32'd0;

            cmd_active <= 1'b0;
            k_cnt <= 32'd0;
            k_limit <= 32'd0;
            pref_k_cnt <= 32'd0;
            pref_addr_a <= 32'd0;
            pref_addr_b <= 32'd0;
            pref_k_row_cnt <= 32'd0;
            startup_prefill_done <= 1'b0;

            use_idx <= {IDX_BITS{1'b0}};
            fill_idx <= {IDX_BITS{1'b0}};
            bq_head <= {IDX_BITS{1'b0}};
            bq_tail <= {IDX_BITS{1'b0}};
            bq_count <= {(IDX_BITS+1){1'b0}};

            tag_head <= {TAG_BITS{1'b0}};
            tag_tail <= {TAG_BITS{1'b0}};
            tag_count <= {(TAG_BITS+1){1'b0}};

            for (n = 0; n < BUF_DEPTH; n = n + 1) begin
                A_buf[n] <= 128'd0;
                B_buf[n] <= 128'd0;
                bq_slot[n] <= {IDX_BITS{1'b0}};
                slot_b_addr[n] <= 32'd0;
                buf_valid[n] <= 1'b0;
                slot_has_a[n] <= 1'b0;
                slot_has_b[n] <= 1'b0;
                slot_pending_a[n] <= 1'b0;
                slot_pending_b[n] <= 1'b0;
            end
            for (n = 0; n < TAGQ_DEPTH; n = n + 1) begin
                tag_slot_q[n] <= {IDX_BITS{1'b0}};
                tag_is_b_q[n] <= 1'b0;
            end

            for (i = 0; i < 4; i = i + 1)
                for (j = 0; j < 4; j = j + 1)
                    c[i][j] <= 32'sd0;

`ifndef SYNTHESIS
            accel_busy_cycles <= 32'd0;
            compute_cycles <= 32'd0;
            compute_stall_cycles <= 32'd0;
            fill_cycles <= 32'd0;
            dma_req_count <= 32'd0;
            produced_blocks <= 32'd0;
            consumed_blocks <= 32'd0;
            occupancy0_cycles <= 32'd0;
            occupancy1_cycles <= 32'd0;
            occupancy2_cycles <= 32'd0;
            max_occupancy <= 32'd0;
            cmd_seq <= 32'd0;
            sched_want_cycles <= 32'd0;
            sched_issue_cycles <= 32'd0;
            dma_not_ready_block_cycles <= 32'd0;
            tagq_block_cycles <= 32'd0;
            slot_block_cycles <= 32'd0;
`endif
        end else begin
            fifo_pop <= 1'b0;
            dma_re <= 1'b0;
            issue_fire = 1'b0;
            issue_is_b = 1'b0;
            issue_slot = {IDX_BITS{1'b0}};
            produce_evt = 1'b0;
            consume_evt = 1'b0;
            fill_state <= F_IDLE;

            // -----------------------------------------------------------------
            // Main command FSM
            // -----------------------------------------------------------------
            case (state)
                S_IDLE: begin
                    if (!fifo_empty) begin
                        fifo_pop <= 1'b1;
                        state <= S_POP;
                    end
                end

                S_POP: begin
                    state <= S_LATCH;
                end

                S_LATCH: begin
                    active_ctrl      <= fifo_dout[31:0];
                    active_w_addr    <= fifo_dout[63:32];
                    active_x_addr    <= fifo_dout[95:64];
                    active_x_stride  <= fifo_dout[127:96];
                    active_m_dim     <= fifo_dout[159:128];
                    active_n_dim     <= fifo_dout[191:160];
                    active_k_dim     <= fifo_dout[223:192];
                    active_k_row_len <= fifo_dout[255:224];

                    cmd_active <= 1'b1;
                    k_cnt <= 32'd0;
                    k_limit <= ceil_div4(fifo_dout[223:192]);
                    pref_k_cnt <= 32'd0;
                    pref_addr_a <= fifo_dout[95:64];
                    pref_addr_b <= fifo_dout[63:32];
                    pref_k_row_cnt <= 32'd0;
                    startup_prefill_done <= 1'b0;

                    use_idx <= {IDX_BITS{1'b0}};
                    fill_idx <= {IDX_BITS{1'b0}};
                    bq_head <= {IDX_BITS{1'b0}};
                    bq_tail <= {IDX_BITS{1'b0}};
                    bq_count <= {(IDX_BITS+1){1'b0}};

                    tag_head <= {TAG_BITS{1'b0}};
                    tag_tail <= {TAG_BITS{1'b0}};
                    tag_count <= {(TAG_BITS+1){1'b0}};

                    for (n = 0; n < BUF_DEPTH; n = n + 1) begin
                        bq_slot[n] <= {IDX_BITS{1'b0}};
                        buf_valid[n] <= 1'b0;
                        slot_has_a[n] <= 1'b0;
                        slot_has_b[n] <= 1'b0;
                        slot_pending_a[n] <= 1'b0;
                        slot_pending_b[n] <= 1'b0;
                    end

                    for (i = 0; i < 4; i = i + 1)
                        for (j = 0; j < 4; j = j + 1)
                            c[i][j] <= 32'sd0;

                    state <= S_COMPUTE;

`ifndef SYNTHESIS
                    cmd_seq <= cmd_seq + 32'd1;
                    accel_busy_cycles <= 32'd0;
                    compute_cycles <= 32'd0;
                    compute_stall_cycles <= 32'd0;
                    fill_cycles <= 32'd0;
                    dma_req_count <= 32'd0;
                    produced_blocks <= 32'd0;
                    consumed_blocks <= 32'd0;
                    occupancy0_cycles <= 32'd0;
                    occupancy1_cycles <= 32'd0;
                    occupancy2_cycles <= 32'd0;
                    max_occupancy <= 32'd0;
                    sched_want_cycles <= 32'd0;
                    sched_issue_cycles <= 32'd0;
                    dma_not_ready_block_cycles <= 32'd0;
                    tagq_block_cycles <= 32'd0;
                    slot_block_cycles <= 32'd0;
`endif
                end

                S_COMPUTE: begin
                    if (k_cnt >= k_limit) begin
                        state <= S_DONE;
                    end else if (!startup_prefill_done) begin
                        if (((k_limit <= 1) && (valid_slots >= 1)) ||
                            ((k_limit > 1) && (valid_slots >= 2))) begin
                            startup_prefill_done <= 1'b1;
                        end
                    end else if (buf_valid[use_idx]) begin
                        curA = A_buf[use_idx];
                        curB = B_buf[use_idx];
                        rem = rem_k(active_k_dim, k_cnt);

                        buf_valid[use_idx] <= 1'b0;
                        slot_has_a[use_idx] <= 1'b0;
                        slot_has_b[use_idx] <= 1'b0;
                        slot_pending_a[use_idx] <= 1'b0;
                        slot_pending_b[use_idx] <= 1'b0;
                        use_idx <= next_idx(use_idx);

                        for (i = 0; i < 4; i = i + 1) begin
                            for (j = 0; j < 4; j = j + 1) begin
                                if ((i < active_m_dim) && (j < active_n_dim)) begin
                                    c[i][j] <= c[i][j]
                                             + ((rem > 0) ? (get_blk(curA,0,i) * get_blk(curB,0,j)) : 0)
                                             + ((rem > 1) ? (get_blk(curA,1,i) * get_blk(curB,1,j)) : 0)
                                             + ((rem > 2) ? (get_blk(curA,2,i) * get_blk(curB,2,j)) : 0)
                                             + ((rem > 3) ? (get_blk(curA,3,i) * get_blk(curB,3,j)) : 0);
                                end
                            end
                        end

                        consume_evt = 1'b1;
                        if (k_cnt + 1 >= k_limit) begin
                            state <= S_DONE;
                        end else begin
                            k_cnt <= k_cnt + 32'd1;
                        end
                    end
                end

                S_DONE: begin
                    cmd_active <= 1'b0;
                    startup_prefill_done <= 1'b0;
                    bq_count <= {(IDX_BITS+1){1'b0}};
                    state <= S_IDLE;
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase

            // -----------------------------------------------------------------
            // Pipelined prefetch scheduler
            // -----------------------------------------------------------------
            if (cmd_active && (state != S_LATCH) && (state != S_DONE) && dma_req_ready && (tag_count < TAGQ_DEPTH)) begin
                if (mode_interleaved) begin
                    if ((pref_k_cnt < k_limit)
                     && !buf_valid[fill_idx]
                     && !slot_has_a[fill_idx] && !slot_has_b[fill_idx]
                     && !slot_pending_a[fill_idx] && !slot_pending_b[fill_idx]) begin
                        reg [31:0] next_pref_addr_a;
                        reg [31:0] next_pref_k_row_cnt;
                        reg [IDX_BITS-1:0] cur_fill_idx;

                        cur_fill_idx = fill_idx;
                        dma_addr <= pref_addr_a;
                        dma_re <= 1'b1;
                        issue_fire = 1'b1;
                        issue_is_b = 1'b0;
                        issue_slot = cur_fill_idx;
                        slot_pending_a[cur_fill_idx] <= 1'b1;
                        slot_pending_b[cur_fill_idx] <= 1'b1;

                        pref_k_cnt <= pref_k_cnt + 32'd1;
                        if ((active_k_row_len > 0) && (pref_k_row_cnt + 1 >= active_k_row_len)) begin
                            next_pref_k_row_cnt = 32'd0;
                            next_pref_addr_a = pref_addr_a + active_x_stride;
                        end else begin
                            next_pref_k_row_cnt = pref_k_row_cnt + 32'd1;
                            next_pref_addr_a = pref_addr_a + 32'd32;
                        end
                        pref_addr_a <= next_pref_addr_a;
                        pref_k_row_cnt <= next_pref_k_row_cnt;
                        fill_idx <= next_idx(cur_fill_idx);
                        fill_state <= F_WAIT_A;
                    end
                end else begin
                    // Prefer issuing B for previously allocated slots, then allocate a new A.
                    if ((bq_count != 0)
                     && !slot_has_b[bq_slot[bq_head]]
                     && !slot_pending_b[bq_slot[bq_head]]) begin
                        dma_addr <= slot_b_addr[bq_slot[bq_head]];
                        dma_re <= 1'b1;
                        issue_fire = 1'b1;
                        issue_is_b = 1'b1;
                        issue_slot = bq_slot[bq_head];
                        slot_pending_b[bq_slot[bq_head]] <= 1'b1;
                        bq_head <= next_idx(bq_head);
                        bq_count <= bq_count - 1'b1;
                        fill_state <= F_WAIT_B;
                    end else if ((pref_k_cnt < k_limit)
                              && (bq_count < BUF_DEPTH)
                              && !buf_valid[fill_idx]
                              && !slot_has_a[fill_idx] && !slot_has_b[fill_idx]
                              && !slot_pending_a[fill_idx] && !slot_pending_b[fill_idx]) begin
                        reg [31:0] next_pref_addr_a;
                        reg [31:0] next_pref_addr_b;
                        reg [31:0] next_pref_k_row_cnt;
                        reg [IDX_BITS-1:0] cur_fill_idx;

                        cur_fill_idx = fill_idx;

                        dma_addr <= pref_addr_a;
                        dma_re <= 1'b1;
                        issue_fire = 1'b1;
                        issue_is_b = 1'b0;
                        issue_slot = cur_fill_idx;
                        slot_pending_a[cur_fill_idx] <= 1'b1;
                        slot_b_addr[cur_fill_idx] <= pref_addr_b;
                        bq_slot[bq_tail] <= cur_fill_idx;
                        bq_tail <= next_idx(bq_tail);
                        bq_count <= bq_count + 1'b1;

                        pref_k_cnt <= pref_k_cnt + 32'd1;
                        next_pref_addr_b = pref_addr_b + 32'd16;
                        if ((active_k_row_len > 0) && (pref_k_row_cnt + 1 >= active_k_row_len)) begin
                            next_pref_k_row_cnt = 32'd0;
                            next_pref_addr_a = pref_addr_a + active_x_stride;
                        end else begin
                            next_pref_k_row_cnt = pref_k_row_cnt + 32'd1;
                            next_pref_addr_a = pref_addr_a + 32'd16;
                        end
                        pref_addr_a <= next_pref_addr_a;
                        pref_addr_b <= next_pref_addr_b;
                        pref_k_row_cnt <= next_pref_k_row_cnt;
                        fill_idx <= next_idx(cur_fill_idx);
                        fill_state <= F_WAIT_A;
                    end
                end
            end

            // -----------------------------------------------------------------
            // Response routing via tag queue
            // -----------------------------------------------------------------
            if (dma_resp_valid) begin
                if (tag_count == 0) begin
`ifndef SYNTHESIS
                    $fatal(1, "Matmul invariant failed: dma_resp_valid with empty tag queue");
`endif
                end else begin
                    rsp_slot = tag_slot_q[tag_head];
                    rsp_is_b = tag_is_b_q[tag_head];
                    tag_head <= next_tag_ptr(tag_head);

                    if (mode_interleaved) begin
                        A_buf[rsp_slot] <= dma_rdata[127:0];
                        B_buf[rsp_slot] <= dma_rdata[255:128];
                        slot_pending_a[rsp_slot] <= 1'b0;
                        slot_pending_b[rsp_slot] <= 1'b0;
                        slot_has_a[rsp_slot] <= 1'b1;
                        slot_has_b[rsp_slot] <= 1'b1;
                        buf_valid[rsp_slot] <= 1'b1;
                        produce_evt = 1'b1;
                    end else if (rsp_is_b) begin
                        B_buf[rsp_slot] <= dma_rdata[127:0];
                        slot_pending_b[rsp_slot] <= 1'b0;
                        slot_has_b[rsp_slot] <= 1'b1;
                        if (slot_has_a[rsp_slot]) begin
                            buf_valid[rsp_slot] <= 1'b1;
                            produce_evt = 1'b1;
                        end
                    end else begin
                        A_buf[rsp_slot] <= dma_rdata[127:0];
                        slot_pending_a[rsp_slot] <= 1'b0;
                        slot_has_a[rsp_slot] <= 1'b1;
                        if (slot_has_b[rsp_slot]) begin
                            buf_valid[rsp_slot] <= 1'b1;
                            produce_evt = 1'b1;
                        end
                    end
                end
            end

            // Tag queue push for issued request
            if (issue_fire) begin
                if (tag_count >= TAGQ_DEPTH) begin
`ifndef SYNTHESIS
                    $fatal(1, "Matmul invariant failed: tag queue overflow");
`endif
                end else begin
                    tag_slot_q[tag_tail] <= issue_slot;
                    tag_is_b_q[tag_tail] <= issue_is_b;
                    tag_tail <= next_tag_ptr(tag_tail);
                end
            end

            // tag_count update (handles simultaneous push/pop)
            case ({issue_fire, dma_resp_valid})
                2'b10: tag_count <= tag_count + 1'b1;
                2'b01: tag_count <= tag_count - 1'b1;
                default: tag_count <= tag_count;
            endcase

`ifndef SYNTHESIS
            if (cmd_active || !fifo_empty || (state != S_IDLE)) begin
                accel_busy_cycles <= accel_busy_cycles + 32'd1;
            end
            if (consume_evt) begin
                compute_cycles <= compute_cycles + 32'd1;
                consumed_blocks <= consumed_blocks + 32'd1;
            end else if ((state == S_COMPUTE) && startup_prefill_done && (k_cnt < k_limit) && !buf_valid[use_idx]) begin
                compute_stall_cycles <= compute_stall_cycles + 32'd1;
            end
            if (issue_fire || (tag_count != 0)) begin
                fill_cycles <= fill_cycles + 32'd1;
            end
            if (issue_fire) begin
                dma_req_count <= dma_req_count + 32'd1;
            end
            if (produce_evt) begin
                produced_blocks <= produced_blocks + 32'd1;
            end

            if (cmd_active && (state != S_LATCH) && (state != S_DONE) &&
                ((pref_k_cnt < k_limit) || (bq_count != 0))) begin
                sched_want_cycles <= sched_want_cycles + 32'd1;
                if (issue_fire) begin
                    sched_issue_cycles <= sched_issue_cycles + 32'd1;
                end else if (!dma_req_ready) begin
                    dma_not_ready_block_cycles <= dma_not_ready_block_cycles + 32'd1;
                end else if (tag_count >= TAGQ_DEPTH) begin
                    tagq_block_cycles <= tagq_block_cycles + 32'd1;
                end else begin
                    slot_block_cycles <= slot_block_cycles + 32'd1;
                end
            end

            if (cmd_active) begin
                if (valid_slots == 0) begin
                    occupancy0_cycles <= occupancy0_cycles + 32'd1;
                end else if (valid_slots == 1) begin
                    occupancy1_cycles <= occupancy1_cycles + 32'd1;
                end else begin
                    occupancy2_cycles <= occupancy2_cycles + 32'd1;
                end
                if (valid_slots > max_occupancy) begin
                    max_occupancy <= valid_slots;
                end
            end

            if (state == S_DONE) begin
                if (produced_blocks != consumed_blocks) begin
                    $fatal(1, "Matmul invariant failed: produced/consumed mismatch cmd=%0d k_limit=%0d produced=%0d consumed=%0d",
                           cmd_seq, k_limit, produced_blocks, consumed_blocks);
                end
                if (consumed_blocks != k_limit) begin
                    $fatal(1, "Matmul invariant failed: consumed != k_limit cmd=%0d k_limit=%0d consumed=%0d",
                           cmd_seq, k_limit, consumed_blocks);
                end
                $display("ACCEL_PERF cmd=%0d k_limit=%0d busy=%0d compute=%0d stall=%0d fill=%0d dma_req=%0d produced=%0d consumed=%0d occ0=%0d occ1=%0d occ2=%0d occ_max=%0d sched_want=%0d sched_issue=%0d blk_dma=%0d blk_tagq=%0d blk_slot=%0d",
                         cmd_seq, k_limit, accel_busy_cycles, compute_cycles, compute_stall_cycles,
                         fill_cycles, dma_req_count, produced_blocks, consumed_blocks,
                         occupancy0_cycles, occupancy1_cycles, occupancy2_cycles, max_occupancy,
                         sched_want_cycles, sched_issue_cycles, dma_not_ready_block_cycles,
                         tagq_block_cycles, slot_block_cycles);
            end
`endif
        end
    end

    // Registered MMIO read address
    reg [7:0] reg_offset_d;
    always @(posedge clk) begin
        if (reset) reg_offset_d <= 8'd0;
        else if (mmio_re) reg_offset_d <= reg_offset;
    end

    // STATUS bits [0]=busy [1]=done [2]=full [3]=empty [4]=error
    always @(*) begin
        case (reg_offset_d)
            8'h00: mmio_rdata = {27'd0, error_sticky, fifo_empty, fifo_full, done_bit, accel_busy};
            8'h18: mmio_rdata = c[0][0];
            8'h1C: mmio_rdata = c[0][1];
            8'h20: mmio_rdata = c[0][2];
            8'h24: mmio_rdata = c[0][3];
            8'h28: mmio_rdata = c[1][0];
            8'h2C: mmio_rdata = c[1][1];
            8'h30: mmio_rdata = c[1][2];
            8'h34: mmio_rdata = c[1][3];
            8'h38: mmio_rdata = c[2][0];
            8'h3C: mmio_rdata = c[2][1];
            8'h40: mmio_rdata = c[2][2];
            8'h44: mmio_rdata = c[2][3];
            8'h48: mmio_rdata = c[3][0];
            8'h4C: mmio_rdata = c[3][1];
            8'h50: mmio_rdata = c[3][2];
            8'h54: mmio_rdata = c[3][3];
            default: mmio_rdata = 32'd0;
        endcase
    end

endmodule
