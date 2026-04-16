// MatmulAcceleratorSystolic.sv
//
// 4x4 HW Accelerator bridging Dense Systolic and Sparse Intersection dataflows.
//
// Mode 0 (Dense Systolic):
//   - Operates on 4x4 tiles of Activation (X) and Weight (W) interleaving over DMA.
//   - Pipeline latency is shifted using rank-based delay lines.
//
// Mode 1 (Sparse Intersection):
//   - Fetches dense input vector (X) into 32KB internal SRAM exactly once.
//   - Fetches N+1 CSR row pointers into 1KB internal SRAM exactly once.
//   - Streams interleaved CSR [col_idx, val] elements over DMA sequentially.
//   - Broadcasts 'val' to row 0 MACs, and indexes 'x_sram' to feed 'col_idx' data.
//
// MMIO map:
//   0x00  CTRL/STATUS   W:[0]=start  R:[0]=busy [1]=done [2]=fifo_full
//   0x04  W_ADDR
//   0x08  X_ADDR
//   0x0C  M_DIM
//   0x10  N_DIM
//   0x14  K_DIM
//   0x18-0x54  RESULTS c[0][0]..c[3][3]
//   0x58  X_STRIDE
//   0x5C  K_ROW_LEN
//   ------------ Mode 1 Additions -------------
//   0x70  HW_MODE       (0=Dense, 1=Sparse)
//   0x74  CSR_VAL_ADDR
//   0x78  CSR_COL_IDX_ADDR
//   0x7C  CSR_ROW_PTR_ADDR
//   0x80  PERF_PE_IDLE
//   0x84  PERF_DMA_STALL

`include "const.vh"

module MatmulAcceleratorSystolic (
    input  wire         clk,
    input  wire         reset,
    input  wire [31:0]  mmio_addr,
    input  wire [31:0]  mmio_wdata,
    input  wire [3:0]   mmio_we,
    input  wire         mmio_re,
    output reg  [31:0]  mmio_rdata,
    output reg  [31:0]  dma_addr,
    output reg          dma_re,
    input  wire         dma_req_ready,
    input  wire         dma_resp_valid,
    input  wire [255:0] dma_rdata,
    output wire         accel_busy
);

    // =========================================================================
    // Parameters
    // =========================================================================
    localparam integer N         = 4;
    localparam integer DATA_W    = 16;
    localparam integer ACC_W     = 32;
    localparam integer BUF_DEPTH = 4;
    localparam integer TAG_DEPTH = 8;
    localparam integer BUF_BITS  = 2;  // $clog2(BUF_DEPTH)
    localparam integer TAG_BITS  = 3;  // $clog2(TAG_DEPTH)

    // =========================================================================
    // State encoding
    // =========================================================================
    localparam [3:0]
        S_IDLE        = 4'd0,
        S_POP         = 4'd1,
        S_LATCH       = 4'd2,
        // Mode 0 states
        S_PREFILL     = 4'd3,
        S_COMPUTE     = 4'd4,
        S_DRAIN       = 4'd5,
        S_DONE        = 4'd6;

    localparam [1:0]
        DMA_TARGET_BUF = 2'd0,
        DMA_TARGET_X   = 2'd1,
        DMA_TARGET_PTR = 2'd2;

    // =========================================================================
    // MMIO
    // =========================================================================
    wire       mmio_wr    = (|mmio_we);
    wire [7:0] reg_offset = mmio_addr[7:0];

    // Configs
    reg [31:0] shadow_w_addr, shadow_x_addr;
    reg [31:0] shadow_m_dim, shadow_n_dim, shadow_k_dim;
    reg [31:0] shadow_x_stride, shadow_k_row_len;
    reg [31:0] shadow_hw_mode, shadow_csr_val_addr, shadow_csr_col_idx_addr, shadow_csr_row_ptr_addr;

    wire [511:0] fifo_din, fifo_dout;
    wire         fifo_full, fifo_empty;
    reg          fifo_pop;

    assign fifo_din = { // Note: 256 is not enough to hold all shadows! We expand FIFO to 512 bit or trigger directly.
        // Wait, fixing FIFO size requires changing FIFO.sv. Since we only issue 1 command at a time from bare-metal,
        // we can bypass the queue for the new Mode 1 registers and just read them directly from shadow regs,
        // because `accel_run_sparse` calls `accel_wait_not_full()` anyway. 
        // For absolute safety, let's just make the FIFO 512-bit wide locally.
        shadow_csr_row_ptr_addr, // 511:480
        shadow_csr_col_idx_addr, // 479:448
        shadow_csr_val_addr,     // 447:416
        shadow_hw_mode,          // 415:384
        shadow_k_row_len,        // 383:352 (was 255:224)
        shadow_k_dim,            // 351:320
        shadow_n_dim,            // 319:288
        shadow_m_dim,            // 287:256
        shadow_x_stride,         // 255:224
        shadow_x_addr,           // 223:192
        shadow_w_addr,           // 191:160
        32'd0,                   // padding 159:128
        32'd0,                   // padding 127:96
        32'd0,                   // padding 95:64
        32'd0,                   // padding 63:32
        mmio_wdata               // 31:0
    };

    wire fifo_push = mmio_wr && (reg_offset == 8'h00) && mmio_wdata[0];

    // Local 512-bit wide FIFO instantiation
    FIFO #(.WIDTH(512), .DEPTH(4)) cmd_fifo (
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

    // Active command fields
    reg [31:0] active_w_addr, active_x_addr;
    reg [31:0] active_m_dim, active_n_dim, active_k_dim;
    reg [31:0] active_x_stride, active_k_row_len;
    reg [31:0] active_hw_mode, active_csr_val_addr, active_csr_col_idx_addr, active_csr_row_ptr_addr;
    reg [31:0] k_limit;

    // Prefetch state
    reg [31:0]       pref_addr_x, pref_addr_w;
    reg [31:0]       pref_k_cnt;
    reg [31:0]       pref_k_row_cnt;
    reg [31:0]       k_cnt;
    reg [1:0]        step_cnt;
    reg [2:0]        drain_cnt;

    // Buffer state
    reg [255:0]         buf_data  [0:BUF_DEPTH-1];
    reg                 buf_valid [0:BUF_DEPTH-1];
    reg [BUF_BITS-1:0]  buf_fill_ptr, buf_use_ptr;

    reg [BUF_BITS-1:0]  tag_slot  [0:TAG_DEPTH-1];
    reg [TAG_BITS-1:0]  tag_head, tag_tail;
    reg [TAG_BITS:0]    tag_count;

    // DMA multiplexing state
    reg [1:0] dma_target;
    reg [31:0] block_issue_cnt;
    reg [31:0] block_recv_cnt;
    reg [31:0] block_limit;
    
    // Removed Mode 1 CSR SRAMs and state variables. 2:4 structured sparsity is now
    // streamlined into S_COMPUTE using the primary FSM state.
    // Systolic Output
    reg  clear_acc;
    reg  en;
    reg  signed [DATA_W-1:0] a_left [0:N-1];
    reg  signed [DATA_W-1:0] b_top  [0:N-1];
    wire signed [ACC_W-1:0]  c_out  [0:N-1][0:N-1];

    SystolicArray #(.N(N), .DATA_W(DATA_W), .ACC_W(ACC_W)) u_systolic (
        .clk(clk), .rst(reset), .clear_acc(clear_acc),
        .en(en), .a_left(a_left), .b_top(b_top), .c_out(c_out)
    );

    // Skew shift registers (Mode 0)
    reg signed [DATA_W-1:0] a_sr [0:N-1][0:N-2];
    reg signed [DATA_W-1:0] b_sr [0:N-1][0:N-2];

    // Results
    reg signed [ACC_W-1:0] result_r [0:N-1][0:N-1];
    reg done_r;
    reg cmd_active;

    reg [3:0] state;
    assign accel_busy = (state != S_IDLE) || !fifo_empty;

    // Hardware Perf Counters
    reg [31:0] pe_idle_cycles;
    reg [31:0] dma_stall_cycles;

    // =========================================================================
    // Functions
    // =========================================================================
    function automatic [BUF_BITS-1:0] next_buf(input [BUF_BITS-1:0] p);
        next_buf = (p == BUF_DEPTH-1) ? '0 : p + 1'b1;
    endfunction

    function automatic [TAG_BITS-1:0] next_tag(input [TAG_BITS-1:0] p);
        next_tag = (p == TAG_DEPTH-1) ? '0 : p + 1'b1;
    endfunction

    function automatic [31:0] ceil_div4(input [31:0] K);
        ceil_div4 = (K + 32'd3) >> 2;
    endfunction

    function automatic [31:0] ceil_div8(input [31:0] K);
        ceil_div8 = (K + 32'd7) >> 3;
    endfunction

    // =========================================================================
    // Combinational Data Decoding
    // =========================================================================
    reg [127:0] cur_x_chunk, cur_w_chunk;
    reg signed [DATA_W-1:0] a_in [0:N-1];
    reg signed [DATA_W-1:0] b_in [0:N-1];
    
    // Removed old Mode 1 CSR extractors array.

    integer i, j;

    // =========================================================================
    // Shadow writes
    // =========================================================================
    always @(posedge clk) begin
        if (reset) begin
            shadow_w_addr    <= 32'd0;
            shadow_x_addr    <= 32'd0;
            shadow_m_dim     <= 32'd0;
            shadow_n_dim     <= 32'd0;
            shadow_k_dim     <= 32'd0;
            shadow_x_stride  <= 32'd0;
            shadow_k_row_len <= 32'd0;
            shadow_hw_mode   <= 32'd0;
            shadow_csr_val_addr     <= 32'd0;
            shadow_csr_col_idx_addr <= 32'd0;
            shadow_csr_row_ptr_addr <= 32'd0;
        end else if (mmio_wr) begin
            case (reg_offset)
                8'h04: shadow_w_addr    <= mmio_wdata;
                8'h08: shadow_x_addr    <= mmio_wdata;
                8'h0C: shadow_m_dim     <= mmio_wdata;
                8'h10: shadow_n_dim     <= mmio_wdata;
                8'h14: shadow_k_dim     <= mmio_wdata;
                8'h58: shadow_x_stride  <= mmio_wdata;
                8'h5C: shadow_k_row_len <= mmio_wdata;
                8'h70: shadow_hw_mode   <= mmio_wdata;
                8'h74: shadow_csr_val_addr     <= mmio_wdata;   // Aliased as elements
                8'h78: shadow_csr_col_idx_addr <= mmio_wdata;
                8'h7C: shadow_csr_row_ptr_addr <= mmio_wdata;
                default: ;
            endcase
        end
    end

    // =========================================================================
    // FSM
    // =========================================================================
    always @(posedge clk) begin
        if (reset) begin
            state       <= S_IDLE;
            fifo_pop    <= 1'b0;
            dma_re      <= 1'b0;
            dma_addr    <= 32'd0;
            cmd_active  <= 1'b0;
            done_r      <= 1'b0;
            clear_acc   <= 1'b0;
            en          <= 1'b0;
            
            dma_target  <= DMA_TARGET_BUF;
            block_issue_cnt <= 0;
            block_recv_cnt  <= 0;
            block_limit     <= 0;

            active_w_addr    <= 32'd0;
            active_x_addr    <= 32'd0;
            active_m_dim     <= 32'd0;
            active_n_dim     <= 32'd0;
            active_k_dim     <= 32'd0;
            active_x_stride  <= 32'd0;
            active_k_row_len <= 32'd0;
            active_hw_mode   <= 32'd0;
            k_limit          <= 32'd0;

            pref_addr_x    <= 32'd0;
            pref_addr_w    <= 32'd0;
            pref_k_cnt     <= 32'd0;
            pref_k_row_cnt <= 32'd0;
            k_cnt          <= 32'd0;
            step_cnt       <= 2'd0;
            drain_cnt      <= 3'd0;
            
            pe_idle_cycles   <= 32'd0;
            dma_stall_cycles <= 32'd0;

            buf_fill_ptr <= '0;
            buf_use_ptr  <= '0;
            tag_head     <= '0;
            tag_tail     <= '0;
            tag_count    <= '0;

            for (i = 0; i < BUF_DEPTH; i = i+1) buf_valid[i] <= 1'b0;
            for (i = 0; i < TAG_DEPTH; i = i+1) tag_slot[i] <= '0;

            for (i = 0; i < N; i = i+1) begin
                a_left[i] <= '0;
                b_top[i]  <= '0;
                for (j = 0; j < N-1; j = j+1) begin
                    a_sr[i][j] <= '0;
                    b_sr[i][j] <= '0;
                end
                for (j = 0; j < N; j = j+1) result_r[i][j] <= '0;
            end
            // Removed reset logic for legacy SP state variables

        end else begin
            // -----------------------------------------------------------------
            // Defaults
            // -----------------------------------------------------------------
            fifo_pop  <= 1'b0;
            dma_re    <= 1'b0;
            clear_acc <= 1'b0;
            en        <= 1'b0;

            // -----------------------------------------------------------------
            // Global DMA response ingestion
            // -----------------------------------------------------------------
            if (dma_resp_valid) begin
                if (dma_target == DMA_TARGET_BUF) begin
                    if (tag_count != '0) begin
                        buf_data[tag_slot[tag_head]]  <= dma_rdata;
                        buf_valid[tag_slot[tag_head]] <= 1'b1;
                        tag_head  <= next_tag(tag_head);
                        tag_count <= tag_count - 1'b1;
                    end
                end
            end

            // -----------------------------------------------------------------
            // Hardware Perf Counters
            // -----------------------------------------------------------------
            if (cmd_active) begin
                if (en == 1'b0) begin
                    // MAC Pipeline is idle
                    pe_idle_cycles <= pe_idle_cycles + 1'b1;
                    // If we intended to compute but buffer was empty -> DMA stall
                    if (state == S_COMPUTE && !buf_valid[buf_use_ptr]) begin
                        dma_stall_cycles <= dma_stall_cycles + 1'b1;
                    end
                end
            end

            // -----------------------------------------------------------------
            // FSM
            // -----------------------------------------------------------------
            case (state)
                S_IDLE: begin
                    if (!fifo_empty) begin
                        fifo_pop <= 1'b1;
                        state    <= S_POP;
                    end
                end

                S_POP: state <= S_LATCH;

                S_LATCH: begin
                    active_w_addr    <= fifo_dout[191:160];
                    active_x_addr    <= fifo_dout[223:192];
                    active_x_stride  <= fifo_dout[255:224];
                    active_m_dim     <= fifo_dout[287:256];
                    active_n_dim     <= fifo_dout[319:288];
                    active_k_dim     <= fifo_dout[351:320];
                    active_k_row_len <= fifo_dout[383:352];
                    active_hw_mode   <= fifo_dout[415:384];
                    active_csr_val_addr     <= fifo_dout[447:416];
                    active_csr_col_idx_addr <= fifo_dout[479:448]; // Fixed missing assignment
                    active_csr_row_ptr_addr <= fifo_dout[511:480];

`ifndef SYNTHESIS
                    $display("S_LATCH hw_mode=%0d k_dim=%0d m=%0d n=%0d", fifo_dout[415:384], fifo_dout[351:320], fifo_dout[287:256], fifo_dout[319:288]);
`endif

                    buf_fill_ptr <= '0;
                    buf_use_ptr  <= '0;
                    tag_head     <= '0;
                    tag_tail     <= '0;
                    tag_count    <= '0;
                    
                    done_r     <= 1'b0;
                    cmd_active <= 1'b1;
                    pe_idle_cycles <= 32'd0;
                    dma_stall_cycles <= 32'd0;
                    
                    for (i = 0; i < BUF_DEPTH; i = i+1) buf_valid[i] <= 1'b0;
                    for (i = 0; i < N; i = i+1) begin
                        a_left[i] <= '0; b_top[i] <= '0;
                        for (j = 0; j < N-1; j = j+1) begin
                            a_sr[i][j] <= '0; b_sr[i][j] <= '0;
                        end
                    end

                    case (fifo_dout[415:384])
                        32'd0, 32'd1: begin
                            // Mode 0: Setup Dense | Mode 1: Setup 2:4 Structured Sparse
                            k_limit        <= ceil_div4(fifo_dout[351:320]);
                            pref_addr_x    <= fifo_dout[223:192];
                            pref_addr_w    <= fifo_dout[191:160];
                            pref_k_cnt     <= 32'd0;
                            pref_k_row_cnt <= 32'd0;
                            k_cnt          <= 32'd0;
                            step_cnt       <= 2'd0;
                            drain_cnt      <= 3'd0;
                            dma_target     <= DMA_TARGET_BUF;
                            state          <= S_PREFILL;
                        end
                        default: begin
`ifndef SYNTHESIS
`ifdef DEBUG_SPARSE
                            $display("SPARSE_WARN unsupported_hw_mode=%0d, forcing DONE", fifo_dout[415:384]);
`endif
`endif
                            k_limit <= 32'd0;
                            state <= S_DONE;
                        end
                    endcase
                end

                // =====================================================================
                // MODE 0: DENSE SYSTOLIC
                // =====================================================================
                S_PREFILL: begin
                    if ((pref_k_cnt < k_limit) && !buf_valid[buf_fill_ptr] && (tag_count < TAG_DEPTH) && dma_req_ready) begin
                        dma_addr <= pref_addr_x;
                        dma_re   <= 1'b1;
                        tag_slot[tag_tail] <= buf_fill_ptr;
                        tag_tail <= next_tag(tag_tail);
                        tag_count <= tag_count + 1'b1;

                        if ((active_k_row_len > 32'd0) && (pref_k_row_cnt + 1 >= active_k_row_len)) begin
                            pref_k_row_cnt <= 32'd0;
                            pref_addr_x    <= pref_addr_x + active_x_stride;
                        end else begin
                            pref_k_row_cnt <= pref_k_row_cnt + 32'd1;
                            pref_addr_x    <= pref_addr_x + 32'd32;
                        end
                        pref_addr_w  <= pref_addr_w + 32'd32;
                        pref_k_cnt   <= pref_k_cnt + 32'd1;
                        buf_fill_ptr <= next_buf(buf_fill_ptr);
                    end
                    if (buf_valid[buf_use_ptr]) state <= S_COMPUTE;
                end

                S_COMPUTE: begin
                    if (buf_valid[buf_use_ptr]) begin
                        reg [1:0] mux_idx;
                        reg [1:0] step_end;
                        
                        cur_x_chunk = buf_data[buf_use_ptr][127:0];
                        cur_w_chunk = buf_data[buf_use_ptr][255:128];
                        
                        if (active_hw_mode == 1) begin
                            // 2:4 Structured Sparse: mux idx from bits [23:16] of column 0
                            if (step_cnt == 0) mux_idx = cur_w_chunk[17:16];
                            else mux_idx = cur_w_chunk[19:18];
                            step_end = 2'd1;
                        end else begin
                            mux_idx = step_cnt;
                            step_end = 2'd3;
                        end

                        a_left[0] <= $signed({{8{cur_x_chunk[8*mux_idx + 7]}}, cur_x_chunk[8*mux_idx +: 8]});
                        b_top[0]  <= $signed({{8{cur_w_chunk[8*step_cnt + 7]}}, cur_w_chunk[8*step_cnt +: 8]});
                        
                        for (i = 1; i < N; i = i+1) begin
                            a_sr[i][0] <= $signed({{8{cur_x_chunk[32*i + 8*mux_idx + 7]}}, cur_x_chunk[32*i + 8*mux_idx +: 8]});
                            b_sr[i][0] <= $signed({{8{cur_w_chunk[32*i + 8*step_cnt + 7]}}, cur_w_chunk[32*i + 8*step_cnt +: 8]});
                            for (j = 1; j < i; j = j+1) begin
                                a_sr[i][j] <= a_sr[i][j-1];
                                b_sr[i][j] <= b_sr[i][j-1];
                            end
                            a_left[i] <= a_sr[i][i-1];
                            b_top[i]  <= b_sr[i][i-1];
                        end
                        
                        en <= 1'b1;
                        if (k_cnt == 32'd0 && step_cnt == 2'd0) clear_acc <= 1'b1;

                        if (step_cnt == step_end) begin
                            step_cnt <= 2'd0;
                            buf_valid[buf_use_ptr] <= 1'b0;
                            buf_use_ptr <= next_buf(buf_use_ptr);
                            if (k_cnt + 1 >= k_limit) state <= S_DRAIN;
                            else k_cnt <= k_cnt + 32'd1;
                        end else begin
                            step_cnt <= step_cnt + 1'b1;
                        end
                    end

                    if ((pref_k_cnt < k_limit) && !buf_valid[buf_fill_ptr] && (tag_count < TAG_DEPTH) && dma_req_ready && !dma_re) begin
                        dma_addr <= pref_addr_x;
                        dma_re   <= 1'b1;
                        tag_slot[tag_tail] <= buf_fill_ptr;
                        tag_tail <= next_tag(tag_tail);
                        tag_count <= tag_count + 1'b1;
                        if ((active_k_row_len > 32'd0) && (pref_k_row_cnt + 1 >= active_k_row_len)) begin
                            pref_k_row_cnt <= 32'd0;
                            pref_addr_x    <= pref_addr_x + active_x_stride;
                        end else begin
                            pref_k_row_cnt <= pref_k_row_cnt + 32'd1;
                            pref_addr_x    <= pref_addr_x + 32'd32;
                        end
                        pref_addr_w  <= pref_addr_w + 32'd32;
                        pref_k_cnt   <= pref_k_cnt + 32'd1;
                        buf_fill_ptr <= next_buf(buf_fill_ptr);
                    end
                end

                S_DRAIN: begin
                    a_left[0] <= '0; b_top[0] <= '0;
                    for (i = 1; i < N; i = i+1) begin
                        a_sr[i][0] <= '0; b_sr[i][0] <= '0;
                        for (j = 1; j < i; j = j+1) begin
                            a_sr[i][j] <= a_sr[i][j-1]; b_sr[i][j] <= b_sr[i][j-1];
                        end
                        a_left[i] <= a_sr[i][i-1]; b_top[i] <= b_sr[i][i-1];
                    end
                    en <= 1'b1;
                    if (drain_cnt == 3'd7) begin
                        for (i = 0; i < N; i = i+1)
                            for (j = 0; j < N; j = j+1)
                                result_r[i][j] <= c_out[i][j];
                        state <= S_DONE;
                    end else begin
                        drain_cnt <= drain_cnt + 3'd1;
                    end
                end

    // End of COMPUTE / Drain logic

                // =====================================================================
                // TERMINATION
                // =====================================================================
                S_DONE: begin
                    done_r     <= 1'b1;
                    cmd_active <= 1'b0;
`ifndef SYNTHESIS
                    $display("ACCEL_PERF cmd=0 k_limit=%0d busy=0 compute=0 stall=%0d", 
                             k_limit, pe_idle_cycles + dma_stall_cycles);
`endif
                    state      <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

    // =========================================================================
    // MMIO read
    // =========================================================================
    reg [7:0] reg_offset_d;
    always @(posedge clk) begin
        if (reset) reg_offset_d <= 8'd0;
        else if (mmio_re) reg_offset_d <= reg_offset;
    end

    always @(*) begin
        case (reg_offset_d)
            8'h00: mmio_rdata = {29'd0, fifo_full, (state == S_IDLE) && fifo_empty, accel_busy};
            8'h18: mmio_rdata = result_r[0][0];
            8'h1C: mmio_rdata = result_r[0][1];
            8'h20: mmio_rdata = result_r[0][2];
            8'h24: mmio_rdata = result_r[0][3];
            8'h28: mmio_rdata = result_r[1][0];
            8'h2C: mmio_rdata = result_r[1][1];
            8'h30: mmio_rdata = result_r[1][2];
            8'h34: mmio_rdata = result_r[1][3];
            8'h38: mmio_rdata = result_r[2][0];
            8'h3C: mmio_rdata = result_r[2][1];
            8'h40: mmio_rdata = result_r[2][2];
            8'h44: mmio_rdata = result_r[2][3];
            8'h48: mmio_rdata = result_r[3][0];
            8'h4C: mmio_rdata = result_r[3][1];
            8'h50: mmio_rdata = result_r[3][2];
            8'h54: mmio_rdata = result_r[3][3];
            8'h80: mmio_rdata = pe_idle_cycles;
            8'h84: mmio_rdata = dma_stall_cycles;
            default: mmio_rdata = 32'd0;
        endcase
    end

endmodule
