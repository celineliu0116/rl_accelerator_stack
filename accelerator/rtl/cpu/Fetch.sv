// FETCH STAGE (3-stage pipeline front end; no local flush)
// instantiates the BHT/BTB for prediction
// redirects on mispredicts (pc mux receives alu output that computes the correct branch address)
// FETCH STAGE: PC + prediction only (no instruction handling here)
module Fetch #(
    parameter BHT_ENTRIES = 256,
    parameter BTB_ENTRIES = 256,
    parameter IDX_BITS    = $clog2(BHT_ENTRIES)
  )(
    input  wire        clk,
    input  wire        reset,
    input  wire        stall,
    input  wire        dual_issued,
  
    // Redirect from D/X stage (has highest priority)
    input wire        taken,      // passed from D/X stage
    input wire        mispredict,
    input wire [31:0] actual_target,
    input wire [31:0] pcx_plus4,  // resolved fall through = branch_pc + 4 (from EX)
  
    // Predictor updates (from DX stage)
    input wire        cf_upd_valid,
    input wire [31:0] cf_upd_pc,
    input wire        cf_upd_taken,
    input wire [31:0] cf_upd_target,  // same thing as actual_target, input to btb
  
    // To Decode/Execute (and IMEM addr wiring at top-level/DX)
    output wire [31:0] pc_req,     // address to fetch this cycle
    output wire        pred_req,   // predicted-taken bit used for this request
    output wire [31:0] btb_target_out
  );
  
    // -PC register / next-PC select-
    reg  [31:0] pc_F = `PC_RESET;
    wire [31:0] pc_plus4 = pc_F + 32'd4;
    wire [31:0] pc_plus8 = pc_F + 32'd8;
    wire [31:0] pc_sequential = pc_plus8;
  
    // Predictors (combinational read on pc_F)
    wire        bht_taken;
    wire        btb_hit;
    wire [31:0] btb_target;
  
    BHT2bc #(.ENTRIES(BHT_ENTRIES), .IDX_BITS(IDX_BITS)) u_bht (
      .clk        (clk),
      .reset      (reset),
      .pc_r       (pc_F),
      .pred_taken (bht_taken),
      .upd_valid  (cf_upd_valid),
      .upd_pc     (cf_upd_pc),
      .upd_taken  (cf_upd_taken)
    );
  
    BTB #(.ENTRIES(BTB_ENTRIES), .IDX_BITS(IDX_BITS)) u_btb (
      .clk        (clk),
      .reset      (reset),
      .pc_r       (pc_F),
      .hit        (btb_hit),
      .target_r   (btb_target),
      .upd_valid  (cf_upd_valid && cf_upd_taken),
      .upd_pc     (cf_upd_pc),
      .upd_target (cf_upd_target)  // same thing as actual_target, why not pass in actual target here? actual target is output of ALU
    );

    assign btb_target_out = btb_target;
  
    wire          use_btb = bht_taken && btb_hit;   // BHT+BTB agree
    wire      redir_taken = mispredict && taken;    // use branch/jump target computed by ALU, mispredict = 1 and taken = 1 for jump asserted in execute.v
    wire redir_not_taken  = mispredict && !taken;   // use PCx + 4
  
    // redirect > prediction > sequential
    wire [1:0] pc_sel = redir_taken     ? 2'b01 : // ALU target (actual taken)
                        redir_not_taken ? 2'b11 : // PCx+4 (actual not-taken)
                        use_btb         ? 2'b00 : // BTB target (predict taken only)
                                          2'b10;  // PC+4 (sequential)
  
    reg [31:0] pc_next;
    always @(*) begin
      case (pc_sel)
        2'b00: pc_next = btb_target;
        2'b01: pc_next = actual_target;
        2'b10: pc_next = pc_sequential;
        2'b11: pc_next = pcx_plus4;
      endcase
    end
  
    always @(posedge clk) begin
      if (reset) begin
        pc_F <= `PC_RESET;
      end else if (!stall) begin
        pc_F <= pc_next;
      end
    end
  
    // Outputs to DX/top-level
    assign pc_req   = pc_F;      // IMEM addr should use this *this cycle*
    assign pred_req = use_btb;   // carry to DX for bookkeeping/flush, comparing predicted direction (taken/not-taken) vs real outcome. avoids ghost "taken without target" case.
  
  // SVA: Upon reset, PC must equal PC_RESET
  // This check ensures that right after asserting reset, pc_F is initialized to the reset address.
  // disable iff(reset) avoids firing while reset is active.
  `ifndef SYNTHESIS
  // 1) On the cycle *after* reset is asserted, PC must be PC_RESET
  assert property (@(posedge clk)
    reset |=> (pc_F == `PC_RESET)
  ) else $fatal("PC not PC_RESET on cycle after reset");

  // 2) While reset remains high, PC must stay at PC_RESET
  // assert property (@(posedge clk)
  //   reset |-> (pc_F == `PC_RESET)
  // ) else $fatal("PC changed during reset");

  // 3) On the first cycle after reset deasserts, PC is still PC_RESET
  assert property (@(posedge clk)
    $fell(reset) |-> (pc_F == `PC_RESET)
  ) else $fatal("PC not PC_RESET right after reset deassert");
`endif
  
  // sync. reset
  
  endmodule

  
