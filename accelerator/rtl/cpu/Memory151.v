
module Memory151( 
  input clk,
  input reset,

  // Cache <=> CPU interface
  input  [31:0] dcache_addr,
  input  [31:0] icache_addr,
  input  [3:0]  dcache_we,
  input         dcache_re,
  input         icache_re,
  input  [31:0] dcache_din,
  output [31:0] dcache_dout,
  output [63:0] icache_dout,
  output        icache_valid,
  output [31:0] icache_resp_addr,
  output        stall,

  // Arbiter <=> Main memory interface
  output                       mem_req_valid,
  input                        mem_req_ready,
  output                       mem_req_rw,
  output [`MEM_ADDR_BITS-1:0]  mem_req_addr,
  output [`MEM_TAG_BITS-1:0]   mem_req_tag,

  output                       mem_req_data_valid,
  input                        mem_req_data_ready,
  output [`MEM_DATA_BITS-1:0]  mem_req_data_bits,
  output [(`MEM_DATA_BITS/8)-1:0]  mem_req_data_mask,

  input                        mem_resp_valid,
  input [`MEM_DATA_BITS-1:0]   mem_resp_data,
  input [`MEM_TAG_BITS-1:0]    mem_resp_tag,

  // DMA Port (Read-only, 128-bit)
  input                        dma_req_valid,
  output                       dma_req_ready,
  input [`MEM_ADDR_BITS+1:0]   dma_req_addr, // Full byte address (30 bits for word address)
  output                       dma_resp_valid,
  output [2*`MEM_DATA_BITS-1:0]  dma_resp_data

);

wire i_stall_n;
wire d_stall_n;

wire ic_mem_req_valid;
wire ic_mem_req_ready;
wire [`MEM_ADDR_BITS-1:0]  ic_mem_req_addr;
wire ic_mem_resp_valid;

wire dc_mem_req_valid;
wire dc_mem_req_ready;
wire dc_mem_req_rw;
wire [`MEM_ADDR_BITS-1:0]  dc_mem_req_addr;
wire dc_mem_resp_valid;

wire [(`MEM_DATA_BITS/8)-1:0]  dc_mem_req_mask;
wire [31:0] icache_dout_32;
wire [`CPU_ADDR_BITS-3:0] icache_resp_addr_word;

`ifdef no_cache_mem
no_cache_mem_64 icache (
  .clk(clk),
  .reset(reset),
  .cpu_req_valid(icache_re),
  .cpu_req_ready(i_stall_n),
  .cpu_req_addr(icache_addr[31:2]),
  .cpu_resp_valid(icache_valid),
  .cpu_resp_data(icache_dout),
  .cpu_resp_addr(icache_resp_addr_word)
);
assign icache_resp_addr = {icache_resp_addr_word, 2'b00};

no_cache_mem dcache (
  .clk(clk),
  .reset(reset),
  .cpu_req_valid((| dcache_we) || dcache_re),
  .cpu_req_ready(d_stall_n),
  .cpu_req_addr(dcache_addr[31:2]),
  .cpu_req_data(dcache_din),
  .cpu_req_write(dcache_we),
  .cpu_resp_valid(),
  .cpu_resp_data(dcache_dout),

  // DMA Port
  .dma_req_valid(dma_req_valid),
  .dma_req_ready(dma_req_ready),
  .dma_req_addr(dma_req_addr[`MEM_ADDR_BITS+1:2]), // Byte address -> Word address
  .dma_resp_valid(dma_resp_valid),
  .dma_resp_data(dma_resp_data)
);
assign stall =  ~i_stall_n || ~d_stall_n;

`else
cache icache (
  .clk(clk),
  .reset(reset),
  .cpu_req_valid(icache_re),
  .cpu_req_ready(i_stall_n),
  .cpu_req_addr(icache_addr[31:2]),
  .cpu_req_data(), // core does not write to icache
  .cpu_req_write(4'b0), // never write
  .cpu_resp_valid(icache_valid),
  .cpu_resp_data(icache_dout_32),
  .mem_req_valid(ic_mem_req_valid),
  .mem_req_ready(ic_mem_req_ready),
  .mem_req_addr(ic_mem_req_addr),
  .mem_req_data_valid(),
  .mem_req_data_bits(),
  .mem_req_data_mask(),
  .mem_req_data_ready(),
  .mem_req_rw(),
  .mem_resp_valid(ic_mem_resp_valid),
  .mem_resp_data(mem_resp_data)
);
assign icache_dout = {32'b0, icache_dout_32};
assign icache_resp_addr = {icache_addr[31:2], 2'b00};

cache dcache (
  .clk(clk),
  .reset(reset),
  .cpu_req_valid((| dcache_we) || dcache_re),
  .cpu_req_ready(d_stall_n),
  .cpu_req_addr(dcache_addr[31:2]),
  .cpu_req_data(dcache_din),
  .cpu_req_write(dcache_we),
  .cpu_resp_valid(),
  .cpu_resp_data(dcache_dout),
  .mem_req_valid(dc_mem_req_valid),
  .mem_req_ready(dc_mem_req_ready),
  .mem_req_addr(dc_mem_req_addr),
  .mem_req_rw(dc_mem_req_rw),
  .mem_req_data_valid(mem_req_data_valid),
  .mem_req_data_bits(mem_req_data_bits),
  .mem_req_data_mask(mem_req_data_mask),
  .mem_req_data_ready(mem_req_data_ready),
  .mem_resp_valid(dc_mem_resp_valid),
  .mem_resp_data(mem_resp_data)
);
assign stall =  ~i_stall_n || ~d_stall_n;

//                           ICache 
//                         /        \
//   Riscv151 --- Memory151          Arbiter <--> ExtMemModel
//                         \        /
//                           DCache 

riscv_arbiter arbiter (
  .clk(clk),
  .reset(reset),
  .ic_mem_req_valid(ic_mem_req_valid),
  .ic_mem_req_ready(ic_mem_req_ready),
  .ic_mem_req_addr(ic_mem_req_addr),
  .ic_mem_resp_valid(ic_mem_resp_valid),

  .dc_mem_req_valid(dc_mem_req_valid),
  .dc_mem_req_ready(dc_mem_req_ready),
  .dc_mem_req_rw(dc_mem_req_rw),
  .dc_mem_req_addr(dc_mem_req_addr),
  .dc_mem_resp_valid(dc_mem_resp_valid),

  .mem_req_valid(mem_req_valid),
  .mem_req_ready(mem_req_ready),
  .mem_req_rw(mem_req_rw),
  .mem_req_addr(mem_req_addr),
  .mem_req_tag(mem_req_tag),
  .mem_resp_valid(mem_resp_valid),
  .mem_resp_tag(mem_resp_tag)
);
);

assign dma_req_ready = 1'b0;
assign dma_resp_valid = 1'b0;
assign dma_resp_data = 256'd0;

`endif

endmodule
