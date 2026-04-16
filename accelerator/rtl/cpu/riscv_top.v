`include "const.vh"

module riscv_top
(
  input clk,
  input reset,

  output                      mem_req_valid,
  input                       mem_req_ready,
  output                      mem_req_rw,
  output [`MEM_ADDR_BITS-1:0] mem_req_addr,
  output [`MEM_TAG_BITS-1:0]  mem_req_tag,

  output                      mem_req_data_valid,
  input                       mem_req_data_ready,
  output [`MEM_DATA_BITS-1:0] mem_req_data_bits,
  output [(`MEM_DATA_BITS/8)-1:0] mem_req_data_mask,

  input                       mem_resp_valid,
  input [`MEM_TAG_BITS-1:0]   mem_resp_tag,
  input [`MEM_DATA_BITS-1:0]  mem_resp_data,
  output [31:0]               csr
);

  // ---- CPU-side signals (directly from Riscv151) ----
  wire [31:0]   cpu_dcache_addr;
  wire [31:0]   cpu_dcache_din;
  wire [31:0]   dcache_dout_mem;  // from Memory151
  wire          cpu_dcache_re;
  wire [3:0]    cpu_dcache_we;
  wire [31:0]   icache_addr;
  wire [63:0]   icache_dout;
  wire          icache_valid;
  wire [31:0]   icache_resp_addr;
  wire          icache_re;
  wire          stall;

  // ---- Accelerator signals ----
  wire [31:0]   accel_mmio_rdata;
  wire [31:0]   accel_dma_addr;
  wire          accel_dma_re;
  wire          accel_dma_req_ready;
  wire          accel_dma_resp_valid;
  wire [255:0]  accel_dma_rdata;
  wire          accel_busy;

  // ---- Address decode: CPU accessing MMIO? (addr bit 31) ----
  wire cpu_mmio_access = cpu_dcache_addr[31];

  // ---- Muxed dcache signals going to Memory151 ----
  // CPU now has exclusive access to dcache port (Memory151 handles DMA via separate port)
  wire [31:0]   dcache_addr_to_mem = cpu_dcache_addr;
  wire [31:0]   dcache_din_to_mem  = cpu_dcache_din;
  wire          dcache_re_to_mem   = cpu_mmio_access ? 1'b0 : cpu_dcache_re;
  wire [3:0]    dcache_we_to_mem   = cpu_mmio_access ? 4'b0 : cpu_dcache_we;

  // ---- Muxed read data back to CPU ----
  // Register the MMIO flag for 1-cycle memory latency alignment
  reg cpu_mmio_access_d;
  always @(posedge clk) begin
    if (reset)
      cpu_mmio_access_d <= 0;
    else
      cpu_mmio_access_d <= cpu_mmio_access && cpu_dcache_re;

  end

  wire [31:0] dcache_dout_to_cpu = cpu_mmio_access_d ? accel_mmio_rdata
                                                     : dcache_dout_mem;

  // ---- Memory151 ----
  Memory151 mem(
    .dcache_dout(dcache_dout_mem),
    .icache_dout(icache_dout),
    .icache_valid(icache_valid),
    .icache_resp_addr(icache_resp_addr),
    .stall(stall),
    .mem_req_valid(mem_req_valid),
    .mem_req_rw(mem_req_rw),
    .mem_req_addr(mem_req_addr),
    .mem_req_tag(mem_req_tag),
    .mem_req_data_valid(mem_req_data_valid),
    .mem_req_data_bits(mem_req_data_bits),
    .mem_req_data_mask(mem_req_data_mask),
    .clk(clk),
    .reset(reset),
    .dcache_addr(dcache_addr_to_mem),
    .icache_addr(icache_addr),
    .dcache_we(dcache_we_to_mem),
    .dcache_re(dcache_re_to_mem),
    .icache_re(icache_re),
    .dcache_din(dcache_din_to_mem),
    .mem_req_ready(mem_req_ready),
    .mem_req_data_ready(mem_req_data_ready),
    .mem_resp_valid(mem_resp_valid),
    .mem_resp_data(mem_resp_data),
    .mem_resp_tag(mem_resp_tag),
    
    // DMA Interface
    .dma_req_valid(accel_dma_re),    // Accel drives RE
    .dma_req_ready(accel_dma_req_ready),
    .dma_req_addr(accel_dma_addr),   // Accel drives byte addr
    .dma_resp_valid(accel_dma_resp_valid),
    .dma_resp_data(accel_dma_rdata)  // 256-bit data back to accel
    );

  // ---- Matmul Accelerator ----
  MatmulAccelerator accel(
    .clk(clk),
    .reset(reset),
    .mmio_addr(cpu_dcache_addr),
    .mmio_wdata(cpu_dcache_din),
    .mmio_we(cpu_mmio_access ? cpu_dcache_we : 4'b0),
    .mmio_re(cpu_mmio_access ? cpu_dcache_re : 1'b0),
    .mmio_rdata(accel_mmio_rdata),
    .dma_addr(accel_dma_addr),
    .dma_re(accel_dma_re),
    .dma_req_ready(accel_dma_req_ready),
    .dma_resp_valid(accel_dma_resp_valid),
    .dma_rdata(accel_dma_rdata),
    .accel_busy(accel_busy));

  // ---- RISC-V 151 CPU (unchanged) ----
  Riscv151 cpu(
      .dcache_addr(cpu_dcache_addr),
      .icache_addr(icache_addr),
      .dcache_we(cpu_dcache_we),
      .dcache_re(cpu_dcache_re),
      .icache_re(icache_re),
      .dcache_din(cpu_dcache_din),
      .clk(clk),
      .reset(reset),
      .dcache_dout(dcache_dout_to_cpu),
      .icache_dout(icache_dout),
      .icache_valid(icache_valid),
      .icache_resp_addr(icache_resp_addr),
      .csr(csr),
      .stall(stall));

endmodule
