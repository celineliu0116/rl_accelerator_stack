`include "legacy/MatmulAcceleratorLegacy.sv"
`include "systolic/SystolicPE.sv"
`include "systolic/SystolicArray.sv"
`include "systolic/MatmulAcceleratorSystolic.sv"

module MatmulAccelerator (
    input  wire         clk,
    input  wire         reset,
    input  wire [31:0]  mmio_addr,
    input  wire [31:0]  mmio_wdata,
    input  wire [3:0]   mmio_we,
    input  wire         mmio_re,
    output wire [31:0]  mmio_rdata,
    output wire [31:0]  dma_addr,
    output wire         dma_re,
    input  wire         dma_req_ready,
    input  wire         dma_resp_valid,
    input  wire [255:0] dma_rdata,
    output wire         accel_busy
);

`ifdef USE_SYSTOLIC_ACCEL
    MatmulAcceleratorSystolic u_matmul_impl (
`else
    MatmulAcceleratorLegacy u_matmul_impl (
`endif
        .clk          (clk),
        .reset        (reset),
        .mmio_addr    (mmio_addr),
        .mmio_wdata   (mmio_wdata),
        .mmio_we      (mmio_we),
        .mmio_re      (mmio_re),
        .mmio_rdata   (mmio_rdata),
        .dma_addr     (dma_addr),
        .dma_re       (dma_re),
        .dma_req_ready(dma_req_ready),
        .dma_resp_valid(dma_resp_valid),
        .dma_rdata    (dma_rdata),
        .accel_busy   (accel_busy)
    );

endmodule
