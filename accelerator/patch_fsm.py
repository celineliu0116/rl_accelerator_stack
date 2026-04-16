import re
path = "rtl/accel/systolic/MatmulAcceleratorSystolic.sv"
with open(path, "r") as f:
    orig = f.read()

# Instrument SP_LOAD_X
orig = orig.replace(
    "state <= SP_LOAD_X;",
    "state <= SP_LOAD_X;\n`ifndef SYNTHESIS\n$display(\"[%0t] -> SP_LOAD_X limit=%0d limit_raw=%0d\", $time, ceil_div8(fifo_dout[351:320]), fifo_dout[351:320]);\n`endif"
)

# Instrument SP_LOAD_PTR
orig = orig.replace(
    "state <= SP_LOAD_PTR;",
    "state <= SP_LOAD_PTR;\n`ifndef SYNTHESIS\n$display(\"[%0t] -> SP_LOAD_PTR limit=%0d N=%0d\", $time, ceil_div8(active_n_dim + 1), active_n_dim);\n`endif"
)

# Instrument SP_PREP_ROW
orig = orig.replace(
    "state <= SP_PREP_ROW;",
    "state <= SP_PREP_ROW;\n`ifndef SYNTHESIS\n$display(\"[%0t] -> SP_PREP_ROW\", $time);\n`endif"
)

orig = orig.replace(
    "state <= SP_COMPUTE;",
    "state <= SP_COMPUTE;\n`ifndef SYNTHESIS\n$display(\"[%0t] -> SP_COMPUTE row=%0d ptr=%0d end=%0d klimit=%0d\", $time, sp_n_cnt, read_ptr(0), read_ptr(1), ceil_div8(read_ptr(active_n_dim)));\n`endif"
)

# Instrument Row Transition inside SP_COMPUTE
orig = orig.replace(
    "// For the next row, update end pointer",
    "// For the next row, update end pointer\n`ifndef SYNTHESIS\n$display(\"[%0t] SP_COMPUTE row done! next=%0d read_ptr(next+1)=%0d\", $time, sp_n_cnt + 1'b1, read_ptr(sp_n_cnt + 2'd2));\n`endif"
)

with open(path, "w") as f:
    f.write(orig)
