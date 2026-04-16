set liberty_file "/Users/peter/.ciel/sky130A/libs.ref/sky130_fd_sc_hd/lib/sky130_fd_sc_hd__tt_025C_1v80.lib"
set netlist_file "/work/OpenLane/designs/riscv_top/runs/syn_single_20260219/results/synthesis/Riscv151.v"

read_liberty $liberty_file
read_verilog $netlist_file
link_design Riscv151

create_clock -name clk -period 17.0 [get_ports clk]
set_clock_uncertainty 0.25 [get_clocks clk]
set_clock_transition 0.15 [get_clocks clk]
set_timing_derate -early 0.95
set_timing_derate -late 1.05

set_false_path -from [all_inputs]
set_false_path -to [all_outputs]

puts "===== report_clock_properties ====="
report_clock_properties [get_clocks clk]

puts "===== check input->* paths (should be empty due false-path) ====="
report_checks -path_delay max -from [all_inputs] -group_count 3 -format summary

puts "===== check *->output paths (should be empty due false-path) ====="
report_checks -path_delay max -to [all_outputs] -group_count 3 -format summary

puts "===== top max timing paths ====="
report_checks -path_delay max -group_count 5 -sort_by_slack

puts "===== summary ====="
report_worst_slack -max
report_tns
report_wns
