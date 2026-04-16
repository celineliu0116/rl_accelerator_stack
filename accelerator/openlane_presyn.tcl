package require openlane;
prep -design riscv_top {*}$argv
run_synthesis
puts_info "pre-syn complete"
