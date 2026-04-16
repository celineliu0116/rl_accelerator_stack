# Makefile for Verilator simulation of RISC-V ML Inference
# Enterprise Layout: rtl/, firmware/, compiler/, auto_tuner/, workloads/

# ---- Domain Paths (self-contained, no ../ references) ----
CPU_DIR     := rtl/cpu
ACCEL_DIR   := rtl/accel
SIM_DIR     := rtl/sim
FW_DIR      := firmware
FW_INC      := firmware/include
COMP_DIR    := compiler
BENCH_BASE  := benchmarks
START_S     := $(CURDIR)/benchmarks/bmark/start.s
LINKER_LD   := $(FW_DIR)/inference.ld
BUILD_DIR   := build
VERILATOR_CFG_STAMP := $(BUILD_DIR)/.verilator_cfg.stamp
INFERENCE_CFG_STAMP := $(BUILD_DIR)/.inference_cfg.stamp

# ---- RISC-V toolchain ----
RISCV       = riscv64-unknown-elf
GCC_OPTS    = -march=rv32im_zicsr -mabi=ilp32 -static -mcmodel=medany \
              -fvisibility=hidden -nostdlib -nostartfiles -Wl,--build-id=none -O3
CFLAGS_EXTRA ?=
INFERENCE_SRC ?= inference_generic.c
GENERIC_CFLAGS ?= -O3 -fno-strict-aliasing
PYTHON ?= python3
MCP_REPORT ?= data/mcp_validation_report_ci.json
MCP_BASELINE_CONFIG ?= .github/mcp_regression_baseline.json
MCP_TIMESTEPS ?= 4
MCP_EVAL_EPISODES ?= 2
MCP_N_STEPS ?= 4
MCP_BATCH_SIZE ?= 4
MCP_RUN_ID_FILE ?= data/mcp_candidate_run_id_ci.txt
MCP_RUN_VERIFY ?= 1
MCP_RUN_REGRESSION_GATE ?= 1
TRACE_CSV ?= data/trace_dataset_latest.csv
TRACE_PARQUET_DIR ?= data/parquet/trace_dataset_v1
TRACE_PARQUET_META ?= $(TRACE_PARQUET_DIR)/_dataset_meta.json
TRACE_DATASET_SUMMARY_JSON ?= data/audits/trace_dataset_summary.json
BEST_PARAMS_JSON ?= data/best_params_v1.json
BEST_PARAMS_LEDGER_OUT ?= auto_tuner/compiler/bkm_ledger.ml_candidate.json
BEST_PARAMS_SELECTION_POLICY ?= best1
BEST_PARAMS_TOP_K ?= 5
BEST_PARAMS_WITHIN_PCT ?= 0.05
BEST_PARAMS_REGRET_WEIGHT_ALPHA ?= 0.0
BEST_PARAMS_MIN_SAMPLE_WEIGHT ?= 0.05
BEST_PARAMS_CONF_FALLBACK_THRESHOLD ?= 0.0
POLICY_TRACE_CSV ?= data/trace_dataset_all.csv
POLICY_TOPK_JSON ?= data/best_params_v1_topk5_all.json
POLICY_WITHIN_JSON ?= data/best_params_v1_within5pct_all.json
POLICY_AB_OUT_CSV ?= data/audits/policy_ab_regret_34keys.csv
POLICY_AB_OUT_JSON ?= data/audits/policy_ab_regret_summary.json
TARGETED_CAMPAIGN_JSON ?= data/campaigns/targeted_campaign_v1.json
RL_TIMESTEPS_LONG ?= 500000
RL_CHECKPOINT_EVERY_STEPS ?= 10000
RL_CHECKPOINT_KEEP_LAST ?= 5
RL_CAMPAIGN_ID ?= zoo_long
RL_CAMPAIGN_STAGE ?= tranche_next
RL_TARGETED_SAMPLE_PROB ?= 0.6
CANONICAL_CONTRACT_JSON ?= auto_tuner/canonical_policy_contract_v1.json
CANONICAL_TRACE_PARQUET_DIR ?= data/canonical/trace_dataset_v1
CANONICAL_TRACE_PARQUET_META ?= $(CANONICAL_TRACE_PARQUET_DIR)/_dataset_meta.json
CANONICAL_RL_SPACE_JSON ?= data/canonical/policy/rl_exploration_space_v1.json
CANONICAL_BEST_PARAMS_JSON ?= data/canonical/policy/best_params_v1.json
CANONICAL_POLICY_MANIFEST_JSON ?= data/canonical/policy/best_params_v1.manifest.json

# ---- Verilator settings ----
VERILATOR   = verilator
ENABLE_DUAL_ISSUE ?= 1
USE_SYSTOLIC_ACCEL ?= 1
VFLAGS      = --cc --exe --build --trace \
              -Wno-fatal \
              -Dno_cache_mem \
              $(if $(filter 1,$(ENABLE_DUAL_ISSUE)),-DENABLE_DUAL_ISSUE,) \
              $(if $(filter 1,$(USE_SYSTOLIC_ACCEL)),-DUSE_SYSTOLIC_ACCEL,) \
              $(EXTRA_VFLAGS) \
              --top-module riscv_top \
              -I$(CPU_DIR) \
              -I$(ACCEL_DIR)

# ---- RTL source files (self-contained) ----
RTL_SRCS = \
	$(CPU_DIR)/riscv_top.v \
	$(CPU_DIR)/Riscv151.v \
	$(CPU_DIR)/Memory151.v \
	$(CPU_DIR)/no_cache_mem.v \
	$(CPU_DIR)/ExtMemModel.v \
	$(CPU_DIR)/Execute.sv \
	$(CPU_DIR)/Fetch.sv \
	$(CPU_DIR)/Writeback.sv \
	$(CPU_DIR)/Control_logic.sv \
	$(CPU_DIR)/RegFile.sv \
	$(CPU_DIR)/ALU.v \
	$(CPU_DIR)/ALUdec.v \
	$(CPU_DIR)/Branch_Comp.v \
	$(CPU_DIR)/BranchPrediction.v \
	$(CPU_DIR)/riscv_arbiter.v \
	$(CPU_DIR)/Multiplier.sv \
	$(CPU_DIR)/HazardUnit.sv \
	$(ACCEL_DIR)/MatmulAccelerator.sv

# ---- Targets ----
.PHONY: all compile_rtl compile_inference run run_bench clean clean_bench export_mnist_model FORCE \
		verify verify_python verify_dense_smoke verify_sparse_mode1 verify_model_blob \
		mcp_regression_gate trace_to_csv train_logistic_baseline materialize_best_params \
		policy_ab_regret_report trace_to_parquet dataset_summary build_targeted_campaign \
		train_logistic_baseline_parquet rl_exploration_campaign export_rl_space_contract \
		policy_train_canonical

all: run

INFERENCE_DEPS = $(FW_DIR)/$(INFERENCE_SRC) $(START_S) $(LINKER_LD)

ifeq ($(INFERENCE_SRC),inference_bare.c)
INFERENCE_DEPS += $(FW_INC)/weights.h $(FW_INC)/test_images.h
endif

ifeq ($(INFERENCE_SRC),inference_generic.c)
INFERENCE_DEPS += $(FW_INC)/test_images.h \
                  $(FW_INC)/model_format.h \
                  $(FW_INC)/accel.h \
                  $(FW_INC)/model_blob.h
INFERENCE_CFLAGS += $(GENERIC_CFLAGS)
endif

# Step 1: Compile inference source -> hex file
INFERENCE_CFLAGS += $(CFLAGS_EXTRA)

$(BUILD_DIR)/inference.hex: $(INFERENCE_DEPS) | $(BUILD_DIR)
	$(RISCV)-gcc $(GCC_OPTS) $(INFERENCE_CFLAGS) -I$(FW_INC) -T $(LINKER_LD) $(START_S) $(FW_DIR)/$(INFERENCE_SRC) -o $(BUILD_DIR)/inference.elf
	$(RISCV)-objdump -D -Mnumeric $(BUILD_DIR)/inference.elf > $(BUILD_DIR)/inference.dump
	$(RISCV)-objcopy $(BUILD_DIR)/inference.elf -O binary $(BUILD_DIR)/inference.bin
	$(PYTHON) bin2hex.py -w 128 $(BUILD_DIR)/inference.bin $(BUILD_DIR)/inference.hex

# Step 2: Build Verilator simulation
obj_dir/Vriscv_top: $(SIM_DIR)/sim_main.cpp $(RTL_SRCS) $(VERILATOR_CFG_STAMP)
	$(VERILATOR) $(VFLAGS) \
		$(RTL_SRCS) \
		$(SIM_DIR)/sim_main.cpp

# Step 3: Run
run: obj_dir/Vriscv_top $(BUILD_DIR)/inference.hex
	./obj_dir/Vriscv_top +loadmem=$(BUILD_DIR)/inference.hex +max-cycles=500000000 $(EXTRA_FLAGS)

verify: verify_python verify_model_blob verify_dense_smoke verify_sparse_mode1

verify_python:
	$(PYTHON) -m unittest discover -s tests -p "test_*.py"

verify_model_blob:
	$(PYTHON) workloads/mnist/mnist_mlp_export.py
	$(PYTHON) compiler/verify_model_blob.py

verify_dense_smoke:
	$(MAKE) run INFERENCE_SRC=inference_generic.c ENABLE_DUAL_ISSUE=1 USE_SYSTOLIC_ACCEL=1

verify_sparse_mode1:
	ACCELERA_ENABLE_SPARSE_MODE1=1 $(MAKE) run INFERENCE_SRC=sparse_mode1_unit.c ENABLE_DUAL_ISSUE=1 USE_SYSTOLIC_ACCEL=1

mcp_regression_gate:
	@cmd="./mcp/run_mcp_regression_gate.sh --python $(PYTHON) --report $(MCP_REPORT) --baseline-config $(MCP_BASELINE_CONFIG) --timesteps $(MCP_TIMESTEPS) --eval-episodes $(MCP_EVAL_EPISODES) --n-steps $(MCP_N_STEPS) --batch-size $(MCP_BATCH_SIZE) --candidate-run-id-output $(MCP_RUN_ID_FILE)"; \
	if [ "$(MCP_RUN_VERIFY)" = "0" ]; then cmd="$$cmd --skip-verify"; fi; \
	if [ "$(MCP_RUN_REGRESSION_GATE)" = "0" ]; then cmd="$$cmd --skip-regression-gate"; fi; \
	eval "$$cmd"

trace_to_csv:
	$(PYTHON) auto_tuner/trace_to_csv.py --workspace $(CURDIR) --output-csv $(TRACE_CSV) --valid-only

trace_to_parquet:
	$(PYTHON) auto_tuner/trace_to_parquet.py --workspace $(CURDIR) --all-traces --valid-only --output-dir $(TRACE_PARQUET_DIR) --metadata-json $(TRACE_PARQUET_META)

dataset_summary:
	$(PYTHON) auto_tuner/dataset_summary.py --workspace $(CURDIR) --all-traces --input-csv $(TRACE_CSV) --input-parquet $(TRACE_PARQUET_DIR) --output-json $(TRACE_DATASET_SUMMARY_JSON)

train_logistic_baseline:
	$(PYTHON) auto_tuner/train_logistic_baseline.py --workspace $(CURDIR) --input-csv $(TRACE_CSV) --output-json $(BEST_PARAMS_JSON) --selection-policy $(BEST_PARAMS_SELECTION_POLICY) --top-k $(BEST_PARAMS_TOP_K) --within-pct $(BEST_PARAMS_WITHIN_PCT) --regret-weight-alpha $(BEST_PARAMS_REGRET_WEIGHT_ALPHA) --min-sample-weight $(BEST_PARAMS_MIN_SAMPLE_WEIGHT) --confidence-fallback-threshold $(BEST_PARAMS_CONF_FALLBACK_THRESHOLD)

train_logistic_baseline_parquet:
	$(PYTHON) auto_tuner/train_logistic_baseline.py --workspace $(CURDIR) --input-parquet $(TRACE_PARQUET_DIR) --output-json $(BEST_PARAMS_JSON) --selection-policy $(BEST_PARAMS_SELECTION_POLICY) --top-k $(BEST_PARAMS_TOP_K) --within-pct $(BEST_PARAMS_WITHIN_PCT) --regret-weight-alpha $(BEST_PARAMS_REGRET_WEIGHT_ALPHA) --min-sample-weight $(BEST_PARAMS_MIN_SAMPLE_WEIGHT) --confidence-fallback-threshold $(BEST_PARAMS_CONF_FALLBACK_THRESHOLD)

materialize_best_params:
	$(PYTHON) auto_tuner/train_logistic_baseline.py --workspace $(CURDIR) --input-csv $(TRACE_CSV) --output-json $(BEST_PARAMS_JSON) --selection-policy $(BEST_PARAMS_SELECTION_POLICY) --top-k $(BEST_PARAMS_TOP_K) --within-pct $(BEST_PARAMS_WITHIN_PCT) --regret-weight-alpha $(BEST_PARAMS_REGRET_WEIGHT_ALPHA) --min-sample-weight $(BEST_PARAMS_MIN_SAMPLE_WEIGHT) --confidence-fallback-threshold $(BEST_PARAMS_CONF_FALLBACK_THRESHOLD) --materialize-ledger-out $(BEST_PARAMS_LEDGER_OUT)

build_targeted_campaign:
	$(PYTHON) auto_tuner/build_targeted_campaign.py --regret-csv $(POLICY_AB_OUT_CSV) --trace-csv $(POLICY_TRACE_CSV) --policy topk --output-json $(TARGETED_CAMPAIGN_JSON)

policy_ab_regret_report:
	$(PYTHON) auto_tuner/policy_regret_report.py --trace-csv $(POLICY_TRACE_CSV) --topk-json $(POLICY_TOPK_JSON) --within-json $(POLICY_WITHIN_JSON) --out-csv $(POLICY_AB_OUT_CSV) --out-json $(POLICY_AB_OUT_JSON)

rl_exploration_campaign:
	$(PYTHON) auto_tuner/rl_daemon.py --timesteps $(RL_TIMESTEPS_LONG) --workload zoo_v1 --train-shape-split train --eval-workloads zoo_v1 --eval-shape-split test --eval-episodes 20 --campaign-id $(RL_CAMPAIGN_ID) --campaign-stage $(RL_CAMPAIGN_STAGE) --targeted-campaign-config $(TARGETED_CAMPAIGN_JSON) --targeted-sample-prob $(RL_TARGETED_SAMPLE_PROB) --checkpoint-every-steps $(RL_CHECKPOINT_EVERY_STEPS) --checkpoint-keep-last $(RL_CHECKPOINT_KEEP_LAST) --export-trace-parquet-dir $(TRACE_PARQUET_DIR)

export_rl_space_contract:
	$(PYTHON) auto_tuner/export_rl_space_contract.py --workspace $(CURDIR) --output-json $(CANONICAL_RL_SPACE_JSON)

policy_train_canonical:
	$(PYTHON) auto_tuner/trace_to_parquet.py --workspace $(CURDIR) --all-traces --valid-only --output-dir $(CANONICAL_TRACE_PARQUET_DIR) --metadata-json $(CANONICAL_TRACE_PARQUET_META) --partition-cols workload_tag,sparsity_bucket,run_date_utc
	$(MAKE) export_rl_space_contract PYTHON=$(PYTHON) CANONICAL_RL_SPACE_JSON=$(CANONICAL_RL_SPACE_JSON)
	$(PYTHON) auto_tuner/train_logistic_baseline.py --workspace $(CURDIR) --input-parquet $(CANONICAL_TRACE_PARQUET_DIR) --output-json $(CANONICAL_BEST_PARAMS_JSON) --selection-policy topk --top-k 5 --within-pct 0.05 --regret-weight-alpha 0.0 --min-sample-weight 0.05 --confidence-fallback-threshold 0.3 --seed 17 --eval-fraction 0.2 --epochs 400 --learning-rate 0.15 --l2 1e-4 --artifact-manifest-out $(CANONICAL_POLICY_MANIFEST_JSON) --dataset-meta-json $(CANONICAL_TRACE_PARQUET_META) --rl-space-contract-json $(CANONICAL_RL_SPACE_JSON) --canonical-contract-json $(CANONICAL_CONTRACT_JSON) --policy-id logistic_topk_v1

# Generic benchmark compile/run path (independent from inference_bare.c)
BENCH_NAME ?= bmark
BENCH_SRC  ?= $(CURDIR)/benchmarks/bmark/fib.c
BENCH_LD   ?= $(CURDIR)/benchmarks/bmark/common.ld
BENCH_START ?= $(CURDIR)/benchmarks/bmark/start.s
MAX_CYCLES ?= 500000000
BENCH_ELF  := $(BUILD_DIR)/$(BENCH_NAME).elf
BENCH_BIN  := $(BUILD_DIR)/$(BENCH_NAME).bin
BENCH_HEX  := $(BUILD_DIR)/$(BENCH_NAME).hex
BENCH_DUMP := $(BUILD_DIR)/$(BENCH_NAME).dump

$(BENCH_HEX): $(BENCH_SRC) $(BENCH_START) $(BENCH_LD) | $(BUILD_DIR)
	$(RISCV)-gcc $(GCC_OPTS) $(CFLAGS_EXTRA) -I$(FW_INC) -T $(BENCH_LD) $(BENCH_START) $(BENCH_SRC) -o $(BENCH_ELF)
	$(RISCV)-objdump -D -Mnumeric $(BENCH_ELF) > $(BENCH_DUMP)
	$(RISCV)-objcopy $(BENCH_ELF) -O binary $(BENCH_BIN)
	$(PYTHON) bin2hex.py -w 128 $(BENCH_BIN) $(BENCH_HEX)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(VERILATOR_CFG_STAMP): FORCE | $(BUILD_DIR)
	@tmp="$@.tmp"; \
	printf "%s\n" "VFLAGS=$(VFLAGS)" > "$$tmp"; \
	printf "%s\n" "RTL_SRCS=$(RTL_SRCS)" >> "$$tmp"; \
	if [ -f "$@" ] && cmp -s "$$tmp" "$@"; then \
		rm -f "$$tmp"; \
	else \
		mv "$$tmp" "$@"; \
	fi

$(INFERENCE_CFG_STAMP): FORCE | $(BUILD_DIR)
	@tmp="$@.tmp"; \
	printf "%s\n" "INFERENCE_SRC=$(INFERENCE_SRC)" > "$$tmp"; \
	printf "%s\n" "INFERENCE_CFLAGS=$(INFERENCE_CFLAGS)" >> "$$tmp"; \
	printf "%s\n" "GCC_OPTS=$(GCC_OPTS)" >> "$$tmp"; \
	for f in $(INFERENCE_DEPS); do \
		if [ -f "$$f" ]; then \
			h=$$(shasum -a 256 "$$f" | awk '{print $$1}'); \
			printf "%s\n" "DEP_HASH $$f $$h" >> "$$tmp"; \
		else \
			printf "%s\n" "DEP_MISSING $$f" >> "$$tmp"; \
		fi; \
	done; \
	if [ -f "$@" ] && cmp -s "$$tmp" "$@"; then \
		rm -f "$$tmp"; \
	else \
		mv "$$tmp" "$@"; \
	fi

$(BUILD_DIR)/inference.hex: $(INFERENCE_CFG_STAMP)

FORCE:

export_mnist_model:
	cd workloads/mnist && $(PYTHON) mnist_mlp_export.py

run_bench: obj_dir/Vriscv_top $(BENCH_HEX)
	./obj_dir/Vriscv_top +loadmem=$(BENCH_HEX) +max-cycles=$(MAX_CYCLES) $(EXTRA_FLAGS)

run_ilp_single:
	$(MAKE) clean
	$(MAKE) run ENABLE_DUAL_ISSUE=0 CFLAGS_EXTRA="-DILP_MICROBENCH"

run_ilp_dual:
	$(MAKE) clean
	$(MAKE) run ENABLE_DUAL_ISSUE=1 CFLAGS_EXTRA="-DILP_MICROBENCH"

clean:
	rm -rf obj_dir $(BUILD_DIR)

clean_bench:
	rm -rf $(BUILD_DIR)

print-v:; @$(VERILATOR) --version
