# Accelera Framework

Accelera is an HW/SW co-design stack for bare-metal ML acceleration. It combines:
- cycle-accurate RTL simulation (`Verilator`) for ground-truth performance,
- compiler-side tensor packing/export,
- an RL-based autotuner for schedule/dataflow selection,
- optional surrogate-model acceleration for sample efficiency.

## Repository Layout

- `rtl/`: RISC-V core + accelerator RTL.
- `firmware/`: bare-metal inference runtime (`inference_generic.c`).
- `compiler/`: model export/packing (`export_model.py`).
- `auto_tuner/`: RL environment, daemon, ledger, workload bank, evaluation, surrogate tools.
- `workloads/`: workload/model sources (MNIST and others).

## Architecture Overview

Accelera is designed as a reliability-first closed loop, not a loose set of scripts.

### Stack Layers

- Hardware plane:
  - `rtl/accel/systolic/MatmulAcceleratorSystolic.sv` implements MMIO-programmable matmul with dense mode (`hw_mode=0`) and structured sparse mode1 (`hw_mode=1`).
  - Read-only capability discovery registers expose interface version and supported features (`IF_VERSION` / `CAPABILITIES`).
- Firmware/runtime plane:
  - `firmware/inference_generic.c` is the model-driven runtime that reads per-layer mode/tiling from model headers and dispatches accelerator calls.
  - Runtime has explicit capability fallback for sparse mode selection when hardware support is unavailable.
- Compiler/export plane:
  - `compiler/export_model.py` selects per-layer schedule/mode, applies layout packing, and emits `model_blob.h` (+ optional metadata sidecar).
  - Sparse mode uses structured 2:4 packing and self-check guards before export.
- Optimization plane:
  - `auto_tuner/rl_daemon.py` explores schedule/dataflow choices and writes overrides/ledger entries.
  - `auto_tuner/workload_bank.py` and benchmark-zoo manifests provide diverse workload families (DNN/CNN/RNN-style inference and training proxies).
- Governance plane:
  - `make verify`, unit tests, regression gate scripts, and CI workflows enforce correctness and controlled performance regressions.

### End-to-End Dataflow

1. A model/workload is exported to firmware artifacts (`model_blob.h`, optional `model_blob.meta.json`).
2. Firmware reads layer headers and dispatches dense/sparse paths based on exported mode and runtime capability checks.
3. RTL executes cycle-accurate commands and emits performance counters/trace-visible mode behavior.
4. RL/eval tooling consumes traces and manifests to update best-known policies and benchmark results.
5. CI/release gates enforce correctness equivalence and regression thresholds before merge.

## Core Concepts

### 1) Workload-Conditioned RL
The RL environment samples from workload families and trains policy decisions conditioned on workload structure.

Primary families now include:
- inference-oriented: `gemm`, `sparse_mlp`, `dnn_infer`, `convolution`, `attention`, `rnn_infer`
- training-oriented proxies: `dnn_train_fwd`, `dnn_train_bwd`, `cnn_train_fwd`, `cnn_train_bwd`, `rnn_train_bptt`

Selector aliases are supported:
- `inference`, `training`, `vision`, `sequence`, `dense`, `legacy`, `zoo_v1`

Action knobs:
- `tile_m`, `tile_n`
- `burst_size`
- `prefetch_depth`
- `tile_b`
- `hardware_dataflow_mode`

### 2) Compiler + Ledger Loop
- The daemon writes `auto_tuner/rl_override.json`.
- Export path (`compiler/export_model.py`) consumes overrides and/or `auto_tuner/compiler/bkm_ledger.json`.
- Best-known configs are keyed by shape + activation + workload/sparsity buckets.
- Sparsity bucket contract is fixed-width deciles: `bucket = floor(sparsity_pct / 10)` clamped to `0..10`.
- Key example: `128x256x256_act1_dnn_infer_sp3` means ~30-39% sparsity region.

### 3) Replay Dataset + Surrogate
- Every env step logs a structured JSONL row (`data/traces/<run_id>.jsonl` by default).
- Each daemon launch writes a run manifest (`data/runs/<run_id>.json`) for status + summary.
- Logged fields include workload features, hardware config, schedule decisions, constraints, and outcomes.
- Trace rows include scratchpad estimates (`scratch_required_bytes`, `scratch_limit_bytes`, `scratch_util_pct`) and whether a sampled action was feasibility-pruned (`action_pruned`).
- Optional surrogate predicts cycles cheaply between periodic full simulations.

## Prerequisites

### Local
- `python3`
- `verilator`
- `riscv64-unknown-elf-gcc` toolchain
- Python deps: `gymnasium`, `stable-baselines3`, `numpy`
- Optional large-dataset deps: `pyarrow` (for Parquet export/import)
- Optional Apple Silicon acceleration: `mlx` (`ACCELERA_ARRAY_BACKEND=mlx`)

### Docker (recommended)
- Uses `docker/rl-daemon/Dockerfile`
- Includes Verilator + RISC-V GCC + Python deps

## Basic Flow

### 1) Export a reference workload (MNIST)
```bash
python3 workloads/mnist/mnist_mlp_export.py
```

### 2) Run cycle-accurate inference
```bash
make clean && make run \
  INFERENCE_SRC=inference_generic.c \
  ENABLE_DUAL_ISSUE=1 \
  GENERIC_CFLAGS='-O3 -fno-strict-aliasing' \
  EXTRA_VFLAGS='-DUSE_SYSTOLIC_ACCEL'
```

### 3) Start RL autotuning
```bash
python3 auto_tuner/rl_daemon.py \
  --timesteps 10000000 \
  --workload all
```

### 4) Run canonical reliability checks
```bash
make verify
```
`make verify` is the canonical pre-merge gate and runs:
- Python unit tests
- exporter + blob integrity round-trip
- dense systolic smoke run
- sparse mode-1 smoke run

## API Orchestrator (A->B->C)

Use the deterministic multi-call orchestration loop under `agents/`:

```bash
export ANTHROPIC_API_KEY="<your_key>"
python agents/orchestrator.py --task "add sparsity histogram to A/B output"
```

What it does:
- Agent A: plan
- Agent B: critique/revise
- Agent C: implement with search/replace edits
- Agent B: review implementation
- Orchestrator validation: `py_compile`, `unittest discover`, optional `make verify`

Run artifacts are written to `agents/runs/<run_id>/` with full request/response traces plus test logs and state for resume.

## RL Daemon Key Options

```bash
python3 auto_tuner/rl_daemon.py --help
```

Most used:
- `--workload`: training families (`all` or CSV list)
- `--train-shape-split`: `all|train|test`
- `--eval-workloads`: held-out families for post-train eval
- `--eval-shape-split`: `all|train|test`
- `--eval-episodes`: number of eval episodes
- `--disable-workload-features`: ablation (remove workload-structure features)
- `--disable-workload-aware-lookup`: ablation (generic ledger fallback)
- `--trace-path`: trace JSONL path (default `data/traces/<run_id>.jsonl`)
- `--run-id`: explicit run id for resumable campaigns
- `--checkpoint-every-steps`: periodic checkpoint save cadence for long runs
- `--checkpoint-keep-last`: number of periodic checkpoints to retain
- `--campaign-id` / `--campaign-stage`: provenance tags written into traces/manifests
- `--targeted-campaign-config`: campaign JSON generated from regret/coverage analysis
- `--targeted-sample-prob`: probability of sampling targeted regions each episode
- `--enable-tile-b4`: staged knob expansion to include `tile_b=4` in the action space
- `--export-trace-parquet-dir`: optional post-run Parquet export directory
- `--force-hw-mode`: force exporter mode for controlled A/B runs (`-1` off, `0` dense, `1` sparse mode1)
- `--allow-eval-train-overlap`: opt-in override when you intentionally want train/eval on overlapping selector+split (default is leakage-safe reject)
- `ACCELERA_EXTENDED_WORKLOAD_FEATURES=1`: opt-in extra RL observation features (`sparsity_bucket`, `mode1_candidate`) for new training runs; keep unset to remain checkpoint-compatible with older 14-D policies.

## Canonical RL->Policy Contract

Canonical contract file:
- `auto_tuner/canonical_policy_contract_v1.json`

Single canonical definitions:
- Canonical dataset source: `data/canonical/trace_dataset_v1` (Parquet, built from all valid JSONL traces).
- Canonical logistic config:
  - `selection_policy=topk`
  - `top_k=5`
  - `regret_weight_alpha=0.0`
  - `min_sample_weight=0.05`
  - `confidence_fallback_threshold=0.3`
  - `seed=17`, `eval_fraction=0.2`
  - `epochs=400`, `learning_rate=0.15`, `l2=1e-4`
- Canonical training command:

```bash
make policy_train_canonical PYTHON=.venv-mcp/bin/python
```

Canonical outputs:
- `data/canonical/policy/best_params_v1.json`
- `data/canonical/policy/best_params_v1.manifest.json`
- `data/canonical/policy/rl_exploration_space_v1.json`

Policy artifact manifest includes:
- exact training config used,
- dataset metadata provenance,
- RL exploration-space contract snapshot,
- output checksums/sizes and entry counts.

Source-of-truth RL exploration space:
- Defined in code by `SystolicEnv.exploration_space_contract(...)`.
- Export command:

```bash
make export_rl_space_contract PYTHON=.venv-mcp/bin/python
```

Experiment isolation rule:
- Keep non-canonical experiments under custom paths (for example `data/experiments/...`).
- Do not overwrite canonical outputs unless intentionally promoting a new canonical policy version.

## Scalable Dataset Campaigns (JSONL + Parquet)

This section is for exploration/experimentation workflows. Canonical policy promotion is handled by `make policy_train_canonical`.

### 1) Build targeted campaign input from regret + coverage

```bash
make build_targeted_campaign
```

This emits `data/campaigns/targeted_campaign_v1.json` with weighted targets derived from:
- high-regret keys
- low-coverage keys
- policy-disagreement keys
- underrepresented workload/sparsity buckets

### 2) Run long exploration safely (checkpoint-friendly + provenance)

```bash
make rl_exploration_campaign
```

Equivalent manual command:

```bash
python3 auto_tuner/rl_daemon.py \
  --timesteps 500000 \
  --workload zoo_v1 \
  --train-shape-split train \
  --eval-workloads zoo_v1 \
  --eval-shape-split test \
  --eval-episodes 20 \
  --campaign-id zoo_long \
  --campaign-stage tranche_next \
  --targeted-campaign-config data/campaigns/targeted_campaign_v1.json \
  --targeted-sample-prob 0.6 \
  --checkpoint-every-steps 10000 \
  --checkpoint-keep-last 5 \
  --export-trace-parquet-dir data/parquet/trace_dataset_v1
```

### 3) Convert historical traces into Parquet dataset

```bash
make trace_to_parquet
```

Manual:

```bash
python3 auto_tuner/trace_to_parquet.py \
  --workspace . \
  --all-traces \
  --valid-only \
  --output-dir data/parquet/trace_dataset_v1 \
  --partition-cols workload_tag,sparsity_bucket,run_date_utc
```

`trace_to_parquet.py` keeps JSONL as source-of-truth and writes a partitioned Parquet dataset + metadata sidecar (`_dataset_meta.json`).
If `pyarrow` is installed only in a project venv, run make targets as:

```bash
make trace_to_parquet PYTHON=.venv-mcp/bin/python
```

### 4) Coverage + storage summary

```bash
make dataset_summary
```

This writes `data/audits/trace_dataset_summary.json` with:
- row/valid/invalid counts
- workload + shape + sparsity coverage
- knob diversity
- key-level coverage stats
- size comparison (CSV vs Parquet when both are provided)

### 5) Retrain policy from Parquet-backed dataset

```bash
make train_logistic_baseline_parquet BEST_PARAMS_SELECTION_POLICY=topk BEST_PARAMS_TOP_K=5
```

The canonical topk workflow remains unchanged; this target only switches input source from CSV to Parquet.

### 6) Re-run realized-regret evaluation after data growth

```bash
make policy_ab_regret_report POLICY_TRACE_CSV=data/trace_dataset_all.csv
```

Use a baseline run on a smaller dataset (for example latest-trace CSV) and compare against full-history/all-traces output to measure whether expanded exploration reduced realized regret.

## Deterministic ML Baseline Pipeline

Use the normalized trace schema as the single source for CSV extraction and baseline model training:

```bash
make trace_to_csv
make train_logistic_baseline
make materialize_best_params
```

This is a flexible experiment path. For canonical promotion, use `make policy_train_canonical`.

What this does:
- `trace_to_csv`: converts JSONL trace rows to a stable CSV (`data/trace_dataset_latest.csv`) using `normalize_trace_row(...)`.
- `train_logistic_baseline`: trains deterministic one-vs-rest logistic baselines for:
  - `executed_hardware_dataflow_mode`
  - `tile_m`, `tile_n`, `burst_size`, `prefetch_depth`, `tile_b`
- writes reproducible `data/best_params_v1.json` with BKM-compatible keys (`MxNxK_actA_<workload>_spB`).
- optional materialization merges emitted entries into a candidate ledger (`auto_tuner/compiler/bkm_ledger.ml_candidate.json`).

Selection-policy options (`auto_tuner/train_logistic_baseline.py`):
- `--selection-policy best1` (default): one best-cycle row per key (backward-compatible behavior)
- `--selection-policy topk --top-k 5`: train on top-k rows per key
- `--selection-policy within_pct --within-pct 0.05`: train on rows within 5% of the key-best cycle
- `--regret-weight-alpha`: regret-aware weighting strength for selected training rows (`0` disables)
- `--min-sample-weight`: lower bound for weighted rows
- `--confidence-fallback-threshold`: if prediction confidence is below threshold, fallback to oracle best-per-key label during materialization

Recommended training setup:
- keep final materialization contract as best-1 per key (unchanged)
- use richer training supervision via `topk` or `within_pct`
- split discipline remains key-level to avoid train/eval leakage across rows from the same key

Manual equivalent:

```bash
python3 auto_tuner/trace_to_csv.py --workspace . --output-csv data/trace_dataset_latest.csv --valid-only
python3 auto_tuner/train_logistic_baseline.py \
  --workspace . \
  --input-csv data/trace_dataset_latest.csv \
  --output-json data/best_params_v1.json \
  --selection-policy topk \
  --top-k 5 \
  --regret-weight-alpha 0.0 \
  --confidence-fallback-threshold 0.3 \
  --materialize-ledger-out auto_tuner/compiler/bkm_ledger.ml_candidate.json
```

Policy A/B regret report (realized cycles on full trace history):

```bash
make policy_ab_regret_report
```

Outputs:
- `data/audits/policy_ab_regret_34keys.csv`: per-key table with oracle best cycles, topk cycles, within-pct cycles, regret %, winner, and coverage counts.
- `data/audits/policy_ab_regret_summary.json`: aggregate metrics (mean/median/p90 regret, within-1/3/5% counts, disagreement buckets by workload+sparsity).

## Benchmark Zoo (v1)

Use the versioned benchmark-zoo runner for broader DNN/CNN/RNN coverage:

```bash
python3 auto_tuner/run_benchmark_zoo.py \
  --manifest auto_tuner/benchmark_zoo_v1.json \
  --timesteps 0
```

Useful options:
- `--list-cases`: print manifest cases and exit
- `--eval-episodes`: override per-case episode count
- `--repeats`: override per-case repeats
- `--checkpoint-path`: evaluate a specific RL checkpoint

## Benchmark Zoo A/B Harness

Use forced-mode A/B runs to compare dense vs sparse on the same zoo cases:

```bash
ACCELERA_ENABLE_SPARSE_MODE1=1 python3 auto_tuner/run_benchmark_zoo_ab.py \
  --manifest auto_tuner/benchmark_zoo_v1.json \
  --dense-mode 0 \
  --sparse-mode 1 \
  --timesteps 0
```

The harness uses the same per-case seed for dense and sparse runs so each A/B pair evaluates matched workload samples and shape rows.

Outputs:
- JSON summary with per-case averages (`avg_sparse_over_dense_cycle_ratio`, `avg_accuracy_delta`, `avg_pe_util_delta`)
- JSON shape-match coverage guardrails (`matched_shape_coverage`, `shape_mismatch_case_count`, `shape_mismatch_case_ids`)
- Per-case diagnostic dump (`cases[].diagnostic_rows`) with layer sizes, tile dimensions, burst/prefetch/tile_b means, PE-util, mode-share evidence, fallback-rate evidence, and cycle deltas.
- Data-driven sparse-unsuitable annotations:
  - row-level: `low_sparsity_flag`, `sparse_regressed_flag`, `sparse_unsuitable_row`
  - case-level: `sparse_unsuitable_case`, `low_sparsity_regression_consistency`, and threshold config echoed in summary.
- CSV with per-shape side-by-side metrics (`dense_cycles_mean`, `sparse_cycles_mean`, cycle ratio, valid-ratio delta, PE-util delta)

`accuracy_delta` in this harness is a correctness-validity proxy (`sparse_valid_ratio - dense_valid_ratio`) for zoo synthetic kernels.
By default, any dense/sparse shape mismatch is a harness failure (`status=failed_shape_mismatch`); use `--allow-shape-mismatch` only for debugging.
Sparse-unsuitable classification defaults to: low sparsity `<=15%` + ratio `>1.0` + consistency `>=0.75` across low-sparsity paired shapes (tunable via `--unsuitable-*` flags).

## Structured 2:4 Export Contract

### Pruning/packing behavior

- Mode1 export is not a passive "check only" path. Exporter performs deterministic top-2 selection per 4-K group over each 4x4 block (sum of magnitudes across the 4 rows), then packs only those values.
- Export metadata records:
  - original sparsity (`sparsity_pct`)
  - pre-prune 2:4 compatibility ratio (`structured_2_4_compat_ratio`)
  - prune deltas when mode1 is selected (`mode1_prune_stats`)
- `ACCELERA_MODE1_APPLY_PRUNE=1` (default) keeps pruning explicit for metadata/reporting and deterministic reproducibility.

### Packed word format

- One packed `uint32` is emitted per row per 4-K group:
  - bits `[7:0]`: `val0` (int8)
  - bits `[15:8]`: `val1` (int8)
  - bits `[17:16]`: `idx0` (2-bit K offset)
  - bits `[19:18]`: `idx1` (2-bit K offset)
  - remaining upper bits reserved/zero
- In hardware (`S_COMPUTE`), mode1 reads index metadata from weight chunk bits `[23:16]` and selects two K positions through the mux path while reusing the existing systolic datapath.

## Dense vs Sparse Suitability (Reliability-First)

Do not choose sparse mode from cycle numbers alone. A workload is "sparse-suitable" only after all gates below pass.

### Gate 1: Capability Gate

- Hardware must advertise sparse mode support through accelerator capabilities.
- Runtime checks this via `accel_supports_hw_mode(1)` before dispatching sparse mode.
- If capability is absent, runtime falls back to dense mode for compatibility.

### Gate 2: Structural/Format Gate

- Sparse mode1 expects structured 2:4-compatible packed weights, not generic dense layout.
- Export-time guards in `compiler/export_model.py` validate shape/tile/sparsity constraints and reject unsafe sparse requests.
- Forcing dense execution on sparse-encoded payloads is not a valid comparison and can fail correctness.

### Gate 3: Correctness Gate (Required)

- Dense vs sparse A/B must preserve correctness (unit checks and end-to-end accuracy).
- If sparse mode improves cycles but reduces correctness, dense remains the correct mode.
- This is the primary blocker gate for production mode promotion.

### Gate 4: Performance Gate (After Correctness)

- Compare cycle/perf counters only after correctness is green.
- Require a minimum gain threshold and stable valid-run ratio before selecting sparse.
- Track per-workload and per-layer behavior in run manifests for regression gating.

### Practical Interpretation of Cycle Numbers

- `Total cycles` confirms execution cost for a chosen mode, not mathematical validity.
- Reliable sparse activation evidence requires:
  - exported layer mode indicates sparse,
  - trace `executed_hardware_dataflow_mode` matches exporter-selected mode (post-export reconciliation),
  - runtime/RTL logs show `hw_mode=1`,
  - workload correctness remains within acceptance criteria.
- "Sparse on dense-like workload" can still produce lower cycles and yet fail correctness.
- "Dense on sparse-encoded payload" can fail because data layout no longer matches expected decode path.

### Recommended Validation Matrix

For each candidate workload/layer family:

1. Dense baseline: correctness + cycles.
2. Sparse candidate: correctness + cycles + mode evidence (`hw_mode=1`).
3. Capability fallback scenario: confirm expected dense fallback behavior.
4. Regression gate update: enforce no correctness regression and bounded cycle regressions.

Canonical quick checks:

```bash
# Required reliability smoke checks
make verify_dense_smoke
make verify_sparse_mode1

# Full pre-merge reliability gate
make verify
```

## Docker Workflow

Build image:
```bash
docker build -t accelera-rl-daemon:latest -f docker/rl-daemon/Dockerfile .
```

Run training:
```bash
docker run --rm -it \
  -v "$(pwd)":/workspace \
  -w /workspace \
  accelera-rl-daemon:latest \
  python -u auto_tuner/rl_daemon.py --timesteps 10000000 --workload all
```

Detached:
```bash
docker run -d --name accelera-rl \
  -v "$(pwd)":/workspace \
  -w /workspace \
  accelera-rl-daemon:latest \
  python -u auto_tuner/rl_daemon.py --timesteps 10000000 --workload all
```

Monitor:
```bash
docker logs -f accelera-rl
```

## Local Dashboard (Frontend MVP)

Run the dashboard:
```bash
python3 auto_tuner/dashboard.py --host 127.0.0.1 --port 8787
```

Then open:
```text
http://127.0.0.1:8787
```

MVP views:
- Runs page: command/config, status, row counts, best/avg metrics, policy saved.
- Run detail: summary cards, config panel, candidate table, valid-cycles chart.
- Constraint diagnostics: reject counts and reasons (for example `scratchpad_capacity`).

## Evaluation Matrix / Ablations

`auto_tuner/eval_matrix.py` runs train/eval scenario grids and writes CSV tables.

Default matrix:
```bash
python3 auto_tuner/eval_matrix.py
```

Full matrix:
```bash
python3 auto_tuner/eval_matrix.py \
  --scenarios all \
  --timesteps 200 \
  --eval-episodes 20 \
  --n-steps 8 \
  --batch-size 8 \
  --output-csv data/rl_eval_matrix_full.csv
```

CSV fields include:
- `valid_runs`
- `avg_cycles`
- `avg_reward`
- `workload_family`
- `shape_signature`
- `mode0_share_mean` / `mode1_share_mean`
- `mode0_valid_cycles_mean` / `mode1_valid_cycles_mean`
- `sparse_over_dense_cycle_ratio_mean` (when both modes were exercised)
- `pe_util_mean`
- `dense_equiv_macs_per_cycle_mean`
- `sparsity_pct_mean` / `sparsity_bucket`
- scenario/ablation metadata

## Surrogate-Assisted Tuning

### 1) Enable surrogate in RL daemon
```bash
python3 auto_tuner/rl_daemon.py \
  --timesteps 100000 \
  --workload all \
  --enable-surrogate \
  --surrogate-verify-every 10 \
  --surrogate-retrain-every 64 \
  --surrogate-min-records 200
```

### 2) Train surrogate offline from traces
```bash
python3 auto_tuner/train_surrogate.py \
  --trace-path data/traces/<run_id>.jsonl \
  --model-path data/surrogate_model.pt
```

### 3) Surrogate top-K candidate search
```bash
python3 auto_tuner/surrogate_topk_search.py \
  --workload all \
  --rounds 20 \
  --candidates 64 \
  --top-k 8
```

## Generated Artifacts

- `auto_tuner/compiler/bkm_ledger.json`: best-known tuning configs.
- `auto_tuner/ppo_systolic_agent_latest.zip`: latest policy checkpoint.
- `data/traces/<run_id>.jsonl`: per-run replay dataset for surrogate and analysis.
- `data/runs/*.json`: per-run manifest (status, config, summary, eval rows).
- `data/surrogate_model.pt` (+ `.meta.json`): surrogate model.
- `data/rl_eval_matrix_*.csv`: evaluation matrix outputs.
- `data/benchmark_zoo_*.json`: benchmark-zoo case summaries.
- `data/benchmark_zoo_*.csv`: benchmark-zoo per-shape metrics.
- `firmware/include/model_blob.meta.json`: optional export metadata (selected mode provenance and required accelerator capabilities).

## MCP Integration

An MCP server is available at `mcp/accelera_mcp_server.py` to expose core flows as tools for AI clients.

Quick start:

```bash
python3 -m venv .venv-mcp
.venv-mcp/bin/python -m pip install -r mcp/requirements.txt
.venv-mcp/bin/python mcp/accelera_mcp_server.py
```

For full RL/eval support in the same venv:

```bash
.venv-mcp/bin/python -m pip install -r mcp/requirements.txt
```

The MCP server includes sync and async tools for inference/tuning/eval plus run/trace analysis (`summarize_trace`, `compare_runs`), and supports interpreter overrides via `ACCELERA_PYTHON` / `ACCELERA_RL_PYTHON`.
You can run the canonical merge-gating path with:

```bash
make mcp_regression_gate PYTHON=.venv-mcp/bin/python
```

The canonical target runs:
- `make verify`
- `mcp/validate_workflow.py` with sparse mode enabled (`ACCELERA_ENABLE_SPARSE_MODE1=1`)
- `mcp/regression_gate.py` against `.github/mcp_regression_baseline.json`

You can also run end-to-end validation directly with:

```bash
.venv-mcp/bin/python mcp/validate_workflow.py
```

To check a candidate run against a pinned baseline:

```bash
.venv-mcp/bin/python mcp/regression_gate.py \
  --report data/mcp_validation_report_20260316_162021.json \
  --baseline-config .github/mcp_regression_baseline.json
```

The baseline config is versioned (`schema_version`) and supports
per-workload regression thresholds (`per_workload_thresholds`).

A GitHub Actions regression gate is defined at `.github/workflows/mcp-regression.yml`.
Manual baseline promotion (PR-based, manual-only) is defined at `.github/workflows/mcp-baseline-promotion.yml`.
Branch protection can be enabled later manually to require this check; until then, `make mcp_regression_gate` remains the canonical local/CI gate path.

See `mcp/README.md` for full tool list and client config example.
Project update history is tracked in `IMPROVEMENT_LOG.md`.

## Recent Optimizations
### Structured Sparsity & Pipelining (11.5% Speedup)
Legacy sparse mode behavior incurred significant control overhead and poor datapath utilization. Firmware-side synchronous issue/wait sequencing also blocked useful overlap.

We implemented **2:4 Block Structured Sparsity** alongside **Dispatch-Before-Read MMIO pipelining**. 
- The hardware multiplexer logic now selects 2 non-zeros per 4 K-steps natively within the same `S_COMPUTE` cycle, essentially doubling the effective sparse datapath bandwidth.
- The `inference_generic.c` runtime overlaps issuing `accel_issue_next_x` to the accelerator with polling the MMIO results from the *previous* spatial block.

**Combined result**: Running the full 100-image MNIST generic inference was reduced from **18.15M cycles** to **16.05M cycles**, saving ~2.1 million cycles.

Attribution guidance:
- Treat this 11.5% as a combined system result (datapath + runtime overlap).
- Use benchmark-zoo outputs (`mode0_*`, `mode1_*`, `sparse_over_dense_cycle_ratio_mean`) for datapath-specific attribution across broader workloads.
- Do not promote sparse mode from cycle deltas alone; correctness gate remains primary.

## Notes

- `tile_b` defaults to `{1,2}`; staged expansion to include `4` is opt-in via `--enable-tile-b4`.
- Workload sampling applies a feasibility prefilter, and infeasible sampled actions are pruned to valid neighbors before execution.
- Constraint violations (e.g., scratchpad overflow) are hard-failed and logged.
- Surrogate mode is optional; full cycle-accurate simulation remains the ground truth.
- Array backend defaults to NumPy. Set `ACCELERA_ARRAY_BACKEND=mlx` to enable MLX for supported Python-side tensor paths (falls back to NumPy if MLX is unavailable).
- Tests can force deterministic backend selection with `ACCELERA_ARRAY_BACKEND_FORCE=numpy|mlx`.
- Mode1 explicit prune reporting can be controlled with `ACCELERA_MODE1_APPLY_PRUNE=0|1` (default `1`).
- Accelerator MMIO/API contract and compatibility policy are documented in `docs/ACCELERATOR_INTERFACE.md`.
- Additive capability discovery registers are available at `0x88` (`IF_VERSION`) and `0x8C` (`CAPABILITIES`).
- Benchmark-zoo training entries are training-phase operator proxies (GEMM/conv/RNN-style kernels), not full optimizer/training-loop execution.
