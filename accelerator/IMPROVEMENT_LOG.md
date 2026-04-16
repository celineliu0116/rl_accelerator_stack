# Improvement Log

## 2026-03-26

### Canonical RL->Policy Contract Lockdown
- Added explicit canonical contract file: `auto_tuner/canonical_policy_contract_v1.json`.
- Added single canonical command for policy promotion:
  - `make policy_train_canonical PYTHON=.venv-mcp/bin/python`
- Canonical outputs are now fixed to:
  - `data/canonical/trace_dataset_v1` (dataset source of truth),
  - `data/canonical/policy/best_params_v1.json`,
  - `data/canonical/policy/best_params_v1.manifest.json`,
  - `data/canonical/policy/rl_exploration_space_v1.json`.
- Added artifact-manifest emission to `auto_tuner/train_logistic_baseline.py`:
  - embeds training config, dataset metadata, RL-space contract, and output checksums/sizes.
- Added source-of-truth exploration-space exporter:
  - `auto_tuner/export_rl_space_contract.py`
  - backed by `SystolicEnv.exploration_space_contract(...)`.
- Updated `.gitignore` to keep local generated candidate artifacts out of commits by default.
- Documentation now distinguishes canonical promotion from experiment workflows.

### RL Data Scaling Tranche (Parquet + Campaign Targeting + Staged Knob Expansion)
- Implemented scalable dataset and campaign infrastructure while keeping the canonical topk policy flow intact.

#### Data Infrastructure
- Added partitioned Parquet export on top of existing JSONL traces:
  - `auto_tuner/trace_to_parquet.py`
  - preserves append-friendly JSONL source flow; Parquet is additive.
  - writes schema/lineage metadata sidecar (`_dataset_meta.json`) with trace schema versions, source paths, row counts, and partition config.
- Extended CSV extraction for large aggregation workflows:
  - `auto_tuner/trace_to_csv.py` now supports `--all-traces` / `--trace-glob`.
  - adds provenance columns (`source_trace_size_bytes`, `source_trace_mtime_epoch_sec`, `run_date_utc`) while preserving existing columns.
- Added dataset analytics:
  - `auto_tuner/dataset_summary.py` reports row counts, valid/invalid rates, workload/shape/sparsity distributions, knob diversity, key coverage, and storage comparisons.

#### Exploration Scaling + Provenance
- Extended RL daemon for safer long campaigns:
  - periodic checkpointing (`--checkpoint-every-steps`, `--checkpoint-keep-last`)
  - explicit run id (`--run-id`)
  - campaign provenance tags (`--campaign-id`, `--campaign-stage`)
  - optional post-run Parquet export (`--export-trace-parquet-dir`, `--parquet-partition-cols`).
- Added campaign-aware trace fields in normalization and env emission:
  - `run_mode`, `campaign_id`, `campaign_stage`,
  - `campaign_targeted`, `campaign_target_reason`, `campaign_target_weight`, `target_key`.

#### Targeted Exploration
- Added `auto_tuner/build_targeted_campaign.py`:
  - builds weighted target config from regret + coverage artifacts.
  - target sources include high-regret keys, low-coverage keys, disagreement keys, and underrepresented buckets.
- Added RL daemon targeted sampling support:
  - `--targeted-campaign-config`
  - `--targeted-sample-prob`
  - workload reset path now supports weighted targeted pools and records target provenance.

#### Controlled Knob Expansion
- Added staged action-space expansion for `tile_b=4` behind explicit opt-in:
  - `--enable-tile-b4` in `rl_daemon.py`
  - `SystolicEnv` action space now expands to `{1,2,4}` only when enabled.
  - default behavior remains `{1,2}` for backward compatibility.
- Added checkpoint namespace separation to avoid policy incompatibility across staged action spaces:
  - default checkpoint becomes `ppo_systolic_agent_tileb4_latest` when `--enable-tile-b4` is active.

#### Training/Data Compatibility
- Added Parquet input support to supervised trainer:
  - `auto_tuner/train_logistic_baseline.py` now accepts `--input-parquet`.
  - output payload includes `input_kind` and parquet provenance.
- Verified parity path:
  - CSV-backed and Parquet-backed topk/within runs produce matching policy A/B summary metrics on full-history trace scoring.

#### Make/Workflow Integration
- Added make targets:
  - `trace_to_parquet`
  - `dataset_summary`
  - `build_targeted_campaign`
  - `train_logistic_baseline_parquet`
  - `rl_exploration_campaign`

#### Validation + Measured Outcomes
- Full-history dataset materialization:
  - `84,245` valid rows aggregated from `143` trace sources.
- Parquet storage footprint:
  - CSV (`data/trace_dataset_all.csv`): `31,919,403` bytes
  - Parquet (`data/parquet/trace_dataset_v1`): `5,312,617` bytes
  - size reduction: `~83.36%`.
- Targeted campaign generation:
  - `20` targets emitted (`12` high-regret, `5` low-coverage, `12` disagreement, overlap allowed by design).
- Regret impact from larger exploration data:
  - latest-only topk policy covered `5` keys.
  - full-history topk policy covered `34` keys (`32` valid scored).
  - on shared valid keys, mean topk regret improved from `51.42%` to `20.99%` (`-30.42` points).
  - artifact: `data/audits/policy_regret_growth_overlap_summary.json`.
- Targeted-sampling smoke:
  - run `smoke_targeted_train` produced `64/64` rows with `campaign_targeted=1`, confirming active targeted path.

#### Tests
- Added tests:
  - `tests/test_trace_to_parquet.py`
  - `tests/test_dataset_summary.py`
  - `tests/test_build_targeted_campaign.py`
  - extended `tests/test_train_logistic_baseline.py` for parquet input path.
- Unit suite status:
  - `python3 -m unittest discover -s tests -p "test_*.py"`
  - `Ran 35 tests`, `OK` (`2` pyarrow-dependent skips under system Python without pyarrow).

### Cross-Stack Findings Audit (Top-to-Bottom)
- Completed a focused audit of:
  - end-to-end design flow reliability (export -> firmware -> sim -> verify -> CI gate),
  - RL daemon data quality and reproducible ML handoff readiness,
  - accelerator + CPU architecture bottlenecks and next-tranche ROI.

### Design Flow Status
- Current status: **blocked** for canonical green top-to-bottom validation.
- Verified blockers:
  - `make verify` can fail in CI-style env due to missing `torchvision` dependency in the active Python environment used for model export.
  - Regression gate currently fails pinned thresholds (`best_cycles`, `avg_valid_cycles`) against baseline policy.
  - Simulator can print `*** FAILED ***` while still returning exit code `0` (false-green risk for smoke gates).
  - MCP validation path can require explicit `ACCELERA_RL_PYTHON` configuration (environment split risk).

### RL / Data Pipeline Findings
- Identified critical data-quality risks:
  - `executed_hardware_dataflow_mode` labeling can drift from exporter-actual mode when exporter fallback rewrites mode (mode evidence corruption risk).
  - Forced mode `2` control surface is inconsistent across CLI/env/export execution paths.
  - Train/eval leakage risk in benchmark-zoo workflows when selector/split are reused for both train and eval.
  - Surrogate training split is row-random and not group-safe by shape/workload family.
  - Trace source/schema fragmentation risk between per-run traces and legacy trace defaults.

### ML Handoff Plan (Simple Models First)
- Agreed direction:
  - add deterministic RL trace -> CSV extraction with stable schema/version fields,
  - train leakage-safe baseline classifiers (logistic regression) for mode/knob selection,
  - materialize reproducible `best_params` keyed by shape + activation + workload tag + sparsity bucket,
  - feed that table into existing exporter/ledger lookup path for repeatable deployment.
- Optional next step after baseline: XGBoost once dataset size and label quality are stable.

### RL / ML Tranche Implemented
- Executed-mode truth fix:
  - `auto_tuner/env/systolic_env.py` now reconciles mode fields against exporter metadata (`model_blob.meta.json`) after export.
  - Trace/info rows now carry explicit mode provenance (`executed_mode_source`, `export_mode_provenance`) and no longer rely on pre-export mode assumptions.
- Mode-control consistency:
  - `auto_tuner/rl_daemon.py` now restricts forced mode to `-1/0/1` (removed implicit mode-2 acceptance in daemon forcing path).
  - Added explicit unsupported forced-mode reason path in env mode resolution.
- Leakage hardening:
  - Added train/eval overlap guard in `auto_tuner/rl_daemon.py` (same split + overlapping families rejected by default unless explicitly opted in with `--allow-eval-train-overlap`).
  - Updated benchmark-zoo runners to pass explicit overlap opt-in where overlap is intentional for case-local A/B/eval behavior.
  - `auto_tuner/surrogate_model.py` now uses group-safe split by workload+shape instead of row-random split for validation.
- Reproducible ML pipeline:
  - Added `auto_tuner/trace_to_csv.py` for deterministic normalized trace extraction.
  - Added `auto_tuner/train_logistic_baseline.py` for deterministic one-vs-rest logistic baselines (mode + knobs).
  - Added training data selection policy controls:
    - `best1` (backward-compatible),
    - `topk` via `--top-k`,
    - `within_pct` via `--within-pct`.
  - Split discipline is enforced at key/group level before row expansion to prevent leakage when using `topk`/`within_pct`.
  - Final materialization contract remains best-1 per key while training can use richer top-k/near-optimal supervision.
  - Emits `best_params_v1.json` with BKM-compatible keys and supports optional ledger materialization output.
- Build/test hooks:
  - Added `Makefile` targets: `trace_to_csv`, `train_logistic_baseline`, `materialize_best_params`.
  - Added tests: `tests/test_trace_to_csv.py`, `tests/test_train_logistic_baseline.py`, `tests/test_surrogate_model.py`.

### Accelerator + CPU Performance Findings
- Highest-impact current bottlenecks:
  - overlap path is effectively constrained by `tile_n == 4` guard in firmware ping-pong flow,
  - command completion semantics remain completion-serialized around `done` behavior,
  - some RL knobs are partially non-operative under current hardware stepping behavior,
  - CPU dual-issue gains are limited on branch-heavy/non-ILP patterns.
- Strategic conclusion:
  - prioritize overlap/control and measurement correctness first,
  - defer major redesigns (8x8/16x16 array, deeper 5/7-stage CPU pipeline) until control/dataflow bottlenecks are removed and re-measured.

### Next Tranche (Priority Order)
1. Reliability fixes first: fail-fast simulator exit code correctness, unified Python env/deps, restore green canonical verify path.
2. RL data correctness: reconcile executed mode with exporter actual mode; resolve mode-2 path consistency.
3. Reproducible ML baseline: CSV extraction, deterministic logistic models, versioned best-params materialization.
4. Performance tranche: extend overlap beyond `tile_n==4`, then retune and rebaseline regression thresholds.

## 2026-03-22

### Orchestrator Provider Migration (OpenAI Direct)
- Migrated `agents/orchestrator.py` from Anthropic Messages API to OpenAI Responses API.
- Added default model policy for agent orchestration:
  - primary: `gpt-5.3-codex`
  - fallback on retryable API errors: `gpt-5.4`
  - default reasoning effort: `high`
- Added retryable-error fallback behavior with per-attempt trace logging.

### Daily Spend Guardrail
- Added hard daily budget enforcement to `agents/orchestrator.py`:
  - default cap: `$6.00/day`
  - per-call output token clamping based on estimated input/output cost
  - early stop when remaining daily budget is insufficient.
- Added persistent ledger tracking in `agents/budget_ledger.json`:
  - per-call token usage
  - per-call estimated USD cost
  - cumulative per-day spend and remaining budget.

### Agent Docs Refresh
- Updated `agents/README.md` for OpenAI-first usage and budget guardrail controls.
- Updated CLI surface in docs and tool help:
  - `--model`, `--fallback-model`, `--reasoning-effort`
  - `--daily-budget-usd`, `--budget-ledger-path`, `--budget-safety-buffer-usd`
  - `--disable-budget-guardrail`.

## 2026-03-21

### Deterministic Agent Orchestrator
- Added new `agents/` implementation for deterministic multi-call execution over Anthropic Messages API:
  - `agents/orchestrator.py`
  - `agents/README.md`
  - `agents/__init__.py`
- Implemented fixed role loop:
  - Agent A plan
  - Agent B critique/revise
  - Agent C implementation via search-and-replace edits
  - Agent B implementation review
  - validation gates (`py_compile`, `unittest discover`, optional `make verify`)
- Added run-state + trace persistence under `agents/runs/<run_id>/`:
  - per-call request/response logs
  - resumable `state.json`
  - `summary.json` status output.
- Added helper tests:
  - `tests/test_agents_orchestrator.py`.

## 2026-03-20

### Reliability Foundation
- Added canonical verification entrypoint in `Makefile`:
  - `make verify` now runs unit tests, model export/blob verification, dense smoke sim, and sparse mode-1 smoke sim.
- Fixed stale sparse unit integration in `firmware/sparse_mode1_unit.c`:
  - replaced deprecated `accel_run_ext` usage with `accel_issue_ext` + `accel_wait_done_level`.

### Interface Contract Hardening
- Added additive read-only MMIO registers in systolic accelerator:
  - `0x88` `IF_VERSION`
  - `0x8C` `CAPABILITIES`
- Added firmware query helpers in `firmware/include/accel.h`:
  - `accel_if_version`, `accel_capabilities`, `accel_has_capability`, `accel_supports_hw_mode`.
- Added interface freeze and compatibility policy doc:
  - `docs/ACCELERATOR_INTERFACE.md`.

### Deterministic Backend Behavior
- Added explicit backend forcing via `ACCELERA_ARRAY_BACKEND_FORCE` in:
  - `auto_tuner/workload_export.py`
  - `compiler/sparsity_utils.py`
- Updated `tests/test_mlx_fallback.py` to force deterministic NumPy backend selection independent of host MLX availability.

### Export + Manifest Observability
- Extended model export with optional metadata sidecar:
  - `model_blob.meta.json` now records selected dataflow mode provenance and required capability hints.
- Updated export paths to emit metadata:
  - `workloads/mnist/mnist_mlp_export.py`
  - `auto_tuner/workload_export.py`
- Extended run manifest summary in `auto_tuner/rl_daemon.py`:
  - per-workload metrics
  - mode distribution
  - per-shape preferred mode summary
  - averaged performance counters.

### CI and Regression Governance
- Updated `.github/workflows/mcp-regression.yml` to run `make verify` as canonical gate.
- Versioned baseline config (`schema_version`) and added per-workload thresholds in `.github/mcp_regression_baseline.json`.
- Hardened `mcp/regression_gate.py`:
  - requires versioned baseline config
  - enforces global plus per-workload regression thresholds when per-workload manifest metrics are available.

### Benchmark Zoo Expansion
- Expanded `auto_tuner/workload_bank.py` beyond MNIST-centric mixes with new workload families:
  - `dnn_infer`, `dnn_train_fwd`, `dnn_train_bwd`
  - `cnn_train_fwd`, `cnn_train_bwd`
  - `rnn_infer`, `rnn_train_bptt`
- Added workload selector aliases for broader suites:
  - `inference`, `training`, `vision`, `sequence`, `dense`, `legacy`, `zoo_v1`.
- Added versioned benchmark-zoo manifest + loader:
  - `auto_tuner/benchmark_zoo_v1.json`
  - `auto_tuner/benchmark_zoo.py`
- Added benchmark-zoo runner:
  - `auto_tuner/run_benchmark_zoo.py` (writes summary JSON + per-shape CSV under `data/`).
- Extended evaluation tooling with benchmark-zoo scenarios in `auto_tuner/eval_matrix.py`.
- Added MCP benchmark-zoo tooling:
  - `run_benchmark_zoo`
  - `start_benchmark_zoo_job`.

### Structured Sparsity Evidence + Metrics
- Hardened structured sparsity utilities in `compiler/sparsity_utils.py`:
  - added `structured_2_4_group_compatibility(...)`
  - added explicit `structured_2_4_prune(...)` with prune delta stats
  - normalized 2:4 padding helper reused by pack/prune paths.
- Extended exporter observability in `compiler/export_model.py`:
  - mode1 path now records explicit prune metadata and 2:4 compatibility ratio
  - metadata sidecar includes `sparsity_pct`, `sparsity_bucket`, `structured_2_4_compat_ratio`, and `mode1_prune_stats`.
- Added sparsity-aware eval metrics:
  - `auto_tuner/env/systolic_env.py` now reports `sparsity_bucket`, mode1 candidacy, `pe_util_est`, and `dense_equiv_macs_per_cycle`.
  - `auto_tuner/rl_daemon.py` eval rows now include mode shares, per-mode valid cycle means, sparse/dense cycle ratio, PE-util, and dense-equivalent MAC/cycle.
  - run manifest `per_shape_mode_choice` now includes dense vs sparse cycle comparisons.
  - `auto_tuner/run_benchmark_zoo.py` summary aggregates mode-share/utilization/sparse-vs-dense ratio.
  - `auto_tuner/eval_matrix.py` console output now surfaces key sparse metrics when available.

### Dense vs Sparse A/B Harness
- Added exporter/runtime forced-mode path for controlled experiments:
  - `ACCELERA_FORCE_HW_MODE` support in `compiler/export_model.py` (0/1/2 override).
  - `--force-hw-mode` in `auto_tuner/rl_daemon.py` with run-manifest provenance.
- Added benchmark-zoo forced A/B runner:
  - `auto_tuner/run_benchmark_zoo_ab.py`
  - executes each zoo case twice (dense + sparse) with matched deterministic per-case seeds.
  - emits per-shape side-by-side CSV and per-case JSON summary.
- Added A/B metric coverage:
  - cycle ratio (`sparse_over_dense_cycle_ratio`)
  - correctness-validity proxy delta (`accuracy_delta`)
  - PE-util delta (`pe_util_delta`)
  - mode-share evidence (`dense_mode1_share`, `sparse_mode1_share`).
  - matched-shape coverage guardrail (`matched_shape_coverage`) with explicit mismatch case IDs.
  - default loud failure on mismatch (`failed_shape_mismatch`), with opt-out `--allow-shape-mismatch` for debugging only.
  - per-case diagnostic rows in JSON summary (`cases[].diagnostic_rows`) including layer dims, tile means, burst/prefetch/tile_b means, PE-util, mode-share, fallback-rate, and cycle deltas.
  - data-driven sparse-unsuitable annotation:
    - row-level (`low_sparsity_flag`, `sparse_regressed_flag`, `sparse_unsuitable_row`)
    - case-level (`sparse_unsuitable_case`, consistency metrics, threshold echo in summary).

### Documentation Refresh
- Updated README with:
  - explicit sparsity bucket contract for BKM lookup keys
  - structured 2:4 prune/pack behavior and packed word bit layout
  - expanded sparsity-aware evaluation metric definitions
  - clarified optimization attribution guidance (combined result vs per-mode evidence).
- Updated `docs/ACCELERATOR_INTERFACE.md` with mode1 packed-word interface details.
- Added CI hard gate for A/B harness shape coverage in `.github/workflows/mcp-regression.yml`:
  - fails when `shape_mismatch_case_count != 0`
  - uploads A/B JSON/CSV artifacts for audit.

## 2026-03-16

### MCP Enhancements
- Upgraded `mcp/accelera_mcp_server.py` with async job lifecycle tools:
  - `start_inference_job`, `start_rl_daemon_job`, `start_eval_matrix_job`
  - `get_job_status`, `list_jobs`, `cancel_job`
- Added analysis tools:
  - `run_manifest`, `summarize_trace`, `compare_runs`
- Added interpreter override support:
  - `ACCELERA_PYTHON` for export/lightweight commands
  - `ACCELERA_RL_PYTHON` for RL/eval commands

### Validation Workflow
- Added reusable validation script: `mcp/validate_workflow.py`
- Executed full top-to-bottom validation and captured report:
  - `data/mcp_validation_report_20260316_162021.json`
  - `overall_ok=true`

### Reliability Fixes
- Updated `auto_tuner/eval_matrix.py` to be forward-compatible with evolving eval row schema:
  - CSV writing now accepts extra fields safely.
  - Console summary now tolerates both legacy and current metric keys.

### CI Regression Gate
- Added `.github/workflows/mcp-regression.yml`:
  - runs MCP validation for PRs touching `rtl/`, `firmware/`, `auto_tuner/`, `compiler/`, and `mcp/`
  - executes `mcp/regression_gate.py` against a pinned baseline
  - uploads validation artifacts (report, eval CSVs, async logs)
- Added baseline policy config: `.github/mcp_regression_baseline.json`
- Added gating utility: `mcp/regression_gate.py`
