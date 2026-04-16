# Accelera MCP Server

This directory contains an MCP server that wraps key `accelera` workflows as tools.

## What You Get

The server exposes these tools:

- `project_status`
- `export_mnist_reference`
- `run_inference`
- `run_rl_daemon`
- `run_eval_matrix`
- `latest_run_manifest`
- `run_manifest`
- `tail_trace`
- `summarize_trace`
- `compare_runs`
- `start_inference_job`
- `start_rl_daemon_job`
- `start_eval_matrix_job`
- `get_job_status`
- `list_jobs`
- `cancel_job`

## Install

From repo root:

```bash
python3 -m venv .venv-mcp
.venv-mcp/bin/python -m pip install -r mcp/requirements.txt
```

For full RL/eval workflow support in the same venv:

```bash
.venv-mcp/bin/python -m pip install stable-baselines3 gymnasium numpy
```

## Run

From repo root:

```bash
.venv-mcp/bin/python mcp/accelera_mcp_server.py
```

The server uses stdio transport by default, which is what most MCP clients expect.

### Interpreter Overrides

- `ACCELERA_PYTHON`: optional interpreter for lightweight/export commands.
- `ACCELERA_RL_PYTHON`: optional interpreter for RL/eval commands (`run_rl_daemon`, `run_eval_matrix`, and async RL/eval jobs).

## Client Config Example

Example MCP client entry:

```json
{
  "mcpServers": {
    "accelera": {
      "command": "/Users/peter/accelera/.venv-mcp/bin/python",
      "args": ["/Users/peter/accelera/mcp/accelera_mcp_server.py"],
      "cwd": "/Users/peter/accelera"
    }
  }
}
```

## Notes

- Long-running flows can be used either synchronously (`run_rl_daemon`, `run_eval_matrix`) or asynchronously (`start_*_job` + `get_job_status`).
- Commands are hardcoded wrappers around existing scripts (no arbitrary shell execution).
- Async logs are written under `data/mcp_jobs/`.

## Async Example

1. Start an RL job:

```json
{"tool":"start_rl_daemon_job","args":{"timesteps":20000,"workload":"gemm"}}
```

2. Poll progress:

```json
{"tool":"get_job_status","args":{"job_id":"rl_daemon_20260316_153000_12345_001"}}
```

3. Compare results when complete:

```json
{"tool":"compare_runs","args":{"baseline_run_id":"run_20260310_001524_54293","candidate_run_id":"run_20260312_211438_80453"}}
```

## End-to-End Validation

Run the full MCP workflow validation and write a JSON report:

```bash
ACCELERA_RL_PYTHON=/Users/peter/accelera/.venv-mcp/bin/python \
.venv-mcp/bin/python mcp/validate_workflow.py
```

The script exercises:
- sync tools (`export_mnist_reference`, `run_inference`, `run_rl_daemon`, `run_eval_matrix`)
- analysis tools (`latest_run_manifest`, `run_manifest`, `tail_trace`, `summarize_trace`, `compare_runs`)
- async lifecycle (`start_rl_daemon_job`, `get_job_status`, `cancel_job`, `list_jobs`)

## Regression Gate

Evaluate the latest validation report against a pinned baseline run:

```bash
.venv-mcp/bin/python mcp/regression_gate.py \
  --report data/mcp_validation_report_20260316_162021.json \
  --baseline-config .github/mcp_regression_baseline.json
```

Policy:
- correctness regressions fail (non-zero exit)
- performance regressions emit warnings only

Baseline policy is configured in `.github/mcp_regression_baseline.json`.
