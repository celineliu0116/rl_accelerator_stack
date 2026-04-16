# Accelera API Orchestrator (`agents/`)

Deterministic 3-role loop using the OpenAI Responses API:

1. `Agent A` plans
2. `Agent B` critiques/revises the plan
3. `Agent C` implements via search-and-replace operations
4. `Agent B` reviews the implementation
5. Orchestrator runs validation (`py_compile`, `unittest discover`, optional `make verify`)

If validation fails or review rejects, the loop retries up to 3 rounds.

## Defaults

- Primary model: `gpt-5.3-codex`
- Fallback model on retryable API errors: `gpt-5.4`
- Reasoning effort: `high`
- Daily budget cap: `$6.00`
- Budget ledger path: `agents/budget_ledger.json`

## Quick Start

From repo root:

```bash
export OPENAI_API_KEY="<your_openai_key>"
python agents/orchestrator.py --task "add sparsity histogram to A/B output"
```

Optional strict validation:

```bash
python agents/orchestrator.py \
  --task "add sparsity histogram to A/B output" \
  --make-verify
```

## Budget Guardrail

The orchestrator enforces a hard daily spend limit by default.

- Before each model call, it estimates input tokens and clamps `max_output_tokens` to stay within the remaining daily budget.
- If remaining budget is too low, the run stops before making the next API call.
- After each call, usage and estimated cost are written to the budget ledger.

Useful flags:

```bash
--daily-budget-usd 6.0
--budget-ledger-path agents/budget_ledger.json
--budget-safety-buffer-usd 0.05
```

To disable the hard stop (not recommended):

```bash
--disable-budget-guardrail
```

## Resume a Run

```bash
python agents/orchestrator.py --resume-run-dir agents/runs/<run_id>
```

## Outputs

Each run writes to:

`agents/runs/<run_id>/`

- `state.json`: live orchestrator state
- `summary.json`: final status
- `trace/*.json`: every API request/response and orchestration event

Budget ledger:

`agents/budget_ledger.json`

- daily spend totals
- per-call token usage
- per-call estimated USD cost

## Notes

- Agent C file edits are **search/replace + optional new file content** (no diff patch format).
- Paths are constrained to repository root.
