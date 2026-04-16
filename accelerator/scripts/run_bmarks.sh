#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BENCHES=("fib" "cachetest" "sum" "replace" "final" "dual_ilp")
if [[ $# -gt 0 ]]; then
  BENCHES=("$@")
fi

mkdir -p results
STAMP="$(date +%Y%m%d_%H%M%S)"
CSV="results/bmark_${STAMP}.csv"

echo "bench,dual_issue,status,tohost,cycles,ipc,retired_lane0,retired_lane1,retired_total,dual_issued,lane1_issue_en,flush_id,freeze" > "$CSV"

extract_field() {
  local text="$1"
  local regex="$2"
  if [[ "$text" =~ $regex ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo ""
  fi
}

for dual in 0 1; do
  echo "[run_bmarks] Building simulator ENABLE_DUAL_ISSUE=${dual}" >&2
  make clean >/dev/null
  make obj_dir/Vriscv_top ENABLE_DUAL_ISSUE=${dual} >/dev/null

  for bench in "${BENCHES[@]}"; do
    echo "[run_bmarks] Running bench=${bench} dual=${dual}" >&2
    set +e
    output=$(make run_bench \
      ENABLE_DUAL_ISSUE=${dual} \
      BENCH_NAME=${bench} \
      BENCH_SRC="${ROOT_DIR}/benchmarks/bmark/${bench}.c" \
      BENCH_LD="${ROOT_DIR}/benchmarks/bmark/common.ld" \
      2>&1)
    cmd_rc=$?
    set -e

    log_file="results/${bench}_di${dual}_${STAMP}.log"
    printf "%s\n" "$output" > "$log_file"

    status="ERROR"
    if grep -q "\*\*\* PASSED \*\*\*" <<< "$output"; then
      status="PASSED"
    elif grep -q "\*\*\* FAILED \*\*\*" <<< "$output"; then
      status="FAILED"
    elif grep -q "\*\*\* TIMEOUT \*\*\*" <<< "$output"; then
      status="TIMEOUT"
    elif [[ $cmd_rc -ne 0 ]]; then
      status="ERROR"
    fi

    tohost=$(extract_field "$output" 'tohost = ([0-9]+)')
    cycles=$(extract_field "$output" 'Total cycles: ([0-9]+)')
    ipc=$(extract_field "$output" 'IPC=([0-9]+\.[0-9]+)')

    retired_lane0=$(extract_field "$output" 'Retired: lane0=([0-9]+)')
    retired_lane1=$(extract_field "$output" 'Retired: lane0=[0-9]+ lane1=([0-9]+)')
    retired_total=$(extract_field "$output" 'Retired: lane0=[0-9]+ lane1=[0-9]+ total=([0-9]+)')

    dual_issued=$(extract_field "$output" 'Counter: dual_issued=([0-9]+)')
    lane1_issue_en=$(extract_field "$output" 'Counter: dual_issued=[0-9]+ lane1_issue_en=([0-9]+)')
    flush_id=$(extract_field "$output" 'Counter: dual_issued=[0-9]+ lane1_issue_en=[0-9]+ flush_id=([0-9]+)')
    freeze=$(extract_field "$output" 'Counter: dual_issued=[0-9]+ lane1_issue_en=[0-9]+ flush_id=[0-9]+ freeze=([0-9]+)')

    echo "${bench},${dual},${status},${tohost},${cycles},${ipc},${retired_lane0},${retired_lane1},${retired_total},${dual_issued},${lane1_issue_en},${flush_id},${freeze}" >> "$CSV"

    printf "[run_bmarks] %-10s DI=%d status=%-7s cycles=%s IPC=%s log=%s\n" \
      "$bench" "$dual" "$status" "${cycles:-NA}" "${ipc:-NA}" "$log_file" >&2
  done
done

echo "[run_bmarks] Wrote $CSV" >&2
