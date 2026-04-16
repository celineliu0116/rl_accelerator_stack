#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import parse_qs, urlparse

try:
    from auto_tuner.tuning_trace import load_traces, normalize_trace_row
except Exception:
    from tuning_trace import load_traces, normalize_trace_row

_RUN_WINDOW_CACHE: Dict[Tuple[str, str, int], Dict[str, Any]] = {}


def _default_workspace() -> str:
    return str(Path(__file__).resolve().parents[1])


def _safe_load_json(path: str) -> Dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_run_manifests(workspace: str) -> List[Dict[str, Any]]:
    runs_dir = os.path.join(workspace, "data", "runs")
    if not os.path.isdir(runs_dir):
        return []
    out: List[Dict[str, Any]] = []
    for p in sorted(Path(runs_dir).glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        rec = _safe_load_json(str(p))
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _rows_for_run(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    run_id = str(manifest.get("run_id", ""))
    trace_path = str(manifest.get("trace_path", ""))
    if not run_id or not trace_path or not os.path.exists(trace_path):
        return []
    rows = [normalize_trace_row(r) for r in load_traces(trace_path)]
    rows = [r for r in rows if str(r.get("run_id", "")) == run_id]
    rows.sort(key=lambda r: (float(r.get("timestamp", 0.0)), int(r.get("episode_id", 0))))
    return rows


def _rows_for_run_recent(manifest: Dict[str, Any], limit: int = 1000) -> tuple[List[Dict[str, Any]], int]:
    run_id = str(manifest.get("run_id", ""))
    trace_path = str(manifest.get("trace_path", ""))
    if not run_id or not trace_path or not os.path.exists(trace_path):
        return [], 0

    lim = max(1, min(5000, int(limit)))
    st = os.stat(trace_path)
    key = (trace_path, run_id, lim)
    cached = _RUN_WINDOW_CACHE.get(key)
    if cached and int(cached.get("mtime_ns", -1)) == int(st.st_mtime_ns) and int(cached.get("size", -1)) == int(st.st_size):
        return list(cached.get("rows", [])), int(cached.get("total", 0))

    window: deque[Dict[str, Any]] = deque(maxlen=lim)
    total = 0
    with open(trace_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except Exception:
                continue
            if str(rec.get("run_id", "")) != run_id:
                continue
            total += 1
            window.append(normalize_trace_row(rec))

    rows = list(window)
    _RUN_WINDOW_CACHE[key] = {
        "mtime_ns": int(st.st_mtime_ns),
        "size": int(st.st_size),
        "rows": rows,
        "total": int(total),
    }
    return rows, total


def _recompute_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid_rows = [r for r in rows if int(r.get("is_valid", 0)) == 1]
    valid_cycles = [int(r.get("cycles", -1)) for r in valid_rows if int(r.get("cycles", -1)) > 0]
    valid_rewards = [float(r.get("reward", -10000.0)) for r in valid_rows]
    reject_rows = []
    for r in rows:
        reason = r.get("reject_reason")
        if isinstance(reason, str):
            rr = reason.strip().lower()
            if rr in ("", "none"):
                continue
        elif reason is None:
            continue
        reject_rows.append(r)
    constraint_reject_rows = []
    for r in rows:
        violations = r.get("constraint_violations")
        if isinstance(violations, list) and violations:
            constraint_reject_rows.append(r)
            continue
        if int(r.get("constraint_ok", 1)) == 0:
            constraint_reject_rows.append(r)
    reject_by_reason: Dict[str, int] = {}
    for r in reject_rows:
        reason = str(r.get("reject_reason", "unknown"))
        reject_by_reason[reason] = reject_by_reason.get(reason, 0) + 1
    return {
        "total_rows": len(rows),
        "valid_simulated_rows": len(valid_rows),
        "rejected_rows": len(reject_rows),
        "constraint_reject_rows": len(constraint_reject_rows),
        "best_cycles": min(valid_cycles) if valid_cycles else -1,
        "avg_valid_cycles": (sum(valid_cycles) / len(valid_cycles)) if valid_cycles else -1.0,
        "avg_valid_reward": (sum(valid_rewards) / len(valid_rewards)) if valid_rewards else -10000.0,
        "reject_by_reason": reject_by_reason,
    }


def _run_list_payload(workspace: str) -> List[Dict[str, Any]]:
    manifests = _load_run_manifests(workspace)
    runs: List[Dict[str, Any]] = []
    for m in manifests:
        runs.append({
            "run_id": str(m.get("run_id", "")),
            "status": str(m.get("status", "unknown")),
            "workload": str(m.get("workload", "")),
            "train_shape_split": str(m.get("train_shape_split", "")),
            "command": str(m.get("command", "")),
            "started_at": str(m.get("started_at", "")),
            "completed_at": str(m.get("completed_at", "")),
            "total_rows": int(m.get("total_rows", 0)),
            "valid_simulated_rows": int(m.get("valid_simulated_rows", 0)),
            "rejected_rows": int(m.get("rejected_rows", 0)),
            "constraint_reject_rows": int(m.get("constraint_reject_rows", 0)),
            "best_cycles": int(m.get("best_cycles", -1)),
            "avg_valid_cycles": float(m.get("avg_valid_cycles", -1.0)),
            "avg_valid_reward": float(m.get("avg_valid_reward", -10000.0)),
            "policy_saved": bool(m.get("policy_saved", False)),
        })
    return runs


def _summary_from_manifest(manifest: Dict[str, Any]) -> Dict[str, Any]:
    reject_by_reason = manifest.get("reject_by_reason", {})
    if not isinstance(reject_by_reason, dict):
        reject_by_reason = {}
    return {
        "total_rows": int(manifest.get("total_rows", 0)),
        "valid_simulated_rows": int(manifest.get("valid_simulated_rows", 0)),
        "rejected_rows": int(manifest.get("rejected_rows", 0)),
        "constraint_reject_rows": int(manifest.get("constraint_reject_rows", 0)),
        "best_cycles": int(manifest.get("best_cycles", -1)),
        "avg_valid_cycles": float(manifest.get("avg_valid_cycles", -1.0)),
        "avg_valid_reward": float(manifest.get("avg_valid_reward", -10000.0)),
        "reject_by_reason": reject_by_reason,
    }


def _run_detail_payload(workspace: str, run_id: str, candidate_limit: int = 1000) -> Dict[str, Any] | None:
    manifests = _load_run_manifests(workspace)
    target = None
    for m in manifests:
        if str(m.get("run_id", "")) == run_id:
            target = m
            break
    if target is None:
        return None
    rows, total_for_run = _rows_for_run_recent(target, candidate_limit)
    summary = _summary_from_manifest(target)
    if str(target.get("status", "")).lower() == "running" and summary["total_rows"] <= 0:
        # Running manifests may not have final aggregates yet.
        summary = _recompute_summary(rows)
        summary["total_rows"] = max(int(summary.get("total_rows", 0)), int(total_for_run))
    return {
        "manifest": target,
        "summary": summary,
        "candidates": rows,
        "candidates_total": int(total_for_run),
        "candidates_truncated": int(total_for_run) > int(len(rows)),
    }


def _run_manifest_path(workspace: str, run_id: str) -> str:
    safe = os.path.basename(str(run_id))
    return os.path.join(workspace, "data", "runs", f"{safe}.json")


def _prune_trace_rows(trace_path: str, run_id: str) -> int:
    if not trace_path or not os.path.exists(trace_path):
        return 0
    removed = 0
    tmp_path = trace_path + ".tmp"
    with open(trace_path, "r", encoding="utf-8", errors="replace") as src, \
         open(tmp_path, "w", encoding="utf-8") as dst:
        for line in src:
            s = line.strip()
            if not s:
                continue
            keep = True
            try:
                rec = json.loads(s)
                if str(rec.get("run_id", "")) == str(run_id):
                    keep = False
                    removed += 1
            except Exception:
                keep = True
            if keep:
                dst.write(line if line.endswith("\n") else (line + "\n"))
    os.replace(tmp_path, trace_path)
    return int(removed)


def _delete_runs(workspace: str, run_ids: List[str],
                 purge_trace: bool = False,
                 force_running: bool = False) -> Dict[str, Any]:
    manifests = {str(m.get("run_id", "")): m for m in _load_run_manifests(workspace)}
    deleted: List[str] = []
    skipped: List[Dict[str, str]] = []
    pruned_rows = 0
    for rid in run_ids:
        run_id = str(rid).strip()
        if not run_id:
            continue
        manifest = manifests.get(run_id)
        if manifest is None:
            skipped.append({"run_id": run_id, "reason": "not_found"})
            continue
        status = str(manifest.get("status", "")).lower()
        if status == "running" and not force_running:
            skipped.append({"run_id": run_id, "reason": "running"})
            continue

        if purge_trace:
            trace_path = str(manifest.get("trace_path", ""))
            try:
                pruned_rows += _prune_trace_rows(trace_path, run_id)
            except Exception:
                skipped.append({"run_id": run_id, "reason": "trace_prune_failed"})
                continue

        manifest_path = _run_manifest_path(workspace, run_id)
        try:
            if os.path.exists(manifest_path):
                os.remove(manifest_path)
            deleted.append(run_id)
        except Exception:
            skipped.append({"run_id": run_id, "reason": "manifest_delete_failed"})
    _RUN_WINDOW_CACHE.clear()
    return {
        "deleted": deleted,
        "skipped": skipped,
        "deleted_count": len(deleted),
        "skipped_count": len(skipped),
        "pruned_trace_rows": int(pruned_rows),
    }


def _html_page() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Accelera Tuning Run Explorer</title>
  <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #f4f1e9;
      --card: #fffef8;
      --ink: #1c2127;
      --muted: #7a7d82;
      --line: #ddd7c8;
      --accent: #0d6b5e;
      --accent-light: #e8f5f1;
      --warn: #b45309;
      --bad: #b91c1c;
      --radius: 10px;
      --shadow: 0 6px 20px rgba(0,0,0,0.06);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: 'Source Serif 4', 'Iowan Old Style', Georgia, serif;
      color: var(--ink);
      background: var(--bg);
      min-height: 100vh;
    }
    .wrap { max-width: 1400px; margin: 0 auto; padding: 20px 20px 40px; }

    header { display: flex; align-items: baseline; gap: 12px; margin-bottom: 18px; flex-wrap: wrap; }
    header h1 { font-size: 26px; margin: 0; letter-spacing: -0.5px; }
    header .sub { color: var(--muted); font-size: 14px; margin: 0; }

    .layout { display: grid; grid-template-columns: 320px 1fr; gap: 16px; align-items: start; }
    @media (max-width: 960px) { .layout { grid-template-columns: 1fr; } }

    .card {
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .card-header {
      padding: 12px 16px;
      font-size: 15px;
      font-weight: 600;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(135deg, #f0f7f4, #fffdf5);
      letter-spacing: -0.2px;
    }
    .card-body { padding: 14px 16px; }

    .run-item {
      padding: 10px 14px;
      border-bottom: 1px solid var(--line);
      cursor: pointer;
      transition: background 0.15s;
    }
    .runs-header-row { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
    .runs-tools { display: flex; gap: 6px; flex-wrap: wrap; justify-content: flex-end; }
    .runs-subrow { margin-top: 8px; display: flex; align-items: center; justify-content: space-between; gap: 8px; flex-wrap: wrap; }
    .runs-msg { font-size: 11px; color: var(--muted); }
    .run-btn {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 6px;
      padding: 4px 8px;
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
    }
    .run-btn.danger { border-color: #ef9a9a; color: #991b1b; }
    .run-btn:hover { background: #f8f5ec; }
    .run-top { display: flex; align-items: center; gap: 8px; }
    .run-check { display: inline-flex; align-items: center; }
    .run-check input { width: 14px; height: 14px; }
    .run-item:hover { background: #f7f5ee; }
    .run-item.active { background: var(--accent-light); border-left: 3px solid var(--accent); }
    .run-item .run-id { font-family: 'DM Mono', monospace; font-size: 13px; font-weight: 500; }
    .run-item .run-meta { font-size: 12px; color: var(--muted); margin-top: 3px; display: flex; gap: 8px; flex-wrap: wrap; }
    .run-item .run-meta span { white-space: nowrap; }

    .pill { font-weight: 600; font-size: 11px; padding: 2px 8px; border-radius: 999px; display: inline-block; text-transform: uppercase; letter-spacing: 0.03em; }
    .pill-running { color: #0369a1; background: #e0f2fe; }
    .pill-completed { color: #166534; background: #dcfce7; }
    .pill-failed { color: #991b1b; background: #fee2e2; }
    .pill-interrupted { color: #92400e; background: #fef3c7; }

    .kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: 8px; margin-bottom: 14px; }
    .kpi {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      background: #fff;
    }
    .kpi .val { font-size: 22px; font-weight: 700; font-family: 'DM Mono', monospace; }
    .kpi .lbl { color: var(--muted); font-size: 11px; margin-top: 2px; text-transform: uppercase; letter-spacing: 0.04em; }

    .cols2 { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px; }
    @media (max-width: 800px) { .cols2 { grid-template-columns: 1fr; } }

    .cfg {
      font-family: 'DM Mono', monospace;
      font-size: 12px;
      background: #faf8f1;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      white-space: pre-wrap;
      word-break: break-all;
      line-height: 1.5;
    }
    .small { font-size: 12px; color: var(--muted); }

    .best-box {
      border: 1px solid var(--accent);
      border-radius: 8px;
      background: var(--accent-light);
      padding: 12px;
      font-size: 13px;
      line-height: 1.6;
    }
    .best-box b { color: var(--accent); }

    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { text-align: left; border-bottom: 1px solid #ece7da; padding: 6px 8px; }
    th { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; font-weight: 600; }

    .bar-track { background: #efebe0; border: 1px solid #d9d3c5; height: 10px; border-radius: 999px; overflow: hidden; }
    .bar-fill { display: block; height: 100%; background: linear-gradient(90deg, #14b8a6, #0f766e); border-radius: 999px; }

    .controls {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 10px;
      align-items: end;
    }
    .ctrl-group { display: flex; flex-direction: column; min-width: 0; }
    .ctrl-group label { font-size: 11px; color: var(--muted); margin-bottom: 3px; text-transform: uppercase; letter-spacing: 0.03em; }
    .ctrl-group select, .ctrl-group input {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 6px;
      padding: 5px 8px;
      font-size: 12px;
      font-family: inherit;
      width: 130px;
    }
    .ctrl-group input { width: 150px; }

    .cand-wrap {
      width: 100%;
      overflow-x: auto;
      border: 1px solid #ece7da;
      border-radius: 8px;
      background: #fff;
    }
    .cand-table { width: 100%; min-width: 0; table-layout: fixed; }
    .cand-table th, .cand-table td {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      padding: 5px 6px;
      font-size: 12px;
    }
    .cand-table th { background: #f8f5ec; position: sticky; top: 0; z-index: 1; }
    .cand-table .c-idx { width: 32px; }
    .cand-table .c-ep { width: 40px; }
    .cand-table .c-wk { width: 11%; }
    .cand-table .c-shape { width: 14%; }
    .cand-table .c-tile { width: 44px; text-align: right; }
    .cand-table .c-mode { width: 11%; }
    .cand-table .c-sp { width: 52px; text-align: right; }
    .cand-table .c-flag { width: 30px; text-align: center; }
    .cand-table .c-num { width: 72px; text-align: right; font-family: 'DM Mono', monospace; }
    .cand-table .c-raw { width: 72px; text-align: right; font-family: 'DM Mono', monospace; }
    .cand-table .c-rej { width: 14%; }

    .cand-table tbody tr { cursor: pointer; transition: background 0.1s; }
    .cand-table tbody tr:hover { background: #f7f5ee; }
    .cand-table tbody tr.expanded { background: var(--accent-light); }

    .cand-detail-row td { white-space: normal; padding: 0; background: #fafdf9; }
    .cand-detail-inner {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 6px 16px;
      padding: 10px 14px;
      font-size: 12px;
      line-height: 1.6;
      border-top: 1px dashed var(--line);
      border-bottom: 1px dashed var(--line);
    }
    .cand-detail-inner .d-label { color: var(--muted); font-size: 11px; }
    .cand-detail-inner .d-val { font-family: 'DM Mono', monospace; font-weight: 500; }
    @media (max-width: 700px) { .cand-detail-inner { grid-template-columns: 1fr 1fr; } }

    .reject-tbl { width: auto; }
    .reject-tbl td:last-child { text-align: right; font-family: 'DM Mono', monospace; }

    .empty-msg { color: var(--muted); font-size: 13px; padding: 20px 0; text-align: center; }
  </style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>Tuning Run Explorer</h1>
    <p class="sub">Local dashboard for RL autotuning runs</p>
  </header>
  <div class="layout">
    <div class="card" id="runsCard">
      <div class="card-header">
        <div class="runs-header-row">
          <span>Runs</span>
          <div class="runs-tools">
            <button id="btnSelectAllRuns" class="run-btn" type="button">Select All</button>
            <button id="btnDeleteSelectedRuns" class="run-btn danger" type="button">Delete Selected</button>
            <button id="btnDeleteCompletedRuns" class="run-btn" type="button">Delete Completed</button>
          </div>
        </div>
        <div class="runs-subrow">
          <label class="small"><input id="chkPurgeTraceRows" type="checkbox" /> also prune trace rows (slower)</label>
          <span id="runsMsg" class="runs-msg"></span>
        </div>
      </div>
      <div id="runsList"></div>
    </div>
    <div class="card">
      <div class="card-header">Run Detail</div>
      <div class="card-body" id="detail">
        <div class="empty-msg">Select a run from the list.</div>
      </div>
    </div>
  </div>
</div>

<script>
const runsList = document.getElementById("runsList");
const detail = document.getElementById("detail");
const runsMsg = document.getElementById("runsMsg");
let selectedRun = null;
let selectedPayload = null;
let expandedCandIdx = null;
let latestRuns = [];
let selectedRunIds = new Set();
const API_CANDIDATE_LIMIT = 1000;
const RUNS_POLL_MS = 10000;

function pillClass(s) {
  return ({ running: "pill-running", completed: "pill-completed", failed: "pill-failed", interrupted: "pill-interrupted" })[s] || "";
}
function fmt(v, digits) {
  if (v == null) return "-";
  if (typeof v === "number") {
    if (!Number.isFinite(v) || v <= -9999) return "-";
    return v.toLocaleString(undefined, { maximumFractionDigits: digits ?? 2 });
  }
  return String(v);
}
function selectorLabel(sel) {
  const s = String(sel || "").trim().toLowerCase();
  if (!s) return "-";
  if (s === "all") return "all (gemm, sparse_mlp, convolution, attention)";
  return s;
}
function opTypeName(row) {
  const tag = String((row && row.workload_tag) || "").trim();
  if (tag) return tag;
  const id = Number((row && row.op_type_id) ?? -1);
  if (id === 0) return "gemm";
  if (id === 1) return "sparse_mlp";
  if (id === 2) return "convolution";
  if (id === 3) return "attention";
  return "unknown";
}
function shapeMeaning(row) {
  const wk = opTypeName(row);
  const M = Number((row && row.M) ?? 0);
  const N = Number((row && row.N) ?? 0);
  const K = Number((row && row.K) ?? 0);
  if (wk === "convolution") return `M=${M} (OH*OW), N=${N} (C_out), K=${K} (C_in*R*S)`;
  if (wk === "attention") return `M=${M} (seq positions), N=${N} (proj out), K=${K} (proj in)`;
  return `M=${M} (rows/batch), N=${N} (output), K=${K} (reduction/input)`;
}
function modeLabel(row) {
  const execName = String((row && (row.executed_mode_name || row.mode_name)) || "UNKNOWN");
  const propName = String((row && row.proposed_mode_name) || execName);
  if (propName === execName) return execName;
  return `${execName} <- ${propName}`;
}
function modeTooltip(row) {
  const execMode = Number((row && row.executed_hardware_dataflow_mode) ?? (row && row.hardware_dataflow_mode) ?? -1);
  const propMode = Number((row && row.proposed_hardware_dataflow_mode) ?? execMode);
  const reason = String((row && row.mode_fallback_reason) || "").trim();
  let t = `executed=${execMode}, proposed=${propMode}`;
  if (reason) t += `, reason=${reason}`;
  return t;
}
function ratio(n, d) { return (!d || d <= 0) ? 0 : Math.max(0, Math.min(100, 100 * n / d)); }
function isRejected(r) { const s = String(r.reject_reason || "").trim().toLowerCase(); return s !== "" && s !== "none"; }
function isValid(r) { return Number(r.is_valid) === 1; }
function fmtTs(ts) {
  const s = String(ts || "").trim();
  if (!s) return "-";
  const d = new Date(s);
  if (Number.isNaN(d.getTime())) return s;
  return d.toLocaleString(undefined, {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
    hour12: false,
    timeZoneName: "short",
  });
}
function setRunsMsg(msg, isErr) {
  if (!runsMsg) return;
  runsMsg.textContent = msg || "";
  runsMsg.style.color = isErr ? "#991b1b" : "var(--muted)";
}

function renderRuns(rows) {
  latestRuns = rows.slice();
  const allowed = new Set(rows.map(r => r.run_id));
  selectedRunIds = new Set([...selectedRunIds].filter(rid => allowed.has(rid)));
  runsList.innerHTML = "";
  if (!rows.length) { runsList.innerHTML = '<div class="empty-msg">No runs found.</div>'; return; }
  rows.forEach(r => {
    const div = document.createElement("div");
    div.className = "run-item" + (selectedRun === r.run_id ? " active" : "");
    const checked = selectedRunIds.has(r.run_id) ? "checked" : "";
    const completedAt = String(r.completed_at || "").trim();
    div.innerHTML = `
      <div class="run-top">
        <label class="run-check"><input type="checkbox" data-runid="${r.run_id}" ${checked}></label>
        <div class="run-id">${r.run_id} <span class="pill ${pillClass(r.status)}">${r.status}</span></div>
      </div>
      <div class="run-meta">
        <span title="${r.workload || "-"}">${selectorLabel(r.workload || "-")}</span>
        <span>${fmt(r.total_rows, 0)} rows</span>
        <span>${fmt(r.valid_simulated_rows, 0)} valid</span>
        <span>best: ${fmt(r.best_cycles, 0)}</span>
        <span title="${r.started_at || ""}">start: ${fmtTs(r.started_at)}</span>
        <span title="${completedAt || ""}">done: ${completedAt ? fmtTs(completedAt) : "-"}</span>
      </div>`;
    div.onclick = (e) => {
      if (e.target && e.target.matches('input[type="checkbox"]')) return;
      loadRun(r.run_id);
    };
    const cb = div.querySelector('input[type="checkbox"]');
    if (cb) {
      cb.addEventListener("click", (e) => e.stopPropagation());
      cb.addEventListener("change", () => {
        if (cb.checked) selectedRunIds.add(r.run_id);
        else selectedRunIds.delete(r.run_id);
        setRunsMsg(`${selectedRunIds.size} selected`, false);
      });
    }
    runsList.appendChild(div);
  });
  if (!selectedRunIds.size) setRunsMsg("", false);
}

function numSort(row, key) {
  const v = Number(row[key]);
  if (!Number.isFinite(v)) return -1;
  if (key === "cycles" && v <= 0) return Number.MAX_SAFE_INTEGER;
  return v;
}
const numKeys = new Set(["cycles","reward","reward_raw","tile_m","tile_n","tile_b","episode_id","prefetch_depth","burst_size","hardware_dataflow_mode","scratch_required_bytes","scratch_limit_bytes","scratch_util_pct","action_pruned"]);

function filterSort(rows, c) {
  let out = rows.slice();
  if (c.validity === "valid") out = out.filter(isValid);
  if (c.validity === "rejected") out = out.filter(isRejected);
  if (c.simulated === "sim") out = out.filter(r => Number(r.simulated) === 1);
  if (c.simulated === "nonsim") out = out.filter(r => Number(r.simulated) === 0);
  const q = (c.rejectQuery || "").trim().toLowerCase();
  if (q) out = out.filter(r => String(r.reject_reason || "").toLowerCase().includes(q));
  out.sort((a, b) => {
    let cmp = numKeys.has(c.sortBy) ? numSort(a, c.sortBy) - numSort(b, c.sortBy) : String(a[c.sortBy]||"").localeCompare(String(b[c.sortBy]||""));
    return c.sortOrder === "desc" ? -cmp : cmp;
  });
  return out;
}

function bestCandidate(rows) {
  const v = rows.filter(isValid).filter(r => Number(r.cycles) > 0);
  if (!v.length) return null;
  v.sort((a, b) => Number(a.cycles) - Number(b.cycles));
  return v[0];
}

function getControls() {
  return {
    sortBy: (document.getElementById("cSortBy") || {}).value || "cycles",
    sortOrder: (document.getElementById("cSortOrd") || {}).value || "asc",
    validity: (document.getElementById("cValid") || {}).value || "all",
    simulated: (document.getElementById("cSim") || {}).value || "all",
    rejectQuery: (document.getElementById("cRejQ") || {}).value || "",
    renderLimit: (document.getElementById("cRenderLimit") || {}).value || "200",
  };
}

function renderDetail(payload) {
  selectedPayload = payload;
  expandedCandIdx = null;
  const m = payload.manifest;
  const s = payload.summary;
  const rows = payload.candidates || [];
  const candidatesTotal = Number(payload.candidates_total || rows.length);
  const truncated = !!payload.candidates_truncated;
  const prev = getControls();
  const viewedAll = filterSort(rows, prev);
  const renderLimit = Math.max(50, Math.min(2000, Number(prev.renderLimit) || 200));
  const viewed = viewedAll.slice(0, renderLimit);
  const best = bestCandidate(rows);

  const rejectItems = Object.entries(s.reject_by_reason || {}).map(([k,v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join("");

  const validBars = rows.filter(isValid).filter(r => Number(r.cycles) > 0).sort((a,b) => Number(a.cycles) - Number(b.cycles)).slice(0, 20).map((r, i) => {
    const bestC = Math.max(1, Number(s.best_cycles) || 1);
    const cyc = Math.max(1, Number(r.cycles) || 1);
    const w = Math.max(2, Math.min(100, 100 * bestC / cyc));
    return `<tr><td>#${i+1}</td><td style="font-family:'DM Mono',monospace">${fmt(cyc,0)}</td><td style="width:55%"><div class="bar-track"><span class="bar-fill" style="width:${w}%"></span></div></td></tr>`;
  }).join("");

  const bestHtml = best ? `
    <div class="best-box">
      <div><b>workload/op:</b> ${opTypeName(best)}</div>
      <div><b>shape:</b> ${best.shape_signature}</div>
      <div><b>shape meaning:</b> ${shapeMeaning(best)}</div>
      <div><b>mode:</b> ${modeLabel(best)}</div>
      <div><b>sparsity:</b> ${fmt(best.sparsity_pct,0)}%</div>
      <div><b>tile:</b> ${best.tile_m} x ${best.tile_n} x ${best.tile_b}</div>
      <div><b>burst / prefetch:</b> ${best.burst_size} / ${best.prefetch_depth}</div>
      <div><b>scratch:</b> ${fmt(best.scratch_required_bytes)} / ${fmt(best.scratch_limit_bytes)} (${fmt(best.scratch_util_pct)}%)</div>
      <div><b>cycles:</b> ${fmt(best.cycles)} &nbsp; <b>reward:</b> ${fmt(best.reward)} &nbsp; <b>raw:</b> ${fmt(best.reward_raw,2)}</div>
      <div><b>episode:</b> ${fmt(best.episode_id)} &nbsp; <b>sim:</b> ${fmt(best.simulated)}</div>
    </div>` : `<div class="small">No valid simulated candidate yet.</div>`;

  const candRows = viewed.map((r, idx) => `
    <tr data-cidx="${idx}">
      <td class="c-idx">${idx+1}</td>
      <td class="c-ep">${fmt(r.episode_id,0)}</td>
      <td class="c-wk" title="${opTypeName(r)}">${opTypeName(r)}</td>
      <td class="c-shape" title="${shapeMeaning(r)}">${r.shape_signature}</td>
      <td class="c-tile">${r.tile_m}x${r.tile_n}x${r.tile_b}</td>
      <td class="c-mode" title="${modeTooltip(r)}">${modeLabel(r)}</td>
      <td class="c-sp">${fmt(r.sparsity_pct,0)}%</td>
      <td class="c-flag">${Number(r.simulated)===1?"Y":"-"}</td>
      <td class="c-num">${fmt(r.cycles,0)}</td>
      <td class="c-num">${fmt(r.reward,2)}</td>
      <td class="c-raw">${fmt(r.reward_raw,2)}</td>
      <td class="c-rej" title="${r.reject_reason||""}">${r.reject_reason||""}</td>
    </tr>`).join("");

  detail.innerHTML = `
    <div class="kpis">
      <div class="kpi"><div class="val">${fmt(s.total_rows,0)}</div><div class="lbl">Total Rows</div></div>
      <div class="kpi"><div class="val">${fmt(s.valid_simulated_rows,0)}</div><div class="lbl">Valid Sim</div></div>
      <div class="kpi"><div class="val">${fmt(s.rejected_rows,0)}</div><div class="lbl">Rejected</div></div>
      <div class="kpi"><div class="val">${fmt(s.best_cycles,0)}</div><div class="lbl">Best Cycles</div></div>
      <div class="kpi"><div class="val">${fmt(s.avg_valid_cycles,1)}</div><div class="lbl">Avg Cycles</div></div>
      <div class="kpi"><div class="val">${fmt(s.avg_valid_reward,2)}</div><div class="lbl">Avg Reward</div></div>
    </div>

    <div class="cols2">
      <div>
        <h3 style="margin:0 0 6px; font-size:15px;">Config</h3>
        <div class="cfg">${m.command || "-"}</div>
        <p class="small" style="margin:6px 0 0">Status: <b>${m.status}</b> | Policy saved: <b>${m.policy_saved}</b><br>
        Workload selector: <b title="${m.workload || "-"}">${selectorLabel(m.workload || "-")}</b><br>
        Started: <span title="${m.started_at || ""}">${fmtTs(m.started_at)}</span> |
        Completed: <span title="${m.completed_at || ""}">${fmtTs(m.completed_at)}</span></p>
      </div>
      <div>
        <h3 style="margin:0 0 6px; font-size:15px;">Constraint Diagnostics</h3>
        <p class="small">Rejected: ${ratio(s.rejected_rows, s.total_rows).toFixed(1)}% | Constraint rejects: ${fmt(s.constraint_reject_rows,0)}</p>
        <table class="reject-tbl"><thead><tr><th>Reason</th><th>Count</th></tr></thead><tbody>${rejectItems || "<tr><td colspan='2'>None</td></tr>"}</tbody></table>
      </div>
    </div>

    <div class="cols2">
      <div>
        <h3 style="margin:0 0 6px; font-size:15px;">Best Candidate</h3>
        ${bestHtml}
      </div>
      <div>
        <h3 style="margin:0 0 6px; font-size:15px;">Valid Cycles (top 20)</h3>
        <table><thead><tr><th>#</th><th>Cycles</th><th style="width:55%">Relative</th></tr></thead><tbody>${validBars || "<tr><td colspan='3'>-</td></tr>"}</tbody></table>
      </div>
    </div>

    <h3 style="margin:14px 0 6px; font-size:15px;">Candidates</h3>
    <div class="controls">
      <div class="ctrl-group"><label>Sort By</label>
        <select id="cSortBy">
          <option value="cycles">cycles</option><option value="reward">reward</option>
          <option value="reward_raw">reward_raw</option>
          <option value="episode_id">episode</option><option value="tile_m">tile_m</option>
          <option value="tile_n">tile_n</option><option value="tile_b">tile_b</option>
          <option value="scratch_util_pct">scratch %</option><option value="action_pruned">pruned</option>
          <option value="mode_name">mode</option><option value="shape_signature">shape</option>
        </select></div>
      <div class="ctrl-group"><label>Order</label>
        <select id="cSortOrd"><option value="asc">asc</option><option value="desc">desc</option></select></div>
      <div class="ctrl-group"><label>Validity</label>
        <select id="cValid"><option value="all">all</option><option value="valid">valid only</option><option value="rejected">rejected only</option></select></div>
      <div class="ctrl-group"><label>Simulated</label>
        <select id="cSim"><option value="all">all</option><option value="sim">sim only</option><option value="nonsim">non-sim</option></select></div>
      <div class="ctrl-group"><label>Reject Filter</label>
        <input id="cRejQ" type="text" placeholder="scratchpad, timeout..."></div>
      <div class="ctrl-group"><label>Render Rows</label>
        <select id="cRenderLimit"><option value="200">200</option><option value="500">500</option><option value="1000">1000</option><option value="2000">2000</option></select></div>
    </div>
    <p class="small" style="margin:0 0 6px">
      showing ${fmt(viewed.length,0)} / ${fmt(viewedAll.length,0)} filtered rows;
      source window ${fmt(rows.length,0)} / ${fmt(candidatesTotal,0)} total
      ${truncated ? "| tail-window mode for performance" : ""}
      | click a row to expand
    </p>
    <div class="cand-wrap" style="max-height: 520px; overflow-y: auto;">
      <table class="cand-table">
        <thead><tr>
          <th class="c-idx">#</th>
          <th class="c-ep">Ep</th>
          <th class="c-wk">Workload</th>
          <th class="c-shape">Shape</th>
          <th class="c-tile">Tile</th>
          <th class="c-mode">Mode</th>
          <th class="c-sp">Sp%</th>
          <th class="c-flag">Sim</th>
          <th class="c-num">Cycles</th>
          <th class="c-num">Reward</th>
          <th class="c-raw">Raw R</th>
          <th class="c-rej">Reject Reason</th>
        </tr></thead>
        <tbody id="candBody">${candRows || "<tr><td colspan='12' class='empty-msg'>No candidates</td></tr>"}</tbody>
      </table>
    </div>`;

  const restore = {
    cSortBy: prev.sortBy,
    cSortOrd: prev.sortOrder,
    cValid: prev.validity,
    cSim: prev.simulated,
    cRejQ: prev.rejectQuery,
    cRenderLimit: String(renderLimit),
  };
  for (const [id, val] of Object.entries(restore)) {
    const el = document.getElementById(id);
    if (el) { el.value = val; el.onchange = el.oninput = () => renderDetail(selectedPayload); }
  }

  const tbody = document.getElementById("candBody");
  if (tbody) {
    tbody.addEventListener("click", e => {
      const tr = e.target.closest("tr[data-cidx]");
      if (!tr) return;
      const idx = Number(tr.dataset.cidx);
      const existing = tbody.querySelector(".cand-detail-row");
      if (existing) existing.remove();
      tbody.querySelectorAll("tr.expanded").forEach(el => el.classList.remove("expanded"));
      if (expandedCandIdx === idx) { expandedCandIdx = null; return; }
      expandedCandIdx = idx;
      tr.classList.add("expanded");
      const r = viewed[idx];
      if (!r) return;
      const detailTr = document.createElement("tr");
      detailTr.className = "cand-detail-row";
      detailTr.innerHTML = `<td colspan="12"><div class="cand-detail-inner">
        <div><span class="d-label">Workload Tag</span><br><span class="d-val">${r.workload_tag || "-"}</span></div>
        <div><span class="d-label">Operation</span><br><span class="d-val">${opTypeName(r)}</span></div>
        <div><span class="d-label">Shape</span><br><span class="d-val">${r.shape_signature || "-"}</span></div>
        <div><span class="d-label">Shape Meaning</span><br><span class="d-val">${shapeMeaning(r)}</span></div>
        <div><span class="d-label">Mode (executed<-proposed)</span><br><span class="d-val">${modeLabel(r)}</span></div>
        <div><span class="d-label">Sparsity %</span><br><span class="d-val">${fmt(r.sparsity_pct,0)}%</span></div>
        <div><span class="d-label">tile_m / tile_n / tile_b</span><br><span class="d-val">${r.tile_m} / ${r.tile_n} / ${r.tile_b}</span></div>
        <div><span class="d-label">Burst Size</span><br><span class="d-val">${fmt(r.burst_size)}</span></div>
        <div><span class="d-label">Prefetch Depth</span><br><span class="d-val">${fmt(r.prefetch_depth)}</span></div>
        <div><span class="d-label">Cycles</span><br><span class="d-val">${fmt(r.cycles)}</span></div>
        <div><span class="d-label">Reward</span><br><span class="d-val">${fmt(r.reward)}</span></div>
        <div><span class="d-label">Reward Raw</span><br><span class="d-val">${fmt(r.reward_raw)}</span></div>
        <div><span class="d-label">Episode</span><br><span class="d-val">${fmt(r.episode_id)}</span></div>
        <div><span class="d-label">Simulated</span><br><span class="d-val">${r.simulated}</span></div>
        <div><span class="d-label">Correctness</span><br><span class="d-val">${r.correctness_passed}</span></div>
        <div><span class="d-label">Action Pruned</span><br><span class="d-val">${fmt(r.action_pruned)}</span></div>
        <div><span class="d-label">Scratch Required</span><br><span class="d-val">${fmt(r.scratch_required_bytes)}</span></div>
        <div><span class="d-label">Scratch Limit</span><br><span class="d-val">${fmt(r.scratch_limit_bytes)}</span></div>
        <div><span class="d-label">Scratch Util %</span><br><span class="d-val">${fmt(r.scratch_util_pct)}%</span></div>
        <div><span class="d-label">Reject Reason</span><br><span class="d-val">${r.reject_reason || "-"}</span></div>
        <div><span class="d-label">Constraint OK</span><br><span class="d-val">${r.constraint_ok}</span></div>
        <div><span class="d-label">Constraint Violations</span><br><span class="d-val">${(r.constraint_violations || []).join(", ") || "-"}</span></div>
      </div></td>`;
      tr.after(detailTr);
    });
  }
}

async function loadRuns(refreshSelected) {
  const doRefresh = !!refreshSelected;
  const resp = await fetch("/api/runs");
  const rows = await resp.json();
  renderRuns(rows);
  if (!rows.length) return;
  if (!selectedRun) {
    await loadRun(rows[0].run_id, false);
    return;
  }
  if (doRefresh) {
    const selectedMeta = rows.find(r => r.run_id === selectedRun);
    const isRunning = !!selectedMeta && String(selectedMeta.status || "").toLowerCase() === "running";
    if (isRunning) {
      await loadRun(selectedRun, false);
    }
  }
}

async function loadRun(runId, refreshRuns) {
  const doRefresh = (refreshRuns === undefined) ? true : !!refreshRuns;
  selectedRun = runId;
  const resp = await fetch(`/api/run/${encodeURIComponent(runId)}?limit=${API_CANDIDATE_LIMIT}`);
  if (resp.status !== 200) {
    detail.innerHTML = `<div class="empty-msg">Run ${runId} not found.</div>`;
    if (doRefresh) await loadRuns(false);
    return;
  }
  renderDetail(await resp.json());
  if (doRefresh) await loadRuns(false);
}

async function deleteRuns(runIds) {
  const ids = (runIds || []).filter(Boolean);
  if (!ids.length) {
    setRunsMsg("No runs selected.", true);
    return;
  }
  const purgeTrace = !!(document.getElementById("chkPurgeTraceRows") || {}).checked;
  const ok = window.confirm(`Delete ${ids.length} run record(s)?${purgeTrace ? " This will also rewrite trace files." : ""}`);
  if (!ok) return;

  const resp = await fetch("/api/runs/delete", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({run_ids: ids, purge_trace: purgeTrace}),
  });
  const payload = await resp.json();
  const deleted = Number(payload.deleted_count || 0);
  const skipped = Number(payload.skipped_count || 0);
  const pruned = Number(payload.pruned_trace_rows || 0);

  ids.forEach(id => selectedRunIds.delete(id));
  if (selectedRun && ids.includes(selectedRun)) {
    selectedRun = null;
    detail.innerHTML = '<div class="empty-msg">Select a run from the list.</div>';
  }
  setRunsMsg(`deleted ${deleted}, skipped ${skipped}, pruned_rows ${pruned}`, skipped > 0);
  await loadRuns(false);
}

document.getElementById("btnSelectAllRuns").addEventListener("click", () => {
  const selectable = latestRuns.filter(r => String(r.status || "").toLowerCase() !== "running").map(r => r.run_id);
  const allSelected = selectable.length > 0 && selectable.every(id => selectedRunIds.has(id));
  if (allSelected) selectedRunIds.clear();
  else selectedRunIds = new Set(selectable);
  renderRuns(latestRuns);
  setRunsMsg(`${selectedRunIds.size} selected`, false);
});

document.getElementById("btnDeleteSelectedRuns").addEventListener("click", async () => {
  await deleteRuns(Array.from(selectedRunIds));
});

document.getElementById("btnDeleteCompletedRuns").addEventListener("click", async () => {
  const ids = latestRuns
    .filter(r => ["completed", "failed", "interrupted"].includes(String(r.status || "").toLowerCase()))
    .map(r => r.run_id);
  await deleteRuns(ids);
});

loadRuns(false);
setInterval(() => { loadRuns(true); }, RUNS_POLL_MS);
</script>
</body>
</html>"""


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")


class DashboardHandler(BaseHTTPRequestHandler):
    workspace = _default_workspace()

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Dict[str, Any]:
        raw_len = self.headers.get("Content-Length", "0")
        try:
            n = max(0, int(raw_len))
        except Exception:
            n = 0
        if n <= 0:
            return {}
        body = self.rfile.read(n)
        try:
            parsed = json.loads(body.decode("utf-8"))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        return {}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query or "")
        if path == "/":
            self._send(HTTPStatus.OK, "text/html; charset=utf-8", _html_page().encode("utf-8"))
            return
        if path == "/api/runs":
            self._send(HTTPStatus.OK, "application/json; charset=utf-8", _json_bytes(_run_list_payload(self.workspace)))
            return
        if path.startswith("/api/run/"):
            run_id = path[len("/api/run/"):]
            raw_limit = (query.get("limit") or ["1000"])[0]
            try:
                limit = int(raw_limit)
            except Exception:
                limit = 1000
            payload = _run_detail_payload(self.workspace, run_id, candidate_limit=limit)
            if payload is None:
                self._send(HTTPStatus.NOT_FOUND, "application/json; charset=utf-8", _json_bytes({"error": "not_found"}))
                return
            self._send(HTTPStatus.OK, "application/json; charset=utf-8", _json_bytes(payload))
            return
        self._send(HTTPStatus.NOT_FOUND, "application/json; charset=utf-8", _json_bytes({"error": "not_found"}))

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query or "")
        if path.startswith("/api/run/"):
            run_id = path[len("/api/run/"):]
            purge_trace = str((query.get("purge_trace") or ["0"])[0]).lower() in {"1", "true", "yes"}
            force_running = str((query.get("force") or ["0"])[0]).lower() in {"1", "true", "yes"}
            result = _delete_runs(
                self.workspace,
                [run_id],
                purge_trace=purge_trace,
                force_running=force_running,
            )
            self._send(HTTPStatus.OK, "application/json; charset=utf-8", _json_bytes(result))
            return
        self._send(HTTPStatus.NOT_FOUND, "application/json; charset=utf-8", _json_bytes({"error": "not_found"}))

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/runs/delete":
            payload = self._read_json()
            run_ids = payload.get("run_ids", [])
            if not isinstance(run_ids, list):
                run_ids = []
            purge_trace = bool(payload.get("purge_trace", False))
            force_running = bool(payload.get("force_running", False))
            result = _delete_runs(
                self.workspace,
                [str(x) for x in run_ids],
                purge_trace=purge_trace,
                force_running=force_running,
            )
            self._send(HTTPStatus.OK, "application/json; charset=utf-8", _json_bytes(result))
            return
        self._send(HTTPStatus.NOT_FOUND, "application/json; charset=utf-8", _json_bytes({"error": "not_found"}))


def main() -> None:
    ap = argparse.ArgumentParser(description="Local dashboard for Accelera tuning runs.")
    ap.add_argument("--workspace", type=str, default=_default_workspace(),
                    help="Workspace root containing data/runs and traces.")
    ap.add_argument("--host", type=str, default="127.0.0.1", help="Bind host.")
    ap.add_argument("--port", type=int, default=8787, help="Bind port.")
    args = ap.parse_args()

    DashboardHandler.workspace = os.path.abspath(args.workspace)
    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"[Dashboard] http://{args.host}:{args.port} workspace={DashboardHandler.workspace}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
