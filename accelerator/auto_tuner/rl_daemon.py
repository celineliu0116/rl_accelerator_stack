import os
import sys
import argparse
import subprocess
import csv
import json
import shlex
import time
import random
import numpy as np
from typing import List, Dict, Any

# Keep matplotlib/font cache writable in sandboxed environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
os.makedirs(os.environ["XDG_CACHE_HOME"], exist_ok=True)

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

# Import our custom environment and ledger
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from env.systolic_env import SystolicEnv
from compiler.bkm_ledger import BKMLedger
from workload_bank import parse_workload_selector, parse_shape_split, selector_tokens
from surrogate_model import SurrogateModel
from tuning_trace import load_traces

def _default_workspace() -> str:
    # auto_tuner/rl_daemon.py -> repo root is 1 parent up
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _configure_lookup_mode(workload_aware_lookup: bool) -> None:
    # Propagates to exporter subprocesses spawned by the environment.
    if workload_aware_lookup:
        os.environ.pop("ACCELERA_GENERIC_LEDGER_LOOKUP", None)
    else:
        os.environ["ACCELERA_GENERIC_LEDGER_LOOKUP"] = "1"


def _configure_forced_hw_mode(force_hw_mode: int) -> None:
    mode = int(force_hw_mode)
    if mode < 0:
        os.environ.pop("ACCELERA_FORCE_HW_MODE", None)
        return
    if mode not in (0, 1):
        raise ValueError("force_hw_mode must be one of -1,0,1")
    os.environ["ACCELERA_FORCE_HW_MODE"] = str(mode)


def _selector_overlap(train_selector: str, eval_selector: str) -> List[str]:
    train_fams = set(parse_workload_selector(train_selector))
    eval_fams = set(parse_workload_selector(eval_selector))
    return sorted(train_fams & eval_fams)


def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _new_run_id() -> str:
    return time.strftime("run_%Y%m%d_%H%M%S", time.localtime()) + f"_{os.getpid()}"


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _load_targeted_campaign_config(path: str) -> Dict[str, Any]:
    if not str(path).strip():
        return {"path": "", "targets": [], "summary": {}}
    cfg_path = os.path.abspath(path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    targets = payload.get("targets", [])
    if not isinstance(targets, list):
        targets = []
    return {
        "path": cfg_path,
        "targets": [dict(t) for t in targets if isinstance(t, dict)],
        "summary": dict(payload.get("summary", {})) if isinstance(payload.get("summary", {}), dict) else {},
        "selection": dict(payload.get("selection", {})) if isinstance(payload.get("selection", {}), dict) else {},
        "weights": dict(payload.get("weights", {})) if isinstance(payload.get("weights", {}), dict) else {},
    }


def _set_global_seeds(seed: int) -> None:
    s = int(seed)
    random.seed(s)
    np.random.seed(s)
    os.environ["PYTHONHASHSEED"] = str(s)
    try:
        import torch
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _mean_std(vals: List[float]) -> tuple[float, float]:
    if not vals:
        return -1.0, -1.0
    if len(vals) == 1:
        return float(vals[0]), 0.0
    a = np.asarray(vals, dtype=np.float64)
    return float(np.mean(a)), float(np.std(a, ddof=0))


def _summarize_run(trace_path: str, run_id: str) -> Dict[str, Any]:
    rows = [r for r in load_traces(trace_path) if str(r.get("run_id", "")) == run_id]
    valid_rows = []
    rejected_count = 0
    constraint_reject_count = 0
    rewards = []
    mode_counts: Dict[str, int] = {}
    mode_names = {"0": "DENSE", "1": "SPARSE_ISECT", "2": "HIGHLY_SPARSE"}
    per_workload_raw: Dict[str, Dict[str, Any]] = {}
    per_shape_mode_raw: Dict[str, Dict[str, Dict[str, float]]] = {}
    perf_fields = (
        "accel_busy",
        "accel_compute",
        "accel_stall",
        "stall_dma_starvation",
        "stall_mmio_control",
        "pe_idle_cycles",
        "mem_arb_wait_cycles",
        "fifo_backpressure_cycles",
    )
    perf_sums: Dict[str, float] = {k: 0.0 for k in perf_fields}
    perf_counts: Dict[str, int] = {k: 0 for k in perf_fields}

    def _mode_key(row: Dict[str, Any]) -> str:
        mode = row.get("executed_hardware_dataflow_mode", row.get("hardware_dataflow_mode", -1))
        try:
            return str(int(mode))
        except Exception:
            return "-1"

    for r in rows:
        reward = float(r.get("reward", -10000.0))
        rewards.append(reward)
        mode_key = _mode_key(r)
        mode_counts[mode_key] = int(mode_counts.get(mode_key, 0) + 1)
        workload_tag = str(r.get("workload_tag", "unknown"))
        per_workload = per_workload_raw.setdefault(
            workload_tag,
            {
                "rows": 0,
                "valid_rows": 0,
                "sum_cycles": 0.0,
                "best_cycles": None,
                "mode_counts": {},
            },
        )
        per_workload["rows"] = int(per_workload["rows"]) + 1
        wm = per_workload["mode_counts"]
        wm[mode_key] = int(wm.get(mode_key, 0) + 1)

        reject_reason = r.get("reject_reason")
        has_reject = False
        if isinstance(reject_reason, str):
            rr = reject_reason.strip().lower()
            has_reject = rr not in ("", "none")
        elif reject_reason is not None:
            has_reject = True
        if has_reject:
            rejected_count += 1
        violations = r.get("constraint_violations")
        if not isinstance(violations, list):
            violations = []
        if violations or int(r.get("constraint_ok", 1)) == 0:
            constraint_reject_count += 1
        is_valid = int(r.get("is_valid", 0))
        if is_valid == 0:
            is_valid = int(r.get("simulated", 0)) == 1 and int(r.get("correctness_passed", 0)) == 1 and int(r.get("cycles", -1)) > 0
        if is_valid:
            valid_rows.append(r)
            cycles = int(r.get("cycles", -1))
            if cycles > 0:
                per_workload["valid_rows"] = int(per_workload["valid_rows"]) + 1
                per_workload["sum_cycles"] = float(per_workload["sum_cycles"]) + float(cycles)
                best_cycles = per_workload.get("best_cycles")
                if best_cycles is None or cycles < int(best_cycles):
                    per_workload["best_cycles"] = int(cycles)
                shape_sig = str(r.get("shape_signature", "unknown"))
                shape_modes = per_shape_mode_raw.setdefault(shape_sig, {})
                mode_stats = shape_modes.setdefault(mode_key, {"sum_cycles": 0.0, "count": 0.0})
                mode_stats["sum_cycles"] = float(mode_stats["sum_cycles"]) + float(cycles)
                mode_stats["count"] = float(mode_stats["count"]) + 1.0
            for field in perf_fields:
                val = r.get(field)
                try:
                    v = float(val)
                except Exception:
                    continue
                if v >= 0.0:
                    perf_sums[field] += v
                    perf_counts[field] += 1

    cycles = [int(r.get("cycles", -1)) for r in valid_rows if int(r.get("cycles", -1)) > 0]
    best_cycles = min(cycles) if cycles else -1
    avg_valid_cycles = (sum(cycles) / len(cycles)) if cycles else -1.0
    avg_valid_reward = (
        sum(float(r.get("reward", -10000.0)) for r in valid_rows) / len(valid_rows)
        if valid_rows else -10000.0
    )
    avg_reward = (sum(rewards) / len(rewards)) if rewards else -10000.0

    per_workload: Dict[str, Any] = {}
    for workload_tag, raw in per_workload_raw.items():
        wr = int(raw.get("rows", 0))
        wv = int(raw.get("valid_rows", 0))
        wbest = int(raw["best_cycles"]) if raw.get("best_cycles") is not None else -1
        wavg = float(raw["sum_cycles"] / wv) if wv > 0 else -1.0
        per_workload[workload_tag] = {
            "rows": wr,
            "valid_rows": wv,
            "valid_ratio": float(wv / wr) if wr > 0 else -1.0,
            "best_cycles": wbest,
            "avg_valid_cycles": wavg,
            "mode_counts": raw.get("mode_counts", {}),
        }

    per_shape_mode_choice: Dict[str, Any] = {}
    for shape_sig, shape_modes in per_shape_mode_raw.items():
        best_mode = -1
        best_avg = None
        dense_avg = None
        sparse_avg = None
        dense_count = 0
        sparse_count = 0
        for mode_key, stats in shape_modes.items():
            count = float(stats.get("count", 0.0))
            if count <= 0:
                continue
            avg_cycles = float(stats.get("sum_cycles", 0.0) / count)
            if str(mode_key) == "0":
                dense_avg = avg_cycles
                dense_count = int(count)
            elif str(mode_key) == "1":
                sparse_avg = avg_cycles
                sparse_count = int(count)
            if best_avg is None or avg_cycles < best_avg:
                best_avg = avg_cycles
                try:
                    best_mode = int(mode_key)
                except Exception:
                    best_mode = -1
        sparse_over_dense = (
            float(sparse_avg / dense_avg)
            if dense_avg is not None and sparse_avg is not None and dense_avg > 0.0
            else -1.0
        )
        per_shape_mode_choice[shape_sig] = {
            "preferred_mode": int(best_mode),
            "preferred_mode_name": mode_names.get(str(best_mode), "UNKNOWN"),
            "avg_cycles": float(best_avg) if best_avg is not None else -1.0,
            "dense_avg_cycles": float(dense_avg) if dense_avg is not None else -1.0,
            "dense_valid_rows": int(dense_count),
            "sparse_avg_cycles": float(sparse_avg) if sparse_avg is not None else -1.0,
            "sparse_valid_rows": int(sparse_count),
            "sparse_over_dense_cycle_ratio": float(sparse_over_dense),
        }

    perf_summary: Dict[str, float] = {}
    for field in perf_fields:
        n = int(perf_counts.get(field, 0))
        perf_summary[f"avg_{field}_valid"] = float(perf_sums[field] / n) if n > 0 else -1.0

    return {
        "total_rows": len(rows),
        "valid_simulated_rows": len(valid_rows),
        "rejected_rows": int(rejected_count),
        "constraint_reject_rows": int(constraint_reject_count),
        "best_cycles": int(best_cycles),
        "avg_valid_cycles": float(avg_valid_cycles),
        "avg_valid_reward": float(avg_valid_reward),
        "avg_reward_all_rows": float(avg_reward),
        "mode_counts": mode_counts,
        "mode_names": mode_names,
        "per_workload": per_workload,
        "per_shape_mode_choice": per_shape_mode_choice,
        "perf_summary": perf_summary,
    }

class BKMLedgerCallback(BaseCallback):
    """
    Custom Stable Baselines3 callback that hooks into the RL training loop.
    After every episode, it checks if the achieved IPC beats the current
    record in the BKM Ledger for this specific layer geometry. If so, it
    commits an atomic write to update the compiler's database.
    """
    def __init__(self, ledger_path: str, verbose=0):
        super().__init__(verbose)
        self.ledger = BKMLedger(filepath=ledger_path)
        self.best_cycles = float('inf')

    def _on_step(self) -> bool:
        # Check if an episode just finished
        if self.locals.get("dones") is not None and self.locals["dones"][0]:
            info = self.locals.get("infos")[0]
            
            # Only update if the math was correct and cycles were reported
            if "error" not in info and "cycles" in info:
                cycles = info["cycles"]
                
                # Calculate IPC roughly (Commands / Cycles). 
                # For RL ranking purposes, lower cycles = higher IPC proxy.
                # Assuming ~3500 ops for this dummy calculation, but we'll 
                # store the exact IPC proxy.
                ipc_proxy = 3500.0 / cycles
                
                if cycles < self.best_cycles:
                    self.best_cycles = cycles
                    
                    # Decode action back to physical hardware shapes
                    action = self.locals["actions"][0]
                    
                    # Decode action back to physical hardware shapes
                    env = self.training_env.envs[0].env
                    dec_tile_m, dec_tile_n, dec_burst_size, dec_prefetch_depth, dec_tile_b, dec_hw_mode = env._decode_action(action)
                    tile_m = int(info.get("tile_m", dec_tile_m))
                    tile_n = int(info.get("tile_n", dec_tile_n))
                    burst_size = int(info.get("burst_size", dec_burst_size))
                    prefetch_depth = int(info.get("prefetch_depth", dec_prefetch_depth))
                    tile_b = int(info.get("tile_b", dec_tile_b))
                    hw_mode = int(info.get("hardware_dataflow_mode", dec_hw_mode))
                    M = int(info.get("M", env.M))
                    N = int(info.get("N", env.N))
                    K = int(info.get("K", env.K))
                    activation = int(info.get("activation", 1))
                    workload_tag = str(info.get("workload_tag", "generic"))
                    sparsity_pct = int(info.get("sparsity_pct", 0))
                    sparsity_bucket = int(info.get("sparsity_bucket", max(0, min(10, sparsity_pct // 10))))

                    updated = self.ledger.update_if_better(
                        M=M, N=N, K=K, activation=activation,
                        tile_m=tile_m, tile_n=tile_n,
                        burst_size=burst_size, prefetch_depth=prefetch_depth,
                        tile_b=tile_b, hardware_dataflow_mode=hw_mode, ipc=ipc_proxy,
                        workload_tag=workload_tag, sparsity_bucket=sparsity_bucket
                    )

                    if updated and self.verbose > 0:
                        hw_names = ["DENSE", "SPARSE_ISECT", "HIGHLY_SPARSE"]
                        print(f"\n[RL Daemon] NEW RECORD for {M}x{N}x{K} ({workload_tag}, sp{10*sparsity_bucket}+)!")
                        print(f"            Cycles: {cycles} | Tiling: [{tile_m}x{tile_n}] | "
                              f"Burst: {burst_size}B | Prefetch: {prefetch_depth} | tile_b: {tile_b} | "
                              f"hw_mode: {hw_mode}({hw_names[hw_mode]})")
                        print(f"            Atomic Ledger Update Successful.")
                        
        return True


class SurrogateUpdateCallback(BaseCallback):
    """
    Periodically retrains the surrogate model from the accumulated replay trace.
    """
    def __init__(self, trace_path: str, model_path: str,
                 retrain_every_steps: int = 64,
                 min_records: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.trace_path = trace_path
        self.model_path = model_path
        self.retrain_every_steps = max(1, int(retrain_every_steps))
        self.min_records = max(1, int(min_records))

    def _on_step(self) -> bool:
        if (self.n_calls % self.retrain_every_steps) != 0:
            return True
        metrics = SurrogateModel.train_from_trace(
            trace_path=self.trace_path,
            model_path=self.model_path,
            min_records=self.min_records,
        )
        if metrics is not None and self.verbose > 0:
            print(
                f"[Surrogate] retrained n_train={metrics.n_train} n_val={metrics.n_val} "
                f"mae_cycles={metrics.mae_cycles:.1f} mape={metrics.mape:.3f}"
            )
        return True


class EntropyAnnealCallback(BaseCallback):
    """
    Linearly anneals PPO entropy coefficient from start->end across this run.
    """
    def __init__(self, start_coef: float, end_coef: float,
                 total_steps: int, log_every_steps: int = 2048,
                 verbose: int = 0):
        super().__init__(verbose)
        self.start_coef = float(start_coef)
        self.end_coef = float(end_coef)
        self.total_steps = max(1, int(total_steps))
        self.log_every_steps = max(1, int(log_every_steps))

    def _on_step(self) -> bool:
        frac = min(1.0, float(self.n_calls) / float(self.total_steps))
        ent_coef = self.start_coef + (self.end_coef - self.start_coef) * frac
        self.model.ent_coef = float(ent_coef)
        if self.verbose > 0 and (self.n_calls % self.log_every_steps) == 0:
            print(f"[Entropy] step={self.n_calls}/{self.total_steps} ent_coef={ent_coef:.6f}")
        return True


class PeriodicCheckpointCallback(BaseCallback):
    """
    Periodically saves PPO weights during long exploration runs.
    This keeps campaigns resumable even if interrupted mid-run.
    """

    def __init__(self, model_path: str, every_steps: int, keep_last: int = 3, verbose: int = 0):
        super().__init__(verbose)
        self.model_path = model_path
        self.every_steps = max(0, int(every_steps))
        self.keep_last = max(1, int(keep_last))
        self.saved_roots: List[str] = []

    def _on_step(self) -> bool:
        if self.every_steps <= 0:
            return True
        if (self.n_calls % self.every_steps) != 0:
            return True
        step = int(self.num_timesteps)
        ckpt_root = f"{self.model_path}.step_{step}"
        try:
            self.model.save(ckpt_root)
            self.saved_roots.append(ckpt_root)
            # Keep latest N periodic checkpoints.
            while len(self.saved_roots) > self.keep_last:
                old = self.saved_roots.pop(0)
                old_zip = old if old.endswith(".zip") else (old + ".zip")
                if os.path.exists(old_zip):
                    os.remove(old_zip)
            if self.verbose > 0:
                print(f"[Checkpoint] saved {ckpt_root}.zip")
        except Exception as exc:
            if self.verbose > 0:
                print(f"[Checkpoint] save failed at step={step}: {exc}")
        return True


def run_daemon(total_timesteps: int = 10_000_000,
               target_M: int = 128,
               target_N: int = 128,
               target_K: int = 784,
               workspace_dir: str | None = None,
               fresh_start: bool = False,
               ppo_n_steps: int = 64,
               ppo_batch_size: int = 16,
               workload_diversity: bool = True,
               workload_selector: str = "all",
               train_shape_split: str = "all",
               include_workload_features: bool = True,
               workload_aware_lookup: bool = True,
               trace_path: str = "",
               surrogate_model_path: str = "",
               surrogate_enabled: bool = False,
               surrogate_verify_every: int = 10,
               surrogate_retrain_every: int = 64,
               surrogate_min_records: int = 200,
               surrogate_trust_min_train: int = 200,
               surrogate_trust_max_mape: float = 0.35,
               reward_normalization: bool = True,
               reward_norm_warmup: int = 32,
               reward_norm_clip: float = 6.0,
               entropy_coef_start: float = 0.05,
               entropy_coef_end: float = 0.005,
               eval_workload_selector: str = "",
               eval_episodes: int = 0,
               eval_shape_split: str = "all",
               eval_output_path: str = "",
               eval_only: bool = False,
               eval_repeats: int = 1,
               eval_deterministic: bool = True,
               global_seed: int = 42,
               checkpoint_path: str = "",
               force_hw_mode: int = -1,
               allow_eval_train_overlap: bool = False,
               run_id: str = "",
               run_command: str = "",
               campaign_id: str = "",
               campaign_stage: str = "",
               targeted_campaign_config: str = "",
               targeted_sample_prob: float = 0.0,
               enable_tile_b4: bool = False,
               checkpoint_every_steps: int = 0,
               checkpoint_keep_last: int = 3,
               export_parquet_dir: str = "",
               export_parquet_append: bool = True,
               parquet_partition_cols: str = "workload_tag,sparsity_bucket,run_date_utc") -> List[Dict[str, Any]]:
    """
    The main background worker that continuously explores the hardware design space.
    """
    print("==================================================")
    print("  Systolic Array RL Optimization Daemon Starting  ")
    print("==================================================")
    
    if workspace_dir is None:
        workspace_dir = _default_workspace()
    _set_global_seeds(global_seed)
    train_workload_families = parse_workload_selector(workload_selector)
    parse_shape_split(train_shape_split)
    if eval_only and not eval_workload_selector:
        eval_workload_selector = workload_selector
    if eval_only and eval_episodes <= 0:
        eval_episodes = 20
    eval_workload_families: List[str] = []
    overlap_families: List[str] = []
    eval_train_overlap_detected = False
    if eval_workload_selector:
        eval_workload_families = parse_workload_selector(eval_workload_selector)
        parse_shape_split(eval_shape_split)
        overlap_families = _selector_overlap(workload_selector, eval_workload_selector)
        eval_train_overlap_detected = (
            (not bool(eval_only))
            and bool(overlap_families)
            and str(train_shape_split) == str(eval_shape_split)
        )
        if eval_train_overlap_detected and not bool(allow_eval_train_overlap):
            raise ValueError(
                "train/eval overlap detected: same shape split with overlapping workload families. "
                "Use disjoint splits/selectors or pass --allow-eval-train-overlap for intentional overlap."
            )
    _configure_lookup_mode(workload_aware_lookup)
    _configure_forced_hw_mode(force_hw_mode)
    run_id = run_id.strip() or _new_run_id()
    if not trace_path:
        trace_path = os.path.join(workspace_dir, "data", "traces", f"{run_id}.jsonl")
    if not surrogate_model_path:
        surrogate_model_path = os.path.join(workspace_dir, "data", "surrogate_model.pt")
    trace_path = os.path.abspath(trace_path)
    targeted_cfg = _load_targeted_campaign_config(targeted_campaign_config) if str(targeted_campaign_config).strip() else {
        "path": "",
        "targets": [],
        "summary": {},
        "selection": {},
        "weights": {},
    }
    targeted_specs: List[Dict[str, Any]] = list(targeted_cfg.get("targets", []))
    run_mode = "eval" if eval_only else "train"
    campaign_id = str(campaign_id or "").strip()
    campaign_stage = str(campaign_stage or "").strip()
    campaign_meta = {
        "campaign_id": campaign_id,
        "campaign_stage": campaign_stage,
        "run_mode": run_mode,
    }

    if not run_command:
        run_command = f"{shlex.quote(sys.executable)} auto_tuner/rl_daemon.py --timesteps {int(total_timesteps)}"
    runs_dir = os.path.join(workspace_dir, "data", "runs")
    manifest_path = os.path.join(runs_dir, f"{run_id}.json")
    run_manifest: Dict[str, Any] = {
        "run_id": run_id,
        "schema_version": 1,
        "command": run_command,
        "status": "running",
        "run_mode": run_mode,
        "started_at": _utc_now_iso(),
        "completed_at": "",
        "workspace": os.path.abspath(workspace_dir),
        "trace_path": trace_path,
        "workload": workload_selector,
        "train_workload_families": train_workload_families,
        "train_shape_split": train_shape_split,
        "timesteps": int(total_timesteps),
        "n_steps": int(ppo_n_steps),
        "batch_size": int(ppo_batch_size),
        "surrogate_enabled": int(bool(surrogate_enabled)),
        "surrogate_verify_every": int(surrogate_verify_every),
        "surrogate_min_records": int(surrogate_min_records),
        "surrogate_trust_min_train": int(surrogate_trust_min_train),
        "surrogate_trust_max_mape": float(surrogate_trust_max_mape),
        "reward_normalization": int(bool(reward_normalization)),
        "reward_norm_warmup": int(reward_norm_warmup),
        "reward_norm_clip": float(reward_norm_clip),
        "entropy_coef_start": float(entropy_coef_start),
        "entropy_coef_end": float(entropy_coef_end),
        "seed": int(global_seed),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_every_steps": int(max(0, int(checkpoint_every_steps))),
        "checkpoint_keep_last": int(max(1, int(checkpoint_keep_last))),
        "forced_hw_mode": int(force_hw_mode),
        "campaign_id": campaign_id,
        "campaign_stage": campaign_stage,
        "targeted_campaign_config": str(targeted_cfg.get("path", "")),
        "targeted_sample_prob": float(max(0.0, min(1.0, float(targeted_sample_prob)))),
        "targeted_target_count": int(len(targeted_specs)),
        "targeted_campaign_summary": targeted_cfg.get("summary", {}),
        "targeted_campaign_selection": targeted_cfg.get("selection", {}),
        "targeted_campaign_weights": targeted_cfg.get("weights", {}),
        "enable_tile_b4": int(bool(enable_tile_b4)),
        "exploration_space_contract": SystolicEnv.exploration_space_contract(enable_tile_b4=bool(enable_tile_b4)),
        "export_parquet_dir": str(export_parquet_dir or ""),
        "export_parquet_append": int(bool(export_parquet_append)),
        "parquet_partition_cols": str(parquet_partition_cols),
        "eval_workloads": eval_workload_selector,
        "eval_workload_families": eval_workload_families,
        "eval_episodes": int(eval_episodes),
        "eval_repeats": int(eval_repeats),
        "eval_deterministic": int(bool(eval_deterministic)),
        "eval_shape_split": eval_shape_split,
        "eval_train_overlap_detected": int(bool(eval_train_overlap_detected)),
        "eval_train_overlap_families": overlap_families,
        "allow_eval_train_overlap": int(bool(allow_eval_train_overlap)),
        "policy_saved": False,
        "error": "",
    }
    _write_json_atomic(manifest_path, run_manifest)

    # Clean once at daemon start so obj_dir is generated by the current
    # toolchain/container, avoiding stale cross-host Verilator paths.
    subprocess.run(
        ["make", "clean"],
        cwd=workspace_dir,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    # 1. Initialize the Environment for a specific target layer size
    env = SystolicEnv(M_target=target_M, N_target=target_N, K_target=target_K,
                      workspace_dir=workspace_dir,
                      workload_diversity=workload_diversity,
                      workload_selector=workload_selector,
                      shape_split=train_shape_split,
                      include_workload_features=include_workload_features,
                      trace_path=trace_path,
                      run_id=run_id,
                      surrogate_enabled=surrogate_enabled,
                      surrogate_model_path=surrogate_model_path,
                      surrogate_verify_every=surrogate_verify_every,
                      surrogate_min_train=surrogate_trust_min_train,
                      surrogate_max_mape=surrogate_trust_max_mape,
                      reward_normalization=reward_normalization,
                      reward_norm_warmup=reward_norm_warmup,
                      reward_norm_clip=reward_norm_clip,
                      campaign_metadata=campaign_meta,
                      targeted_specs=targeted_specs,
                      targeted_sample_prob=targeted_sample_prob,
                      enable_tile_b4=enable_tile_b4)
    
    # Paths
    ledger_path = os.path.join(os.path.dirname(__file__), "compiler", "bkm_ledger.json")
    default_ckpt_name = "ppo_systolic_agent_tileb4_latest" if bool(enable_tile_b4) else "ppo_systolic_agent_latest"
    model_path = checkpoint_path.strip() or os.path.join(os.path.dirname(__file__), default_ckpt_name)
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]
    override_path = os.path.join(workspace_dir, "auto_tuner", "rl_override.json")

    # Preserve any user-provided override and restore it on exit.
    prior_override_bytes = None
    if os.path.exists(override_path):
        with open(override_path, "rb") as f:
            prior_override_bytes = f.read()
    
    # 2. Setup the Callback
    entropy_callback = EntropyAnnealCallback(
        start_coef=entropy_coef_start,
        end_coef=entropy_coef_end,
        total_steps=max(1, int(total_timesteps)),
        verbose=1,
    )
    bkm_callback = BKMLedgerCallback(ledger_path=ledger_path, verbose=1)
    callbacks: list[BaseCallback] = [entropy_callback, bkm_callback]
    if int(checkpoint_every_steps) > 0:
        callbacks.append(
            PeriodicCheckpointCallback(
                model_path=model_path,
                every_steps=int(checkpoint_every_steps),
                keep_last=int(checkpoint_keep_last),
                verbose=1,
            )
        )
    if surrogate_enabled:
        callbacks.append(
            SurrogateUpdateCallback(
                trace_path=trace_path,
                model_path=surrogate_model_path,
                retrain_every_steps=surrogate_retrain_every,
                min_records=surrogate_min_records,
                verbose=1,
            )
        )
    train_callback: BaseCallback = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)
    
    # 3. Warm-Start or Initialize Model
    # PPO requires a smaller n_steps buffer for slow physical environments
    # like Verilator. Default 2048 takes hours before a single weight update.
    # We use 64 to learn rapidly from the simulator hook.
    if (not fresh_start) and os.path.exists(model_path + ".zip"):
        print(f"Loading existing policy from {model_path}.zip...")
        try:
            model = PPO.load(model_path, env=env)
            model.set_random_seed(global_seed)
            model.ent_coef = float(entropy_coef_start)
        except Exception as exc:
            print(f"[RL Daemon] Existing checkpoint incompatible ({exc}). Starting fresh policy.")
            model = PPO("MlpPolicy", env, verbose=1,
                        n_steps=ppo_n_steps, batch_size=ppo_batch_size, seed=global_seed,
                        ent_coef=float(entropy_coef_start))
    else:
        print("Initializing new PPO policy...")
        model = PPO("MlpPolicy", env, verbose=1,
                    n_steps=ppo_n_steps, batch_size=ppo_batch_size, seed=global_seed,
                    ent_coef=float(entropy_coef_start))

    eval_rows: List[Dict[str, Any]] = []
    run_status = "completed"
    run_error = ""
    pending_exc: Exception | None = None
    policy_saved = False

    # 4. The Infinite Training Loop with Graceful Exit
    if eval_only:
        print("\nEval-only mode: skipping training updates and running fixed evaluation suite.")
    else:
        print("\nBeginning exploration. Press Ctrl+C to save policy and exit safely.")
    try:
        if not eval_only:
            # Run practically forever (10M timesteps) in the background
            model.learn(total_timesteps=total_timesteps, callback=train_callback, reset_num_timesteps=False)
    except KeyboardInterrupt:
        run_status = "interrupted"
        print("\n\n[RL Daemon] KeyboardInterrupt caught! Initiating graceful shutdown...")
    except Exception as exc:
        run_status = "failed"
        run_error = str(exc)
        pending_exc = exc
    finally:
        if run_status != "failed" and eval_episodes > 0 and eval_workload_selector:
            repeats = max(1, int(eval_repeats))
            print(
                f"[RL Daemon] Running eval: selector={eval_workload_selector} split={eval_shape_split} "
                f"episodes={eval_episodes} repeats={repeats} deterministic={int(bool(eval_deterministic))} "
                f"seed={global_seed} force_hw_mode={int(force_hw_mode)}"
            )
            by_shape_stats: Dict[tuple[str, str], Dict[str, Any]] = {}
            per_repeat_rows: List[Dict[str, Any]] = []

            for rep in range(repeats):
                seed_for_rep = int(global_seed if eval_deterministic else (global_seed + rep))
                eval_env = SystolicEnv(
                    M_target=target_M, N_target=target_N, K_target=target_K,
                    workspace_dir=workspace_dir,
                    workload_diversity=True,
                    workload_selector=eval_workload_selector,
                    shape_split=eval_shape_split,
                    include_workload_features=include_workload_features,
                    trace_path=trace_path,
                    run_id=run_id,
                    surrogate_enabled=False,
                    surrogate_model_path=surrogate_model_path,
                    surrogate_min_train=surrogate_trust_min_train,
                    surrogate_max_mape=surrogate_trust_max_mape,
                    reward_normalization=reward_normalization,
                    reward_norm_warmup=reward_norm_warmup,
                    reward_norm_clip=reward_norm_clip,
                    campaign_metadata=campaign_meta,
                    targeted_specs=[],
                    targeted_sample_prob=0.0,
                    enable_tile_b4=enable_tile_b4,
                )
                by_shape = {}
                for ep in range(eval_episodes):
                    if ep == 0:
                        obs, _ = eval_env.reset(seed=seed_for_rep)
                    else:
                        obs, _ = eval_env.reset()
                    action, _ = model.predict(obs, deterministic=bool(eval_deterministic))
                    _, reward, _, _, info = eval_env.step(action)
                    tag = str(info.get("workload_tag", "unknown"))
                    shape_sig = str(info.get("shape_signature", "unknown"))
                    k = (tag, shape_sig)
                    rec = by_shape.setdefault(
                        k,
                        {
                            "episodes": 0,
                            "sum_reward": 0.0,
                            "sum_cycles": 0,
                            "valid": 0,
                            "mode_counts": {},
                            "valid_mode_counts": {},
                            "sum_cycles_by_mode": {},
                            "sum_layer_m": 0.0,
                            "sum_layer_n": 0.0,
                            "sum_layer_k": 0.0,
                            "layer_dim_samples": 0,
                            "sum_tile_m": 0.0,
                            "sum_tile_n": 0.0,
                            "sum_burst_size": 0.0,
                            "sum_prefetch_depth": 0.0,
                            "sum_tile_b": 0.0,
                            "schedule_samples": 0,
                            "sum_mode_fallback": 0.0,
                            "mode_fallback_samples": 0,
                            "sum_pe_util": 0.0,
                            "pe_util_count": 0,
                            "sum_dense_equiv_macs_per_cycle": 0.0,
                            "dense_equiv_macs_per_cycle_count": 0,
                            "sum_sparsity_pct": 0.0,
                            "sparsity_samples": 0,
                            "sum_sparsity_bucket": 0.0,
                            "sparsity_bucket_samples": 0,
                        },
                    )
                    rec["episodes"] += 1
                    rec["sum_reward"] += float(reward)
                    mode = int(info.get("executed_hardware_dataflow_mode", info.get("hardware_dataflow_mode", -1)))
                    rec["mode_counts"][mode] = int(rec["mode_counts"].get(mode, 0) + 1)
                    m_dim = info.get("M")
                    n_dim = info.get("N")
                    k_dim = info.get("K")
                    if isinstance(m_dim, (int, float)) and isinstance(n_dim, (int, float)) and isinstance(k_dim, (int, float)):
                        rec["sum_layer_m"] += float(m_dim)
                        rec["sum_layer_n"] += float(n_dim)
                        rec["sum_layer_k"] += float(k_dim)
                        rec["layer_dim_samples"] += 1
                    tile_m = info.get("tile_m")
                    tile_n = info.get("tile_n")
                    burst_size = info.get("burst_size")
                    prefetch_depth = info.get("prefetch_depth")
                    tile_b = info.get("tile_b")
                    if (
                        isinstance(tile_m, (int, float))
                        and isinstance(tile_n, (int, float))
                        and isinstance(burst_size, (int, float))
                        and isinstance(prefetch_depth, (int, float))
                        and isinstance(tile_b, (int, float))
                    ):
                        rec["sum_tile_m"] += float(tile_m)
                        rec["sum_tile_n"] += float(tile_n)
                        rec["sum_burst_size"] += float(burst_size)
                        rec["sum_prefetch_depth"] += float(prefetch_depth)
                        rec["sum_tile_b"] += float(tile_b)
                        rec["schedule_samples"] += 1
                    mode_fallback = info.get("mode_fallback")
                    if isinstance(mode_fallback, (int, float)):
                        rec["sum_mode_fallback"] += float(mode_fallback)
                        rec["mode_fallback_samples"] += 1
                    sparsity_pct = info.get("sparsity_pct")
                    if isinstance(sparsity_pct, (int, float)):
                        rec["sum_sparsity_pct"] += float(sparsity_pct)
                        rec["sparsity_samples"] += 1
                    sparsity_bucket = info.get("sparsity_bucket")
                    if isinstance(sparsity_bucket, (int, float)):
                        rec["sum_sparsity_bucket"] += float(sparsity_bucket)
                        rec["sparsity_bucket_samples"] += 1
                    if "cycles" in info:
                        cycles = int(info["cycles"])
                        if cycles > 0:
                            rec["sum_cycles"] += cycles
                            rec["valid"] += 1
                            rec["valid_mode_counts"][mode] = int(rec["valid_mode_counts"].get(mode, 0) + 1)
                            rec["sum_cycles_by_mode"][mode] = float(rec["sum_cycles_by_mode"].get(mode, 0.0) + float(cycles))
                            pe_util = info.get("pe_util_est")
                            if isinstance(pe_util, (int, float)) and float(pe_util) >= 0.0:
                                rec["sum_pe_util"] += float(pe_util)
                                rec["pe_util_count"] += 1
                            mpc = info.get("dense_equiv_macs_per_cycle")
                            if isinstance(mpc, (int, float)) and float(mpc) > 0.0:
                                rec["sum_dense_equiv_macs_per_cycle"] += float(mpc)
                                rec["dense_equiv_macs_per_cycle_count"] += 1

                for (tag, shape_sig), rec in by_shape.items():
                    avg_reward = rec["sum_reward"] / max(1, rec["episodes"])
                    avg_cycles = float(rec["sum_cycles"] / rec["valid"]) if rec["valid"] > 0 else -1.0
                    valid_ratio = float(rec["valid"] / max(1, rec["episodes"]))
                    mode0_count = int(rec["mode_counts"].get(0, 0))
                    mode1_count = int(rec["mode_counts"].get(1, 0))
                    mode0_share = float(mode0_count / max(1, rec["episodes"]))
                    mode1_share = float(mode1_count / max(1, rec["episodes"]))
                    layer_m_mean = (
                        float(rec["sum_layer_m"] / rec["layer_dim_samples"])
                        if int(rec["layer_dim_samples"]) > 0
                        else -1.0
                    )
                    layer_n_mean = (
                        float(rec["sum_layer_n"] / rec["layer_dim_samples"])
                        if int(rec["layer_dim_samples"]) > 0
                        else -1.0
                    )
                    layer_k_mean = (
                        float(rec["sum_layer_k"] / rec["layer_dim_samples"])
                        if int(rec["layer_dim_samples"]) > 0
                        else -1.0
                    )
                    tile_m_mean = (
                        float(rec["sum_tile_m"] / rec["schedule_samples"])
                        if int(rec["schedule_samples"]) > 0
                        else -1.0
                    )
                    tile_n_mean = (
                        float(rec["sum_tile_n"] / rec["schedule_samples"])
                        if int(rec["schedule_samples"]) > 0
                        else -1.0
                    )
                    burst_size_mean = (
                        float(rec["sum_burst_size"] / rec["schedule_samples"])
                        if int(rec["schedule_samples"]) > 0
                        else -1.0
                    )
                    prefetch_depth_mean = (
                        float(rec["sum_prefetch_depth"] / rec["schedule_samples"])
                        if int(rec["schedule_samples"]) > 0
                        else -1.0
                    )
                    tile_b_mean = (
                        float(rec["sum_tile_b"] / rec["schedule_samples"])
                        if int(rec["schedule_samples"]) > 0
                        else -1.0
                    )
                    mode_fallback_rate = (
                        float(rec["sum_mode_fallback"] / rec["mode_fallback_samples"])
                        if int(rec["mode_fallback_samples"]) > 0
                        else -1.0
                    )
                    mode0_valid = int(rec["valid_mode_counts"].get(0, 0))
                    mode1_valid = int(rec["valid_mode_counts"].get(1, 0))
                    dense_cycles_mean = (
                        float(rec["sum_cycles_by_mode"].get(0, 0.0) / mode0_valid) if mode0_valid > 0 else -1.0
                    )
                    sparse_cycles_mean = (
                        float(rec["sum_cycles_by_mode"].get(1, 0.0) / mode1_valid) if mode1_valid > 0 else -1.0
                    )
                    sparse_over_dense_ratio = (
                        float(sparse_cycles_mean / dense_cycles_mean)
                        if dense_cycles_mean > 0.0 and sparse_cycles_mean > 0.0
                        else -1.0
                    )
                    pe_util_mean = (
                        float(rec["sum_pe_util"] / rec["pe_util_count"]) if int(rec["pe_util_count"]) > 0 else -1.0
                    )
                    mpc_mean = (
                        float(rec["sum_dense_equiv_macs_per_cycle"] / rec["dense_equiv_macs_per_cycle_count"])
                        if int(rec["dense_equiv_macs_per_cycle_count"]) > 0
                        else -1.0
                    )
                    sparsity_pct_mean = (
                        float(rec["sum_sparsity_pct"] / rec["sparsity_samples"])
                        if int(rec["sparsity_samples"]) > 0
                        else -1.0
                    )
                    sparsity_bucket_mean = (
                        float(rec["sum_sparsity_bucket"] / rec["sparsity_bucket_samples"])
                        if int(rec["sparsity_bucket_samples"]) > 0
                        else -1.0
                    )
                    per_repeat_row = {
                        "repeat_idx": int(rep),
                        "seed": int(seed_for_rep),
                        "workload_family": tag,
                        "shape_signature": shape_sig,
                        "episodes": int(rec["episodes"]),
                        "valid_runs": int(rec["valid"]),
                        "valid_ratio": float(valid_ratio),
                        "avg_reward": float(avg_reward),
                        "avg_cycles": float(avg_cycles),
                        "layer_m_mean": float(layer_m_mean),
                        "layer_n_mean": float(layer_n_mean),
                        "layer_k_mean": float(layer_k_mean),
                        "tile_m_mean": float(tile_m_mean),
                        "tile_n_mean": float(tile_n_mean),
                        "burst_size_mean": float(burst_size_mean),
                        "prefetch_depth_mean": float(prefetch_depth_mean),
                        "tile_b_mean": float(tile_b_mean),
                        "mode_fallback_rate": float(mode_fallback_rate),
                        "mode0_share": float(mode0_share),
                        "mode1_share": float(mode1_share),
                        "mode0_valid_avg_cycles": float(dense_cycles_mean),
                        "mode1_valid_avg_cycles": float(sparse_cycles_mean),
                        "sparse_over_dense_cycle_ratio": float(sparse_over_dense_ratio),
                        "pe_util_mean": float(pe_util_mean),
                        "dense_equiv_macs_per_cycle_mean": float(mpc_mean),
                        "sparsity_pct_mean": float(sparsity_pct_mean),
                        "sparsity_bucket_mean": float(sparsity_bucket_mean),
                    }
                    per_repeat_rows.append(per_repeat_row)

                    agg = by_shape_stats.setdefault((tag, shape_sig), {
                        "episodes": int(rec["episodes"]),
                        "avg_cycles": [],
                        "avg_reward": [],
                        "valid_ratio": [],
                        "layer_m_mean": [],
                        "layer_n_mean": [],
                        "layer_k_mean": [],
                        "tile_m_mean": [],
                        "tile_n_mean": [],
                        "burst_size_mean": [],
                        "prefetch_depth_mean": [],
                        "tile_b_mean": [],
                        "mode_fallback_rate": [],
                        "mode0_share": [],
                        "mode1_share": [],
                        "mode0_valid_avg_cycles": [],
                        "mode1_valid_avg_cycles": [],
                        "sparse_over_dense_cycle_ratio": [],
                        "pe_util_mean": [],
                        "dense_equiv_macs_per_cycle_mean": [],
                        "sparsity_pct_mean": [],
                        "sparsity_bucket_mean": [],
                    })
                    agg["episodes"] = int(rec["episodes"])
                    agg["avg_reward"].append(float(avg_reward))
                    agg["valid_ratio"].append(float(valid_ratio))
                    if avg_cycles > 0:
                        agg["avg_cycles"].append(float(avg_cycles))
                    if layer_m_mean > 0:
                        agg["layer_m_mean"].append(float(layer_m_mean))
                    if layer_n_mean > 0:
                        agg["layer_n_mean"].append(float(layer_n_mean))
                    if layer_k_mean > 0:
                        agg["layer_k_mean"].append(float(layer_k_mean))
                    if tile_m_mean > 0:
                        agg["tile_m_mean"].append(float(tile_m_mean))
                    if tile_n_mean > 0:
                        agg["tile_n_mean"].append(float(tile_n_mean))
                    if burst_size_mean > 0:
                        agg["burst_size_mean"].append(float(burst_size_mean))
                    if prefetch_depth_mean > 0:
                        agg["prefetch_depth_mean"].append(float(prefetch_depth_mean))
                    if tile_b_mean > 0:
                        agg["tile_b_mean"].append(float(tile_b_mean))
                    if mode_fallback_rate >= 0:
                        agg["mode_fallback_rate"].append(float(mode_fallback_rate))
                    agg["mode0_share"].append(float(mode0_share))
                    agg["mode1_share"].append(float(mode1_share))
                    if dense_cycles_mean > 0:
                        agg["mode0_valid_avg_cycles"].append(float(dense_cycles_mean))
                    if sparse_cycles_mean > 0:
                        agg["mode1_valid_avg_cycles"].append(float(sparse_cycles_mean))
                    if sparse_over_dense_ratio > 0:
                        agg["sparse_over_dense_cycle_ratio"].append(float(sparse_over_dense_ratio))
                    if pe_util_mean >= 0:
                        agg["pe_util_mean"].append(float(pe_util_mean))
                    if mpc_mean > 0:
                        agg["dense_equiv_macs_per_cycle_mean"].append(float(mpc_mean))
                    if sparsity_pct_mean >= 0:
                        agg["sparsity_pct_mean"].append(float(sparsity_pct_mean))
                    if sparsity_bucket_mean >= 0:
                        agg["sparsity_bucket_mean"].append(float(sparsity_bucket_mean))

            eval_rows = []
            for (tag, shape_sig), agg in by_shape_stats.items():
                cycles_mean, cycles_std = _mean_std(agg["avg_cycles"])
                reward_mean, reward_std = _mean_std(agg["avg_reward"])
                valid_ratio_mean, valid_ratio_std = _mean_std(agg["valid_ratio"])
                layer_m_mean, _ = _mean_std(agg["layer_m_mean"])
                layer_n_mean, _ = _mean_std(agg["layer_n_mean"])
                layer_k_mean, _ = _mean_std(agg["layer_k_mean"])
                tile_m_mean, _ = _mean_std(agg["tile_m_mean"])
                tile_n_mean, _ = _mean_std(agg["tile_n_mean"])
                burst_size_mean, _ = _mean_std(agg["burst_size_mean"])
                prefetch_depth_mean, _ = _mean_std(agg["prefetch_depth_mean"])
                tile_b_mean, _ = _mean_std(agg["tile_b_mean"])
                mode_fallback_rate_mean, _ = _mean_std(agg["mode_fallback_rate"])
                mode0_share_mean, mode0_share_std = _mean_std(agg["mode0_share"])
                mode1_share_mean, mode1_share_std = _mean_std(agg["mode1_share"])
                dense_cycles_mean, dense_cycles_std = _mean_std(agg["mode0_valid_avg_cycles"])
                sparse_cycles_mean, sparse_cycles_std = _mean_std(agg["mode1_valid_avg_cycles"])
                sparse_dense_ratio_mean, sparse_dense_ratio_std = _mean_std(agg["sparse_over_dense_cycle_ratio"])
                pe_util_mean, pe_util_std = _mean_std(agg["pe_util_mean"])
                mpc_mean, mpc_std = _mean_std(agg["dense_equiv_macs_per_cycle_mean"])
                sparsity_pct_mean, sparsity_pct_std = _mean_std(agg["sparsity_pct_mean"])
                sparsity_bucket_mean, _ = _mean_std(agg["sparsity_bucket_mean"])
                row = {
                    "workload_family": tag,
                    "shape_signature": shape_sig,
                    "repeats": int(repeats),
                    "episodes_per_repeat": int(agg["episodes"]),
                    "cycles_mean": float(cycles_mean),
                    "cycles_std": float(cycles_std),
                    "reward_mean": float(reward_mean),
                    "reward_std": float(reward_std),
                    "valid_ratio_mean": float(valid_ratio_mean),
                    "valid_ratio_std": float(valid_ratio_std),
                    "layer_m": int(round(layer_m_mean)) if layer_m_mean > 0 else -1,
                    "layer_n": int(round(layer_n_mean)) if layer_n_mean > 0 else -1,
                    "layer_k": int(round(layer_k_mean)) if layer_k_mean > 0 else -1,
                    "tile_m_mean": float(tile_m_mean),
                    "tile_n_mean": float(tile_n_mean),
                    "burst_size_mean": float(burst_size_mean),
                    "prefetch_depth_mean": float(prefetch_depth_mean),
                    "tile_b_mean": float(tile_b_mean),
                    "mode_fallback_rate_mean": float(mode_fallback_rate_mean),
                    "mode0_share_mean": float(mode0_share_mean),
                    "mode0_share_std": float(mode0_share_std),
                    "mode1_share_mean": float(mode1_share_mean),
                    "mode1_share_std": float(mode1_share_std),
                    "mode0_valid_cycles_mean": float(dense_cycles_mean),
                    "mode0_valid_cycles_std": float(dense_cycles_std),
                    "mode1_valid_cycles_mean": float(sparse_cycles_mean),
                    "mode1_valid_cycles_std": float(sparse_cycles_std),
                    "sparse_over_dense_cycle_ratio_mean": float(sparse_dense_ratio_mean),
                    "sparse_over_dense_cycle_ratio_std": float(sparse_dense_ratio_std),
                    "pe_util_mean": float(pe_util_mean),
                    "pe_util_std": float(pe_util_std),
                    "dense_equiv_macs_per_cycle_mean": float(mpc_mean),
                    "dense_equiv_macs_per_cycle_std": float(mpc_std),
                    "sparsity_pct_mean": float(sparsity_pct_mean),
                    "sparsity_pct_std": float(sparsity_pct_std),
                    "sparsity_bucket": int(round(max(0.0, sparsity_bucket_mean))) if sparsity_bucket_mean >= 0 else -1,
                }
                eval_rows.append(row)
                print(
                    f"[EvalSummary] {tag} {shape_sig}: cycles_mean={cycles_mean:.1f} cycles_std={cycles_std:.1f} "
                    f"valid_ratio_mean={valid_ratio_mean:.3f} mode1_share={mode1_share_mean:.3f}"
                )

            if eval_output_path:
                out_dir = os.path.dirname(eval_output_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                with open(eval_output_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=[
                        "workload_family", "shape_signature", "repeats", "episodes_per_repeat",
                        "cycles_mean", "cycles_std", "reward_mean", "reward_std",
                        "valid_ratio_mean", "valid_ratio_std",
                        "layer_m", "layer_n", "layer_k",
                        "tile_m_mean", "tile_n_mean", "burst_size_mean",
                        "prefetch_depth_mean", "tile_b_mean", "mode_fallback_rate_mean",
                        "mode0_share_mean", "mode0_share_std",
                        "mode1_share_mean", "mode1_share_std",
                        "mode0_valid_cycles_mean", "mode0_valid_cycles_std",
                        "mode1_valid_cycles_mean", "mode1_valid_cycles_std",
                        "sparse_over_dense_cycle_ratio_mean", "sparse_over_dense_cycle_ratio_std",
                        "pe_util_mean", "pe_util_std",
                        "dense_equiv_macs_per_cycle_mean", "dense_equiv_macs_per_cycle_std",
                        "sparsity_pct_mean", "sparsity_pct_std",
                        "sparsity_bucket",
                    ])
                    w.writeheader()
                    for row in eval_rows:
                        w.writerow(row)

        # 5. Guaranteed Save
        print(f"[RL Daemon] Saving latest policy weights to {model_path}.zip...")
        try:
            model.save(model_path)
            policy_saved = True
        except Exception as exc:
            if run_status != "failed":
                run_status = "failed"
            run_error = run_error or f"model_save_failed: {exc}"

        if prior_override_bytes is not None:
            with open(override_path, "wb") as f:
                f.write(prior_override_bytes)
        elif os.path.exists(override_path):
            os.remove(override_path)

        run_manifest.update(_summarize_run(trace_path, run_id))
        run_manifest.update({
            "status": run_status,
            "error": run_error,
            "policy_saved": bool(policy_saved),
            "completed_at": _utc_now_iso(),
            "eval_rows": eval_rows,
        })
        if str(export_parquet_dir).strip():
            parquet_cmd = [
                sys.executable,
                os.path.join(workspace_dir, "auto_tuner", "trace_to_parquet.py"),
                "--workspace", workspace_dir,
                "--trace", trace_path,
                "--output-dir", str(export_parquet_dir),
                "--dataset-name", "trace_dataset_v1",
                "--partition-cols", str(parquet_partition_cols),
            ]
            if bool(export_parquet_append):
                parquet_cmd.append("--append")
            parquet_result = subprocess.run(
                parquet_cmd,
                cwd=workspace_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            parquet_meta_path = os.path.join(str(export_parquet_dir), "_dataset_meta.json")
            parquet_meta: Dict[str, Any] = {}
            if os.path.exists(parquet_meta_path):
                try:
                    with open(parquet_meta_path, "r", encoding="utf-8") as f:
                        loaded = json.load(f)
                    if isinstance(loaded, dict):
                        parquet_meta = loaded
                except Exception:
                    parquet_meta = {}
            run_manifest["parquet_export"] = {
                "command": " ".join(shlex.quote(x) for x in parquet_cmd),
                "returncode": int(parquet_result.returncode),
                "stdout": parquet_result.stdout.strip(),
                "stderr": parquet_result.stderr.strip(),
                "metadata": parquet_meta,
            }
        _write_json_atomic(manifest_path, run_manifest)
        print("[RL Daemon] Shutdown complete. BKM Ledger is safe.")
    if pending_exc is not None:
        raise pending_exc
    return eval_rows

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL daemon for inference_generic autotuning.")
    parser.add_argument("--timesteps", type=int, default=10_000_000,
                        help="Total PPO timesteps to train (default: 10,000,000).")
    parser.add_argument("--M", type=int, default=128, help="Target M dimension.")
    parser.add_argument("--N", type=int, default=128, help="Target N dimension.")
    parser.add_argument("--K", type=int, default=784, help="Target K dimension.")
    parser.add_argument("--workspace", type=str, default=_default_workspace(),
                        help="Accelera workspace root directory.")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore existing PPO checkpoint and start from scratch.")
    parser.add_argument("--n-steps", type=int, default=64,
                        help="PPO rollout steps before each update.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="PPO minibatch size.")
    parser.add_argument("--disable-workload-diversity", action="store_true",
                        help="Use only the fixed M/N/K target workload.")
    parser.add_argument("--workload", type=str, default="all",
                        help=("Training workload selector: all/* or CSV list using workload families/aliases. "
                              f"Valid tokens: {','.join(selector_tokens())}"))
    parser.add_argument("--train-shape-split", type=str, default="all",
                        help="Training shape split: all|train|test.")
    parser.add_argument("--disable-workload-features", action="store_true",
                        help="Ablation: remove workload-structure features from RL observation.")
    parser.add_argument("--disable-workload-aware-lookup", action="store_true",
                        help="Ablation: use generic ledger lookup key (ignore workload-tag/sparsity buckets).")
    parser.add_argument("--trace-path", type=str, default="",
                        help="Replay dataset path (JSONL).")
    parser.add_argument("--surrogate-model-path", type=str, default="",
                        help="Surrogate model checkpoint path.")
    parser.add_argument("--enable-surrogate", action="store_true",
                        help="Enable surrogate-assisted tuning (predicted steps between periodic simulations).")
    parser.add_argument("--surrogate-verify-every", type=int, default=10,
                        help="Run full simulator every N env steps when surrogate is enabled.")
    parser.add_argument("--surrogate-retrain-every", type=int, default=64,
                        help="Retrain surrogate every N RL steps.")
    parser.add_argument("--surrogate-min-records", type=int, default=200,
                        help="Minimum simulated trace rows before surrogate training starts.")
    parser.add_argument("--surrogate-trust-min-train", type=int, default=200,
                        help="Minimum surrogate training rows before using predicted-only steps.")
    parser.add_argument("--surrogate-trust-max-mape", type=float, default=0.35,
                        help="Maximum validation MAPE allowed for surrogate-only steps.")
    parser.add_argument("--disable-reward-normalization", action="store_true",
                        help="Use legacy raw reward instead of per-workload normalized reward.")
    parser.add_argument("--reward-norm-warmup", type=int, default=32,
                        help="Minimum samples per workload/shape before z-score reward normalization.")
    parser.add_argument("--reward-norm-clip", type=float, default=6.0,
                        help="Clip value for normalized reward z-score.")
    parser.add_argument("--entropy-coef-start", type=float, default=0.05,
                        help="Initial PPO entropy coefficient.")
    parser.add_argument("--entropy-coef-end", type=float, default=0.005,
                        help="Final PPO entropy coefficient at end of run.")
    parser.add_argument("--eval-workloads", type=str, default="",
                        help="Optional held-out workload selector for post-train eval.")
    parser.add_argument("--eval-episodes", type=int, default=0,
                        help="Held-out evaluation episodes to run on shutdown (requires --eval-workloads).")
    parser.add_argument("--eval-only", action="store_true",
                        help="Skip PPO training and run evaluation only.")
    parser.add_argument("--eval-repeats", type=int, default=1,
                        help="Number of repeated eval sweeps for mean/std reporting.")
    parser.add_argument("--stochastic-eval", action="store_true",
                        help="Use stochastic policy actions during eval (default deterministic).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed for Python/NumPy/Torch/PPO.")
    parser.add_argument("--checkpoint-path", type=str, default="",
                        help="Policy checkpoint path to load/save (default depends on tile_b stage: base -> ppo_systolic_agent_latest.zip, tile_b4 -> ppo_systolic_agent_tileb4_latest.zip).")
    parser.add_argument("--checkpoint-every-steps", type=int, default=0,
                        help="If >0, save periodic checkpoints every N RL callback steps.")
    parser.add_argument("--checkpoint-keep-last", type=int, default=3,
                        help="How many periodic checkpoints to keep (old ones removed).")
    parser.add_argument("--run-id", type=str, default="",
                        help="Optional explicit run_id for resumable campaigns.")
    parser.add_argument("--force-hw-mode", type=int, default=-1,
                        help="Force exporter hardware mode via ACCELERA_FORCE_HW_MODE (-1=off, 0=dense, 1=sparse mode1).")
    parser.add_argument("--campaign-id", type=str, default="",
                        help="Optional campaign identifier stored in run manifest and trace rows.")
    parser.add_argument("--campaign-stage", type=str, default="",
                        help="Optional campaign stage label (e.g. tranche_1, high_regret_pass).")
    parser.add_argument("--targeted-campaign-config", type=str, default="",
                        help="Optional JSON config from build_targeted_campaign.py.")
    parser.add_argument("--targeted-sample-prob", type=float, default=0.0,
                        help="Probability [0..1] of drawing from targeted candidate pool each episode.")
    parser.add_argument("--enable-tile-b4", action="store_true",
                        help="Stage-2 knob expansion: include tile_b=4 in RL action space.")
    parser.add_argument("--allow-eval-train-overlap", action="store_true",
                        help="Allow non-eval-only runs where train/eval selectors overlap on the same shape split.")
    parser.add_argument("--eval-shape-split", type=str, default="all",
                        help="Held-out evaluation shape split: all|train|test.")
    parser.add_argument("--eval-output-csv", type=str, default="",
                        help="Optional CSV path for eval rows.")
    parser.add_argument("--export-trace-parquet-dir", type=str, default="",
                        help="Optional Parquet dataset directory for post-run trace export.")
    parser.add_argument("--disable-parquet-append", action="store_true",
                        help="If set, Parquet export rewrites output dir instead of append mode.")
    parser.add_argument("--parquet-partition-cols", type=str, default="workload_tag,sparsity_bucket,run_date_utc",
                        help="Comma-separated partition columns for Parquet export.")
    args = parser.parse_args()
    run_cmd = shlex.quote(sys.executable) + " " + " ".join(shlex.quote(a) for a in sys.argv)

    run_daemon(total_timesteps=args.timesteps,
               target_M=args.M, target_N=args.N, target_K=args.K,
               workspace_dir=args.workspace,
               fresh_start=args.fresh,
               ppo_n_steps=args.n_steps,
               ppo_batch_size=args.batch_size,
               workload_diversity=(not args.disable_workload_diversity),
               workload_selector=args.workload,
               train_shape_split=args.train_shape_split,
               include_workload_features=(not args.disable_workload_features),
               workload_aware_lookup=(not args.disable_workload_aware_lookup),
               trace_path=args.trace_path,
               surrogate_model_path=args.surrogate_model_path,
               surrogate_enabled=args.enable_surrogate,
               surrogate_verify_every=args.surrogate_verify_every,
               surrogate_retrain_every=args.surrogate_retrain_every,
               surrogate_min_records=args.surrogate_min_records,
               surrogate_trust_min_train=args.surrogate_trust_min_train,
               surrogate_trust_max_mape=args.surrogate_trust_max_mape,
               reward_normalization=(not args.disable_reward_normalization),
               reward_norm_warmup=args.reward_norm_warmup,
               reward_norm_clip=args.reward_norm_clip,
               entropy_coef_start=args.entropy_coef_start,
               entropy_coef_end=args.entropy_coef_end,
               eval_workload_selector=args.eval_workloads,
               eval_episodes=args.eval_episodes,
               eval_only=args.eval_only,
               eval_repeats=args.eval_repeats,
               eval_deterministic=(not args.stochastic_eval),
               global_seed=args.seed,
               checkpoint_path=args.checkpoint_path,
               checkpoint_every_steps=args.checkpoint_every_steps,
               checkpoint_keep_last=args.checkpoint_keep_last,
               force_hw_mode=args.force_hw_mode,
               campaign_id=args.campaign_id,
               campaign_stage=args.campaign_stage,
               targeted_campaign_config=args.targeted_campaign_config,
               targeted_sample_prob=args.targeted_sample_prob,
               enable_tile_b4=args.enable_tile_b4,
               allow_eval_train_overlap=args.allow_eval_train_overlap,
               eval_shape_split=args.eval_shape_split,
               eval_output_path=args.eval_output_csv,
               run_id=args.run_id,
               export_parquet_dir=args.export_trace_parquet_dir,
               export_parquet_append=(not args.disable_parquet_append),
               parquet_partition_cols=args.parquet_partition_cols,
               run_command=run_cmd)
