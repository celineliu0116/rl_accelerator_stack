import json
import os
import re
import subprocess
from pathlib import Path
import time
from typing import Any, Dict, List, Tuple
import numpy as np
import gymnasium as gym
from gymnasium import spaces
try:
    import fcntl
except Exception:
    fcntl = None
try:
    from auto_tuner.workload_bank import sample_workload, parse_workload_selector, parse_shape_split, candidate_workloads
    from auto_tuner.tuning_trace import append_trace, estimate_dma_bytes, estimate_macs, estimate_pe_util
    from auto_tuner.surrogate_model import SurrogateModel
except Exception:
    from workload_bank import sample_workload, parse_workload_selector, parse_shape_split, candidate_workloads
    from tuning_trace import append_trace, estimate_dma_bytes, estimate_macs, estimate_pe_util
    from surrogate_model import SurrogateModel

class SystolicEnv(gym.Env):
    """
    Custom Environment that wraps the cycle-accurate Verilator simulation
    of the bare-metal RISC-V ML SoC.

    RL Action Space (6 dimensions):
      [0] tile_m  : spatial batch height      -> 4, 8, 12, 16          (4 choices)
      [1] tile_n  : output channels per call  -> 4, 8, 12, 16, 20, 24, 28, 32 (8)
      [2] burst   : DMA burst size (bytes)    -> 16, 32, 64             (3 choices)
      [3] prefetch: dispatch depth            -> 1=sequential, 2=ping-pong (2)
      [4] tile_b  : images per HW call        -> 1, 2                   (2 choices)
      [5] hw_mode : hardware dataflow mode    -> 0=dense, 1=sparse (2)

    Total configurations: 4 × 8 × 3 × 2 × 2 × 2 = 768

    Deterministic compiler passes (Fix 2 & Fix 3) are permanently hardcoded
    in the C firmware — the RL agent never wastes exploration budget on them.
    The agent controls only the non-linear hardware trade-offs above,
    including the sparsity-aware dataflow mode selection.
    """
    metadata = {"render_modes": ["ansi"]}
    ACTION_ORDER = (
        "tile_m",
        "tile_n",
        "burst_size",
        "prefetch_depth",
        "tile_b",
        "hardware_dataflow_mode",
    )
    TILE_M_OPTIONS = (4, 8, 12, 16)
    TILE_N_OPTIONS = (4, 8, 12, 16, 20, 24, 28, 32)
    BURST_OPTIONS = (16, 32, 64)
    PREFETCH_OPTIONS = (1, 2)
    TILE_B_OPTIONS_BASE = (1, 2)
    TILE_B_OPTIONS_STAGE2 = (1, 2, 4)
    HW_MODE_OPTIONS = (0, 1)

    @classmethod
    def exploration_space_contract(cls, enable_tile_b4: bool = False) -> Dict[str, Any]:
        tile_b_options = cls.TILE_B_OPTIONS_STAGE2 if bool(enable_tile_b4) else cls.TILE_B_OPTIONS_BASE
        knobs = [
            {"name": "tile_m", "values": list(cls.TILE_M_OPTIONS), "active": True},
            {"name": "tile_n", "values": list(cls.TILE_N_OPTIONS), "active": True},
            {"name": "burst_size", "values": list(cls.BURST_OPTIONS), "active": True},
            {"name": "prefetch_depth", "values": list(cls.PREFETCH_OPTIONS), "active": True},
            {"name": "tile_b", "values": list(tile_b_options), "active": True},
            {"name": "hardware_dataflow_mode", "values": list(cls.HW_MODE_OPTIONS), "active": True},
        ]
        return {
            "schema_version": 1,
            "action_order": list(cls.ACTION_ORDER),
            "knobs": knobs,
            "enable_tile_b4": int(bool(enable_tile_b4)),
            "tile_b_stage": "stage2" if bool(enable_tile_b4) else "base",
            "action_space_cardinality": int(
                len(cls.TILE_M_OPTIONS)
                * len(cls.TILE_N_OPTIONS)
                * len(cls.BURST_OPTIONS)
                * len(cls.PREFETCH_OPTIONS)
                * len(tile_b_options)
                * len(cls.HW_MODE_OPTIONS)
            ),
        }

    def __init__(self, M_target: int, N_target: int, K_target: int,
                 workspace_dir: str | None = None,
                 workload_diversity: bool = True,
                 workload_selector: str = "all",
                 shape_split: str = "all",
                 include_workload_features: bool = True,
                 trace_path: str | None = None,
                 run_id: str = "",
                 surrogate_enabled: bool = False,
                 surrogate_model_path: str | None = None,
                 surrogate_verify_every: int = 10,
                 surrogate_min_train: int = 200,
                 surrogate_max_mape: float = 0.35,
                 reward_normalization: bool = True,
                 reward_norm_warmup: int = 32,
                 reward_norm_clip: float = 6.0,
                 reward_norm_min_std: float = 1e-3,
                 campaign_metadata: Dict[str, Any] | None = None,
                 targeted_specs: List[Dict[str, Any]] | None = None,
                 targeted_sample_prob: float = 0.0,
                 enable_tile_b4: bool = False):
        super().__init__()

        self.base_M = M_target
        self.base_N = N_target
        self.base_K = K_target
        self.M = M_target
        self.N = N_target
        self.K = K_target
        if workspace_dir is None:
            # auto_tuner/env/systolic_env.py -> repo root is 2 parents up
            workspace_dir = str(Path(__file__).resolve().parents[2])
        self.workspace = workspace_dir
        self.workload_diversity = bool(workload_diversity)
        self.workload_selector = workload_selector
        self._enabled_families = parse_workload_selector(workload_selector)
        self.shape_split = parse_shape_split(shape_split)
        self.include_workload_features = bool(include_workload_features)
        self._workload_tag = "gemm"
        self._op_type_id = 0
        self._workload_kind_id = 0
        self._activation = 1
        self._batch_size = 1
        self._seq_len = 1
        self._channels = 1
        self._kernel_h = 1
        self._kernel_w = 1
        self._seed = 0
        self._sram_bytes = 65536
        self._pe_rows = 4
        self._pe_cols = 4
        self._dma_bus_bits = 256
        self._cache_cfg = "none"
        self.extended_workload_features = (
            os.environ.get("ACCELERA_EXTENDED_WORKLOAD_FEATURES", "0") == "1"
        )

        if trace_path is None:
            trace_path = os.path.join(self.workspace, "data", "tuning_trace.jsonl")
        if surrogate_model_path is None:
            surrogate_model_path = os.path.join(self.workspace, "data", "surrogate_model.pt")
        self.trace_path = trace_path
        self.run_id = str(run_id)
        self.surrogate_enabled = bool(surrogate_enabled)
        self.surrogate_verify_every = max(1, int(surrogate_verify_every))
        self.surrogate_min_train = int(surrogate_min_train)
        self.surrogate_max_mape = float(surrogate_max_mape)
        self.reward_normalization = bool(reward_normalization)
        self.reward_norm_warmup = max(2, int(reward_norm_warmup))
        self.reward_norm_clip = max(0.5, float(reward_norm_clip))
        self.reward_norm_min_std = max(1e-9, float(reward_norm_min_std))
        self.campaign_metadata = dict(campaign_metadata or {})
        self.targeted_specs = list(targeted_specs or [])
        self.targeted_sample_prob = float(max(0.0, min(1.0, float(targeted_sample_prob))))
        self.enable_tile_b4 = bool(enable_tile_b4)
        self._campaign_targeted = 0
        self._campaign_target_reason = ""
        self._campaign_target_bucket = ""
        self._campaign_target_weight = 0.0
        self._target_key = ""
        self._reward_norm_stats: dict[tuple[str, str], dict[str, float]] = {}
        self._surrogate = SurrogateModel(surrogate_model_path) if self.surrogate_enabled else None
        self._shape_feasible_cache: dict[tuple[int, int, int], bool] = {}
        self._action_pruned = 0
        self._timeout_blacklist: set[tuple[int, int, int, int, int, int, int, int, int]] = set()
        self._python_bin = (
            os.environ.get("ACCELERA_RL_PYTHON")
            or os.environ.get("ACCELERA_PYTHON")
            or "python3"
        )
        self._last_export_signature: tuple[int, ...] | None = None

        # [tile_m, tile_n, burst_size, prefetch_depth, tile_b, hw_dataflow_mode]
        # tile_b: default {1,2}; {1,2,4} is optional behind enable_tile_b4 flag.
        # hw_mode: {0=dense_systolic, 1=sparse_intersection}
        self._tile_m_options = self.TILE_M_OPTIONS
        self._tile_n_options = self.TILE_N_OPTIONS
        self._burst_options = self.BURST_OPTIONS
        self._prefetch_options = self.PREFETCH_OPTIONS
        self._tile_b_options = self.TILE_B_OPTIONS_STAGE2 if self.enable_tile_b4 else self.TILE_B_OPTIONS_BASE
        self._hw_mode_options = self.HW_MODE_OPTIONS
        self.action_space = spaces.MultiDiscrete([4, 8, 3, 2, len(self._tile_b_options), 2])
        self._candidate_specs = candidate_workloads(self.workload_selector, self.shape_split)
        self._targeted_candidates = self._build_targeted_candidates()

        if self.include_workload_features:
            if self.extended_workload_features:
                # Extended mode (opt-in): adds sparsity decile and mode1 candidacy hints.
                self.observation_space = spaces.Box(
                    low=np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0]),
                    high=np.array([4096, 4096, 4096, 262144, 100, 10, 512, 4096, 4096, 11, 11, 20000, 100000000, 10000000, 10, 1]),
                    dtype=np.int32
                )
            else:
                # Backward-compatible default observation layout.
                self.observation_space = spaces.Box(
                    low=np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1]),
                    high=np.array([4096, 4096, 4096, 262144, 100, 8, 512, 4096, 4096, 11, 11, 20000, 100000000, 10000000]),
                    dtype=np.int32
                )
        else:
            # Ablation mode: only core geometry/sparsity signals.
            self.observation_space = spaces.Box(
                low=np.array([1, 1, 1, 0, 0]),
                high=np.array([4096, 4096, 4096, 262144, 100]),
                dtype=np.int32
            )

        self.cycles_regex = re.compile(r"Total cycles: (\d+)")
        self.stall_regex  = re.compile(r"stall=(\d+)")
        self.accel_perf_regex = re.compile(r"ACCEL_PERF cmd=(\d+) k_limit=(\d+) busy=(\d+) compute=(\d+) stall=(\d+)")
        self.tohost_regex = re.compile(r"\*\*\* (PASSED|FAILED) \*\*\*")
        self._episode_id = 0

    def _sparsity_bucket(self, sparsity_pct: int) -> int:
        sp = max(0, min(100, int(sparsity_pct)))
        return int(max(0, min(10, sp // 10)))

    def _mode1_candidate(self, n_dim: int, k_dim: int, sparsity_pct: int) -> int:
        # Mirror exporter-side structural guard at observation time so policy can
        # condition on likely mode-1 feasibility instead of trying blind toggles.
        min_sparsity = int(os.environ.get("ACCELERA_SPARSE_MODE1_MIN_SPARSITY_PCT", "20"))
        max_n = int(os.environ.get("ACCELERA_SPARSE_MODE1_MAX_N", "8192"))
        max_k = int(os.environ.get("ACCELERA_SPARSE_MODE1_MAX_K", "8192"))
        if int(k_dim) <= 0 or int(n_dim) <= 0:
            return 0
        if (int(k_dim) % 4) != 0:
            return 0
        if int(n_dim) > int(max_n) or int(k_dim) > int(max_k):
            return 0
        if int(sparsity_pct) < int(min_sparsity):
            return 0
        return 1

    def _shape_signature(self, m_dim: int, n_dim: int, k_dim: int) -> str:
        return f"{int(m_dim)}x{int(n_dim)}x{int(k_dim)}"

    def _spec_to_pick(self, spec, episode_idx: int) -> Dict[str, int | str]:
        return {
            "workload_tag": spec.tag,
            "op_type_id": int(spec.op_type_id),
            "workload_kind_id": int(spec.kind_id),
            "M": int(spec.m_dim),
            "N": int(spec.n_dim),
            "K": int(spec.k_dim),
            "sparsity_pct": int(spec.sparsity_pct),
            "activation": int(spec.activation),
            "batch_size": int(spec.batch_size),
            "seq_len": int(spec.seq_len),
            "channels": int(spec.channels),
            "kernel_h": int(spec.kernel_h),
            "kernel_w": int(spec.kernel_w),
            "seed": int((episode_idx * 2654435761 + spec.kind_id * 2246822519) & 0xFFFFFFFF),
        }

    def _target_matches_spec(self, target: Dict[str, Any], spec) -> bool:
        workload = str(target.get("workload_tag", "")).strip()
        if workload and workload != spec.tag:
            return False
        shape_sig = str(target.get("shape_signature", "")).strip()
        if shape_sig:
            expected = self._shape_signature(spec.m_dim, spec.n_dim, spec.k_dim)
            if shape_sig != expected:
                return False
        sp_bucket = target.get("sparsity_bucket", "")
        if str(sp_bucket).strip() not in ("", "-1", "None"):
            try:
                sp_val = int(sp_bucket)
            except Exception:
                sp_val = -1
            if sp_val >= 0 and int(spec.sparsity_pct // 10) != int(sp_val):
                return False
        return True

    def _build_targeted_candidates(self) -> List[Tuple[Any, Dict[str, Any]]]:
        out: List[Tuple[Any, Dict[str, Any]]] = []
        if not self.targeted_specs:
            return out
        for spec in self._candidate_specs:
            for target in self.targeted_specs:
                if not isinstance(target, dict):
                    continue
                if not self._target_matches_spec(target, spec):
                    continue
                reasons = target.get("reasons", [])
                if isinstance(reasons, list):
                    reason = ",".join(str(x) for x in reasons if str(x).strip())
                else:
                    reason = str(target.get("campaign_target_reason", "") or "")
                target_id = str(target.get("target_id", "") or "")
                bucket = str(target.get("campaign_target_bucket", "") or "")
                weight = float(target.get("weight", 1.0) or 1.0)
                out.append(
                    (
                        spec,
                        {
                            "target_key": target_id,
                            "campaign_target_reason": reason,
                            "campaign_target_bucket": bucket,
                            "campaign_target_weight": float(weight),
                        },
                    )
                )
        return out

    def _get_obs(self):
        scratchpad_avail = self._sram_bytes - self._estimate_scratch_bytes(
            tile_m=max(self._tile_m_options),
            tile_n=max(self._tile_n_options),
            prefetch_depth=max(self._prefetch_options),
            tile_b=max(self._tile_b_options),
            hw_dataflow_mode=max(self._hw_mode_options),
        )
        sparsity_pct = getattr(self, "_sparsity_pct", 35)
        sparsity_bucket = self._sparsity_bucket(sparsity_pct)
        mode1_candidate = self._mode1_candidate(self.N, self.K, sparsity_pct)

        est_macs = int(self.M) * int(self.N) * int(self.K)
        est_bytes = int(self.M) * int(self.K) + int(self.N) * int(self.K) + int(self.M) * int(self.N) * 4
        ai_x100 = int((100 * est_macs) // max(1, est_bytes))
        est_macs_k = max(1, est_macs // 1024)
        est_bytes_kb = max(1, est_bytes // 1024)

        base_obs = np.array([
            int(self.M),
            int(self.N),
            int(self.K),
            max(0, int(scratchpad_avail)),
            int(sparsity_pct),
            int(self._op_type_id),
            int(self._batch_size),
            int(self._seq_len),
            int(self._channels),
            int(self._kernel_h),
            int(self._kernel_w),
            int(ai_x100),
            int(est_macs_k),
            int(est_bytes_kb),
        ], dtype=np.int32)
        if self.include_workload_features:
            if self.extended_workload_features:
                return np.concatenate(
                    [base_obs, np.array([int(sparsity_bucket), int(mode1_candidate)], dtype=np.int32)],
                    axis=0,
                )
            return base_obs
        return base_obs[:5]

    def _decode_action(self, action):
        tile_m             = (action[0] + 1) * 4           # {4,8,12,16}
        tile_n             = (action[1] + 1) * 4           # {4,8,...,32}
        burst_size         = [16, 32, 64][action[2]]
        prefetch_depth     = int(action[3]) + 1            # {1,2}
        tile_b             = self._tile_b_options[int(action[4])]
        hw_dataflow_mode   = int(action[5])                # {0,1}
        return tile_m, tile_n, burst_size, prefetch_depth, tile_b, hw_dataflow_mode

    def _estimate_scratch_bytes_for_dims(self,
                                         m_dim: int, n_dim: int, k_dim: int,
                                         tile_m: int, tile_n: int,
                                         prefetch_depth: int, tile_b: int,
                                         hw_dataflow_mode: int) -> int:
        # Tile-local working set estimate (bytes). This prevents rejecting
        # obviously feasible tiled schedules due to full-layer footprint.
        m_tile = max(1, min(int(m_dim), int(tile_m)))
        n_tile = max(1, min(int(n_dim), int(tile_n)))
        batch_tile = max(1, min(int(tile_b), int(m_tile)))
        k_tile = int(k_dim)

        input_bytes = int(batch_tile * k_tile)
        weight_bytes = int(k_tile * n_tile)
        output_bytes = int(batch_tile * n_tile * 4)

        # Sparse modes require metadata/indirection storage.
        if int(hw_dataflow_mode) == 1:
            weight_bytes = int(weight_bytes * 1.15)
        elif int(hw_dataflow_mode) == 2:
            weight_bytes = int(weight_bytes * 1.30)

        required = input_bytes + weight_bytes + output_bytes
        if int(prefetch_depth) == 2:
            # Ping-pong buffering for producer-side overlap.
            required += input_bytes + weight_bytes
        required += 1024  # control/FIFO safety margin
        return int(required)

    def _estimate_scratch_bytes(self,
                                tile_m: int, tile_n: int,
                                prefetch_depth: int, tile_b: int,
                                hw_dataflow_mode: int) -> int:
        return self._estimate_scratch_bytes_for_dims(
            self.M, self.N, self.K,
            tile_m, tile_n, prefetch_depth, tile_b, hw_dataflow_mode
        )

    def _shape_has_feasible_action(self, m_dim: int, n_dim: int, k_dim: int) -> bool:
        key = (int(m_dim), int(n_dim), int(k_dim))
        cached = self._shape_feasible_cache.get(key)
        if cached is not None:
            return cached

        feasible = False
        for tile_m in self._tile_m_options:
            for tile_n in self._tile_n_options:
                for prefetch in self._prefetch_options:
                    for tile_b in self._tile_b_options:
                        for hw_mode in self._hw_mode_options:
                            req = self._estimate_scratch_bytes_for_dims(
                                key[0], key[1], key[2], tile_m, tile_n, prefetch, tile_b, hw_mode
                            )
                            if req <= self._sram_bytes:
                                feasible = True
                                break
                        if feasible:
                            break
                    if feasible:
                        break
                if feasible:
                    break
            if feasible:
                break

        self._shape_feasible_cache[key] = feasible
        return feasible

    def _project_to_feasible_action(self,
                                    tile_m: int, tile_n: int, burst_size: int,
                                    prefetch_depth: int, tile_b: int,
                                    hw_dataflow_mode: int):
        req = self._estimate_scratch_bytes(tile_m, tile_n, prefetch_depth, tile_b, hw_dataflow_mode)
        if req <= self._sram_bytes:
            return tile_m, tile_n, burst_size, prefetch_depth, tile_b, hw_dataflow_mode, 0

        tile_m_candidates = sorted([v for v in self._tile_m_options if v <= tile_m], reverse=True) or [4]
        tile_n_candidates = sorted([v for v in self._tile_n_options if v <= tile_n], reverse=True) or [4]
        tile_b_candidates = sorted([v for v in self._tile_b_options if v <= tile_b], reverse=True) or [1]
        prefetch_candidates = sorted([v for v in self._prefetch_options if v <= prefetch_depth], reverse=True) or [1]
        hw_mode_candidates = [hw_dataflow_mode, 1, 0]
        hw_mode_candidates = [m for i, m in enumerate(hw_mode_candidates) if m in self._hw_mode_options and m not in hw_mode_candidates[:i]]

        for cand_tile_m in tile_m_candidates:
            for cand_tile_n in tile_n_candidates:
                for cand_tile_b in tile_b_candidates:
                    for cand_prefetch in prefetch_candidates:
                        for cand_mode in hw_mode_candidates:
                            req = self._estimate_scratch_bytes(
                                cand_tile_m, cand_tile_n, cand_prefetch, cand_tile_b, cand_mode
                            )
                            if req <= self._sram_bytes:
                                return cand_tile_m, cand_tile_n, burst_size, cand_prefetch, cand_tile_b, cand_mode, 1

        # No feasible mapping for this shape under current SRAM budget.
        return tile_m, tile_n, burst_size, prefetch_depth, tile_b, hw_dataflow_mode, 0

    def _constraint_violations(self, tile_m, tile_n, burst_size, prefetch_depth, tile_b, hw_dataflow_mode):
        violations = []
        if (tile_m % 4) != 0 or tile_m <= 0:
            violations.append("tile_m_alignment")
        if (tile_n % 4) != 0 or tile_n <= 0:
            violations.append("tile_n_alignment")
        if burst_size not in (16, 32, 64):
            violations.append("burst_size")
        if prefetch_depth not in (1, 2):
            violations.append("prefetch_depth")
        if tile_b not in self._tile_b_options:
            violations.append("tile_b")
        if hw_dataflow_mode not in (0, 1):
            violations.append("hw_dataflow_mode")

        scratchpad_avail = self._sram_bytes - self._estimate_scratch_bytes(
            tile_m, tile_n, prefetch_depth, tile_b, hw_dataflow_mode
        )
        if scratchpad_avail < 0:
            violations.append("scratchpad_capacity")
        return violations

    def _base_trace_record(self, tile_m, tile_n, burst_size, prefetch_depth, tile_b, hw_dataflow_mode):
        scratch_required = self._estimate_scratch_bytes(
            tile_m, tile_n, prefetch_depth, tile_b, hw_dataflow_mode
        )
        scratchpad_avail = self._sram_bytes - scratch_required
        macs = estimate_macs(self.M, self.N, self.K)
        dma_est = estimate_dma_bytes(self.M, self.N, self.K)
        sparsity_bucket = self._sparsity_bucket(self._sparsity_pct)
        mode1_candidate = self._mode1_candidate(self.N, self.K, self._sparsity_pct)
        forced_hw_mode = int(os.environ.get("ACCELERA_FORCE_HW_MODE", "-1"))
        campaign_id = str(self.campaign_metadata.get("campaign_id", "") or "")
        campaign_stage = str(self.campaign_metadata.get("campaign_stage", "") or "")
        run_mode = str(self.campaign_metadata.get("run_mode", "") or "")
        return {
            "timestamp": time.time(),
            "run_id": self.run_id,
            "run_mode": run_mode,
            "campaign_id": campaign_id,
            "campaign_stage": campaign_stage,
            "campaign_targeted": int(self._campaign_targeted),
            "campaign_target_reason": str(self._campaign_target_reason),
            "campaign_target_bucket": str(self._campaign_target_bucket),
            "campaign_target_weight": float(self._campaign_target_weight),
            "target_key": str(self._target_key),
            "episode_id": int(self._episode_id),
            "workload_tag": self._workload_tag,
            "shape_signature": f"{int(self.M)}x{int(self.N)}x{int(self.K)}",
            "op_type_id": int(self._op_type_id),
            "M": int(self.M),
            "N": int(self.N),
            "K": int(self.K),
            "sparsity_pct": int(self._sparsity_pct),
            "sparsity_bucket": int(sparsity_bucket),
            "mode1_candidate": int(mode1_candidate),
            "forced_hw_mode": int(forced_hw_mode),
            "activation": int(self._activation),
            "batch_size": int(self._batch_size),
            "seq_len": int(self._seq_len),
            "channels": int(self._channels),
            "kernel_h": int(self._kernel_h),
            "kernel_w": int(self._kernel_w),
            "tile_m": int(tile_m),
            "tile_n": int(tile_n),
            "burst_size": int(burst_size),
            "prefetch_depth": int(prefetch_depth),
            "tile_b": int(tile_b),
            "hardware_dataflow_mode": int(hw_dataflow_mode),
            "executed_hardware_dataflow_mode": int(hw_dataflow_mode),
            "executed_mode_source": "env_pre_export",
            "exported_hardware_dataflow_mode": int(hw_dataflow_mode),
            "export_mode_provenance": "",
            "export_mode_fallback_reason": "",
            "pe_rows": int(self._pe_rows),
            "pe_cols": int(self._pe_cols),
            "sram_bytes": int(self._sram_bytes),
            "dma_bus_bits": int(self._dma_bus_bits),
            "cache_cfg": self._cache_cfg,
            "scratchpad_avail": int(scratchpad_avail),
            "scratch_required_bytes": int(scratch_required),
            "scratch_limit_bytes": int(self._sram_bytes),
            "scratch_util_pct": float((100.0 * scratch_required) / max(1, self._sram_bytes)),
            "macs": int(macs),
            "dma_bytes_est": int(dma_est),
            "cache_misses": -1,
            "stall_dma_starvation": -1,
            "stall_mmio_control": -1,
            "pe_idle_cycles": -1,
            "mem_arb_wait_cycles": -1,
            "fifo_backpressure_cycles": -1,
            "surrogate_n_train": -1,
            "surrogate_mape": -1.0,
            "action_pruned": int(self._action_pruned),
            "reward_raw": -10000.0,
            "reward_norm_active": 0,
            "reward_norm_count": 0,
            "reward_norm_mean": 0.0,
            "reward_norm_std": 0.0,
            "reward_norm_z": 0.0,
            "dense_equiv_macs_per_cycle": -1.0,
        }

    def _extract_exported_mode(self) -> tuple[int | None, str, str]:
        meta_path = os.path.join(self.workspace, "firmware", "include", "model_blob.meta.json")
        if not os.path.exists(meta_path):
            return None, "", ""
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            return None, "", ""
        layers = meta.get("layers")
        if not isinstance(layers, list) or not layers:
            return None, "", ""
        first = layers[0]
        if not isinstance(first, dict):
            return None, "", ""
        try:
            selected = int(first.get("selected_hw_mode", -1))
        except Exception:
            selected = -1
        reason = str(first.get("mode_fallback_reason", "") or "")
        provenance = str(first.get("mode_provenance", "") or "")
        if selected < 0:
            return None, reason, provenance
        return selected, reason, provenance

    def _apply_mode_fields(self,
                           row: dict,
                           *,
                           proposed_hw_mode: int,
                           executed_hw_mode: int,
                           mode_fallback: int,
                           mode_fallback_reason: str,
                           mode_source: str,
                           mode_provenance: str) -> None:
        row["hardware_dataflow_mode"] = int(executed_hw_mode)
        row["executed_hardware_dataflow_mode"] = int(executed_hw_mode)
        row["proposed_hardware_dataflow_mode"] = int(proposed_hw_mode)
        row["mode_fallback"] = int(mode_fallback)
        row["mode_fallback_reason"] = str(mode_fallback_reason)
        row["executed_mode_source"] = str(mode_source)
        row["exported_hardware_dataflow_mode"] = int(executed_hw_mode)
        row["export_mode_provenance"] = str(mode_provenance)
        row["export_mode_fallback_reason"] = str(mode_fallback_reason)

    def _config_key(self,
                    tile_m: int, tile_n: int, burst_size: int,
                    prefetch_depth: int, tile_b: int,
                    hw_dataflow_mode: int) -> tuple[int, int, int, int, int, int, int, int, int]:
        return (
            int(self.M), int(self.N), int(self.K),
            int(tile_m), int(tile_n), int(burst_size),
            int(prefetch_depth), int(tile_b), int(hw_dataflow_mode),
        )

    def _resolve_executed_mode(self, requested_hw_mode: int) -> tuple[int, int, str]:
        req = int(requested_hw_mode)
        mode1_enabled = os.environ.get("ACCELERA_ENABLE_SPARSE_MODE1", "0") == "1"
        fallback = 0
        reason = ""
        forced_raw = os.environ.get("ACCELERA_FORCE_HW_MODE", "").strip()
        if forced_raw != "":
            try:
                forced_mode = int(forced_raw)
            except Exception:
                forced_mode = -1
            if forced_mode in self._hw_mode_options:
                if forced_mode != req:
                    fallback = 1
                    reason = "forced_hw_mode_env"
                req = int(forced_mode)
            else:
                fallback = 1
                reason = "forced_hw_mode_unsupported"
        if req not in self._hw_mode_options:
            req = 0
            fallback = 1
            reason = "unsupported_hw_mode"
        if req == 1 and not mode1_enabled:
            req = 0
            fallback = 1
            reason = "mode1_disabled_env"
        return req, fallback, reason

    def _reward_key(self) -> tuple[str, str]:
        return (str(self._workload_tag), f"{int(self.M)}x{int(self.N)}x{int(self.K)}")

    def _normalize_reward(self, raw_reward: float) -> tuple[float, int, float, int, float, float]:
        key = self._reward_key()
        st = self._reward_norm_stats.setdefault(key, {"count": 0.0, "mean": 0.0, "m2": 0.0})
        count = int(st["count"])
        mean = float(st["mean"])
        m2 = float(st["m2"])

        used_norm = int(self.reward_normalization and count >= self.reward_norm_warmup)
        std = 0.0
        z_score = 0.0
        reward = float(raw_reward)
        if used_norm:
            var = (m2 / max(1, count - 1))
            std = float(np.sqrt(max(var, 0.0)))
            denom = max(self.reward_norm_min_std, std)
            z_score = float((raw_reward - mean) / denom)
            reward = float(np.clip(z_score, -self.reward_norm_clip, self.reward_norm_clip))

        new_count = count + 1
        delta = float(raw_reward - mean)
        mean = mean + (delta / new_count)
        delta2 = float(raw_reward - mean)
        m2 = m2 + (delta * delta2)
        st["count"] = float(new_count)
        st["mean"] = float(mean)
        st["m2"] = float(m2)

        return reward, used_norm, z_score, new_count, mean, std

    def _acquire_workspace_lock(self):
        lock_path = os.path.join(self.workspace, "auto_tuner", ".rl_step.lock")
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        lock_f = open(lock_path, "a+", encoding="utf-8")
        if fcntl is not None:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        return lock_f

    def _release_workspace_lock(self, lock_f):
        if lock_f is None:
            return
        try:
            if fcntl is not None:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        finally:
            lock_f.close()

    def _export_model_blob(self, tile_m, tile_n, burst_size, prefetch_depth, tile_b, hw_dataflow_mode):
        import json
        export_sig = (
            int(self.workload_diversity),
            int(self.M), int(self.N), int(self.K),
            int(self._sparsity_pct), int(self._activation),
            int(self._workload_kind_id), int(self._seed),
            int(tile_m), int(tile_n), int(burst_size),
            int(prefetch_depth), int(tile_b), int(hw_dataflow_mode),
        )
        include_dir = os.path.join(self.workspace, "firmware", "include")
        has_export_artifacts = (
            os.path.exists(os.path.join(include_dir, "model_blob.h"))
            and os.path.exists(os.path.join(include_dir, "autotune_workload.h"))
        )
        if self._last_export_signature == export_sig and has_export_artifacts:
            return

        override = {
            "M": self.M,
            "N": self.N,
            "K": self.K,
            "tile_m": int(tile_m),
            "tile_n": int(tile_n),
            "burst_size": int(burst_size),
            "prefetch_depth": int(prefetch_depth),
            "tile_b": int(tile_b),
            "hardware_dataflow_mode": int(hw_dataflow_mode),
        }
        override_file = os.path.join(self.workspace, "auto_tuner", "rl_override.json")
        with open(override_file, "w", encoding="utf-8") as f:
            json.dump(override, f)

        if self.workload_diversity:
            cmd = [
                self._python_bin,
                os.path.join("auto_tuner", "workload_export.py"),
                "--workload-tag", self._workload_tag,
                "--kind-id", str(self._workload_kind_id),
                "--m", str(self.M),
                "--n", str(self.N),
                "--k", str(self.K),
                "--sparsity-pct", str(self._sparsity_pct),
                "--seed", str(self._seed),
                "--activation", str(self._activation),
            ]
            subprocess.run(
                cmd,
                cwd=self.workspace,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        else:
            subprocess.run(
                [self._python_bin, "mnist_mlp_export.py"],
                cwd=os.path.join(self.workspace, "workloads", "mnist"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
        self._last_export_signature = export_sig

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._campaign_targeted = 0
        self._campaign_target_reason = ""
        self._campaign_target_bucket = ""
        self._campaign_target_weight = 0.0
        self._target_key = ""
        if self.workload_diversity:
            picked = None
            chosen_target_meta: Dict[str, Any] = {}

            use_targeted_pool = (
                bool(self._targeted_candidates)
                and float(self.targeted_sample_prob) > 0.0
                and float(self.np_random.random()) < float(self.targeted_sample_prob)
            )
            if use_targeted_pool:
                for _ in range(32):
                    idx = int(self.np_random.integers(0, len(self._targeted_candidates)))
                    spec, target_meta = self._targeted_candidates[idx]
                    if self._shape_has_feasible_action(int(spec.m_dim), int(spec.n_dim), int(spec.k_dim)):
                        picked = self._spec_to_pick(spec, self._episode_id + 1)
                        chosen_target_meta = dict(target_meta)
                        break

            if picked is None:
                for _ in range(32):
                    cand = sample_workload(
                        self.np_random,
                        self._episode_id + 1,
                        selector=self.workload_selector,
                        shape_split=self.shape_split,
                    )
                    if self._shape_has_feasible_action(int(cand["M"]), int(cand["N"]), int(cand["K"])):
                        picked = cand
                        break
            if picked is None:
                # Deterministic fallback: first feasible workload for selector/split.
                for spec in self._candidate_specs:
                    if self._shape_has_feasible_action(spec.m_dim, spec.n_dim, spec.k_dim):
                        picked = self._spec_to_pick(spec, self._episode_id + 1)
                        for cand_spec, target_meta in self._targeted_candidates:
                            if int(cand_spec.m_dim) == int(spec.m_dim) and int(cand_spec.n_dim) == int(spec.n_dim) and int(cand_spec.k_dim) == int(spec.k_dim) and str(cand_spec.tag) == str(spec.tag):
                                chosen_target_meta = dict(target_meta)
                                break
                        break
            if picked is None:
                picked = {
                    "workload_tag": "gemm",
                    "op_type_id": 0,
                    "workload_kind_id": 0,
                    "M": int(self.base_M),
                    "N": int(self.base_N),
                    "K": int(self.base_K),
                    "sparsity_pct": 5,
                    "activation": 0,
                    "batch_size": 1,
                    "seq_len": 1,
                    "channels": 1,
                    "kernel_h": 1,
                    "kernel_w": 1,
                    "seed": int(self._episode_id + 1),
                }
            if chosen_target_meta:
                self._campaign_targeted = 1
                self._campaign_target_reason = str(chosen_target_meta.get("campaign_target_reason", "") or "")
                self._campaign_target_bucket = str(chosen_target_meta.get("campaign_target_bucket", "") or "")
                self._campaign_target_weight = float(chosen_target_meta.get("campaign_target_weight", 1.0) or 1.0)
                self._target_key = str(chosen_target_meta.get("target_key", "") or "")
            self.M = int(picked["M"])
            self.N = int(picked["N"])
            self.K = int(picked["K"])
            self._sparsity_pct = int(picked["sparsity_pct"])
            self._op_type_id = int(picked["op_type_id"])
            self._activation = int(picked["activation"])
            self._workload_tag = str(picked["workload_tag"])
            self._workload_kind_id = int(picked["workload_kind_id"])
            self._batch_size = int(picked["batch_size"])
            self._seq_len = int(picked["seq_len"])
            self._channels = int(picked["channels"])
            self._kernel_h = int(picked["kernel_h"])
            self._kernel_w = int(picked["kernel_w"])
            self._seed = int(picked["seed"])
        else:
            self.M = self.base_M
            self.N = self.base_N
            self.K = self.base_K
            self._sparsity_pct = getattr(self, "_sparsity_pct", 35)
            self._op_type_id = 0
            self._activation = 1
            self._workload_tag = "mnist"
            self._workload_kind_id = 0
            self._batch_size = 1
            self._seq_len = int(self.base_M)
            self._channels = int(self.base_K)
            self._kernel_h = 1
            self._kernel_w = 1
            self._seed = 0
        return self._get_obs(), {}

    def step(self, action):
        self._episode_id += 1
        req_tile_m, req_tile_n, req_burst_size, req_prefetch_depth, req_tile_b, req_hw_dataflow_mode = self._decode_action(action)
        tile_m, tile_n, burst_size, prefetch_depth, tile_b, proposed_hw_mode, action_pruned = self._project_to_feasible_action(
            req_tile_m, req_tile_n, req_burst_size, req_prefetch_depth, req_tile_b, req_hw_dataflow_mode
        )
        hw_dataflow_mode, mode_fallback, mode_fallback_reason = self._resolve_executed_mode(proposed_hw_mode)
        self._action_pruned = int(action_pruned)
        base_trace = self._base_trace_record(tile_m, tile_n, burst_size, prefetch_depth, tile_b, hw_dataflow_mode)
        base_trace.update({
            "requested_tile_m": int(req_tile_m),
            "requested_tile_n": int(req_tile_n),
            "requested_burst_size": int(req_burst_size),
            "requested_prefetch_depth": int(req_prefetch_depth),
            "requested_tile_b": int(req_tile_b),
            "requested_hw_mode": int(req_hw_dataflow_mode),
            "proposed_hardware_dataflow_mode": int(proposed_hw_mode),
            "executed_hardware_dataflow_mode": int(hw_dataflow_mode),
            "mode_fallback": int(mode_fallback),
            "mode_fallback_reason": str(mode_fallback_reason),
            "action_pruned": int(action_pruned),
        })
        base_info = {
            "M": int(self.M),
            "N": int(self.N),
            "K": int(self.K),
            "activation": int(self._activation),
            "workload_tag": self._workload_tag,
            "op_type_id": int(self._op_type_id),
            "sparsity_pct": int(self._sparsity_pct),
            "sparsity_bucket": int(self._sparsity_bucket(self._sparsity_pct)),
            "mode1_candidate": int(self._mode1_candidate(self.N, self.K, self._sparsity_pct)),
            "forced_hw_mode": int(os.environ.get("ACCELERA_FORCE_HW_MODE", "-1")),
            "run_mode": str(self.campaign_metadata.get("run_mode", "") or ""),
            "campaign_id": str(self.campaign_metadata.get("campaign_id", "") or ""),
            "campaign_stage": str(self.campaign_metadata.get("campaign_stage", "") or ""),
            "campaign_targeted": int(self._campaign_targeted),
            "campaign_target_reason": str(self._campaign_target_reason),
            "campaign_target_bucket": str(self._campaign_target_bucket),
            "campaign_target_weight": float(self._campaign_target_weight),
            "target_key": str(self._target_key),
            "shape_signature": f"{int(self.M)}x{int(self.N)}x{int(self.K)}",
            "tile_m": int(tile_m),
            "tile_n": int(tile_n),
            "burst_size": int(burst_size),
            "prefetch_depth": int(prefetch_depth),
            "tile_b": int(tile_b),
            "hardware_dataflow_mode": int(hw_dataflow_mode),
            "proposed_hardware_dataflow_mode": int(proposed_hw_mode),
            "executed_hardware_dataflow_mode": int(hw_dataflow_mode),
            "mode_fallback": int(mode_fallback),
            "mode_fallback_reason": str(mode_fallback_reason),
            "action_pruned": int(action_pruned),
            "requested_tile_m": int(req_tile_m),
            "requested_tile_n": int(req_tile_n),
            "requested_burst_size": int(req_burst_size),
            "requested_prefetch_depth": int(req_prefetch_depth),
            "requested_tile_b": int(req_tile_b),
            "requested_hw_mode": int(req_hw_dataflow_mode),
            "scratch_required_bytes": int(base_trace.get("scratch_required_bytes", -1)),
            "scratch_limit_bytes": int(base_trace.get("scratch_limit_bytes", self._sram_bytes)),
            "scratch_util_pct": float(base_trace.get("scratch_util_pct", -1.0)),
        }
        self._apply_mode_fields(
            base_trace,
            proposed_hw_mode=int(proposed_hw_mode),
            executed_hw_mode=int(hw_dataflow_mode),
            mode_fallback=int(mode_fallback),
            mode_fallback_reason=str(mode_fallback_reason),
            mode_source="env_pre_export",
            mode_provenance="env_resolve",
        )
        self._apply_mode_fields(
            base_info,
            proposed_hw_mode=int(proposed_hw_mode),
            executed_hw_mode=int(hw_dataflow_mode),
            mode_fallback=int(mode_fallback),
            mode_fallback_reason=str(mode_fallback_reason),
            mode_source="env_pre_export",
            mode_provenance="env_resolve",
        )
        violations = self._constraint_violations(tile_m, tile_n, burst_size, prefetch_depth, tile_b, hw_dataflow_mode)
        if violations:
            info = {
                "error": "constraint_violation",
                "violations": violations,
            }
            info.update(base_info)
            trace = dict(base_trace)
            trace.update({
                "simulated": 0,
                "surrogate_used": 0,
                "constraint_ok": 0,
                "constraint_violations": violations,
                "correctness_passed": 0,
                "cycles": -1,
                "stalls": -1,
                "reward": -10000.0,
                "pe_util_est": -1.0,
            })
            append_trace(self.trace_path, trace)
            return self._get_obs(), -10000.0, True, False, info

        cfg_key = self._config_key(
            tile_m, tile_n, burst_size, prefetch_depth, tile_b, hw_dataflow_mode
        )
        if cfg_key in self._timeout_blacklist:
            info = {"error": "known_timeout_blacklist"}
            info.update(base_info)
            trace = dict(base_trace)
            trace.update({
                "simulated": 0,
                "surrogate_used": 0,
                "constraint_ok": 1,
                "constraint_violations": [],
                "correctness_passed": 0,
                "cycles": -1,
                "stalls": -1,
                "reward": -10000.0,
                "error": "known_timeout_blacklist",
                "pe_util_est": -1.0,
            })
            append_trace(self.trace_path, trace)
            return self._get_obs(), -10000.0, True, False, info

        if self.surrogate_enabled and self._surrogate is not None:
            self._surrogate.refresh_if_stale()
            if self._surrogate.is_trustworthy(self.surrogate_min_train, self.surrogate_max_mape) and \
               (self._episode_id % self.surrogate_verify_every) != 0:
                pred_cycles = int(self._surrogate.predict_cycles(base_trace))
                raw_reward = float(1000000.0 / max(1, pred_cycles))
                reward, norm_active, norm_z, norm_count, norm_mean, norm_std = self._normalize_reward(raw_reward)
                info = {
                    "surrogate_pred": 1,
                    "pred_cycles": pred_cycles,
                    "reward_raw": float(raw_reward),
                    "reward_norm_active": int(norm_active),
                    "reward_norm_z": float(norm_z),
                    "reward_norm_count": int(norm_count),
                    "reward_norm_mean": float(norm_mean),
                    "reward_norm_std": float(norm_std),
                    "surrogate_mape": float(self._surrogate.meta_mape),
                    "surrogate_n_train": int(self._surrogate.meta_n_train),
                }
                info.update(base_info)
                trace = dict(base_trace)
                trace.update({
                    "simulated": 0,
                    "surrogate_used": 1,
                    "constraint_ok": 1,
                    "constraint_violations": [],
                    "correctness_passed": -1,
                    "pred_cycles": pred_cycles,
                    "cycles": -1,
                    "stalls": -1,
                    "reward_raw": float(raw_reward),
                    "reward_norm_active": int(norm_active),
                    "reward_norm_count": int(norm_count),
                    "reward_norm_mean": float(norm_mean),
                    "reward_norm_std": float(norm_std),
                    "reward_norm_z": float(norm_z),
                    "reward": float(reward),
                    "pe_util_est": float(estimate_pe_util(pred_cycles, self.M, self.N, self.K)),
                    "surrogate_n_train": int(self._surrogate.meta_n_train),
                    "surrogate_mape": float(self._surrogate.meta_mape),
                })
                append_trace(self.trace_path, trace)
                return self._get_obs(), reward, True, False, info

        cmd = [
            "make", "run",
            "INFERENCE_SRC=inference_generic.c",
            "ENABLE_DUAL_ISSUE=1",
            "GENERIC_CFLAGS=-O3 -fno-strict-aliasing",
            f"CFLAGS_EXTRA=-DRL_EPISODE_ID={self._episode_id} -DRL_AUTOTUNE_MODE",
            "EXTRA_FLAGS=+max-cycles=50000000",
            "EXTRA_VFLAGS=-DUSE_SYSTOLIC_ACCEL"
        ]

        hw_mode_names = {0: "DENSE", 1: "SPARSE_ISECT"}
        print(f"[SystolicEnv] workload={self._workload_tag} MxNxK={self.M}x{self.N}x{self.K} "
              f"sp={self._sparsity_pct}% tile_m={tile_m} tile_n={tile_n} burst={burst_size} "
              f"prefetch={prefetch_depth} tile_b={tile_b} "
              f"hw_mode={hw_dataflow_mode}({hw_mode_names.get(hw_dataflow_mode, 'UNKNOWN')})"
              f"{' [pruned]' if action_pruned else ''}")
        lock_f = None
        try:
            lock_f = self._acquire_workspace_lock()
            self._export_model_blob(tile_m, tile_n, burst_size, prefetch_depth, tile_b, hw_dataflow_mode)
            exported_mode, export_fallback_reason, export_provenance = self._extract_exported_mode()
            if exported_mode is not None:
                prev_mode = int(hw_dataflow_mode)
                hw_dataflow_mode = int(exported_mode)
                if hw_dataflow_mode != prev_mode:
                    mode_fallback = 1
                    if not mode_fallback_reason:
                        mode_fallback_reason = str(export_fallback_reason or "exporter_mode_fallback")
                    print(
                        f"[SystolicEnv] exporter corrected hw_mode {prev_mode}->{hw_dataflow_mode} "
                        f"reason={mode_fallback_reason or 'n/a'}"
                    )
                elif export_fallback_reason and not mode_fallback_reason:
                    # Preserve exporter-side reason visibility even when final mode
                    # numerically matches the pre-export plan.
                    mode_fallback_reason = str(export_fallback_reason)
                self._apply_mode_fields(
                    base_trace,
                    proposed_hw_mode=int(proposed_hw_mode),
                    executed_hw_mode=int(hw_dataflow_mode),
                    mode_fallback=int(mode_fallback),
                    mode_fallback_reason=str(mode_fallback_reason),
                    mode_source="export_metadata",
                    mode_provenance=str(export_provenance),
                )
                self._apply_mode_fields(
                    base_info,
                    proposed_hw_mode=int(proposed_hw_mode),
                    executed_hw_mode=int(hw_dataflow_mode),
                    mode_fallback=int(mode_fallback),
                    mode_fallback_reason=str(mode_fallback_reason),
                    mode_source="export_metadata",
                    mode_provenance=str(export_provenance),
                )
            result = subprocess.run(cmd, cwd=self.workspace, capture_output=True, timeout=120)
            stdout = result.stdout.decode("utf-8", errors="replace")
            stderr = result.stderr.decode("utf-8", errors="replace")
            sim_log = stdout + stderr

            # Cross-environment workspace mounts can leave stale obj_dir makefiles
            # pointing at a different Verilator include path. Auto-heal once.
            if result.returncode != 0 and "No rule to make target" in sim_log and "verilated.h" in sim_log:
                subprocess.run(["make", "clean"], cwd=self.workspace,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                result = subprocess.run(cmd, cwd=self.workspace, capture_output=True, timeout=120)
                stdout = result.stdout.decode("utf-8", errors="replace")
                stderr = result.stderr.decode("utf-8", errors="replace")
                sim_log = stdout + stderr

            if result.returncode != 0:
                print(f"[SystolicEnv] Make Failed! Exit Code: {result.returncode}")
                info = {"error": "make_failed", "make_rc": int(result.returncode)}
                info.update(base_info)
                trace = dict(base_trace)
                trace.update({
                    "simulated": 1,
                    "surrogate_used": 0,
                    "constraint_ok": 1,
                    "constraint_violations": [],
                    "correctness_passed": 0,
                    "cycles": -1,
                    "stalls": -1,
                    "reward": -10000.0,
                    "error": "make_failed",
                    "pe_util_est": -1.0,
                })
                append_trace(self.trace_path, trace)
                return self._get_obs(), -10000.0, True, False, info

        except subprocess.TimeoutExpired:
            print("[SystolicEnv] Simulation Timeout!")
            self._timeout_blacklist.add(cfg_key)
            trace = dict(base_trace)
            trace.update({
                "simulated": 1,
                "surrogate_used": 0,
                "constraint_ok": 1,
                "constraint_violations": [],
                "correctness_passed": 0,
                "cycles": -1,
                "stalls": -1,
                "reward": -10000.0,
                "error": "timeout",
                "pe_util_est": -1.0,
            })
            append_trace(self.trace_path, trace)
            info = {"error": "timeout"}
            info.update(base_info)
            return self._get_obs(), -10000.0, True, False, info
        except subprocess.CalledProcessError as exc:
            trace = dict(base_trace)
            trace.update({
                "simulated": 0,
                "surrogate_used": 0,
                "constraint_ok": 1,
                "constraint_violations": [],
                "correctness_passed": 0,
                "cycles": -1,
                "stalls": -1,
                "reward": -10000.0,
                "error": "model_export_failed",
                "model_export_rc": int(getattr(exc, "returncode", -1)),
                "pe_util_est": -1.0,
            })
            append_trace(self.trace_path, trace)
            info = {
                "error": "model_export_failed",
                "model_export_rc": int(getattr(exc, "returncode", -1)),
            }
            info.update(base_info)
            return self._get_obs(), -10000.0, True, False, info
        except Exception as exc:
            trace = dict(base_trace)
            trace.update({
                "simulated": 0,
                "surrogate_used": 0,
                "constraint_ok": 1,
                "constraint_violations": [],
                "correctness_passed": 0,
                "cycles": -1,
                "stalls": -1,
                "reward": -10000.0,
                "error": "env_step_exception",
                "error_detail": f"{type(exc).__name__}: {exc}",
                "pe_util_est": -1.0,
            })
            append_trace(self.trace_path, trace)
            info = {
                "error": "env_step_exception",
                "error_detail": f"{type(exc).__name__}: {exc}",
            }
            info.update(base_info)
            return self._get_obs(), -10000.0, True, False, info
        finally:
            self._release_workspace_lock(lock_f)

        reward     = 0.0
        terminated = True
        info       = {}
        accel_busy = 0
        accel_compute = 0
        accel_stall = 0
        for m in self.accel_perf_regex.finditer(sim_log):
            accel_busy += int(m.group(3))
            accel_compute += int(m.group(4))
            accel_stall += int(m.group(5))

        tohost_match = self.tohost_regex.search(sim_log)
        if not tohost_match or tohost_match.group(1) == "FAILED":
            print(f"[SystolicEnv] Math Corruption or FAILED tohost. Reward: -10000")
            reward = -10000.0
            info["error"] = "math_corruption"
            info.update(base_info)
            trace = dict(base_trace)
            trace.update({
                "simulated": 1,
                "surrogate_used": 0,
                "constraint_ok": 1,
                "constraint_violations": [],
                "correctness_passed": 0,
                "cycles": -1,
                "stalls": -1,
                "accel_busy": int(accel_busy),
                "accel_compute": int(accel_compute),
                "accel_stall": int(accel_stall),
                "reward": float(reward),
                "pe_util_est": -1.0,
            })
            append_trace(self.trace_path, trace)
            return self._get_obs(), reward, terminated, False, info

        cycles_match = self.cycles_regex.search(sim_log)
        if cycles_match:
            total_cycles = int(cycles_match.group(1))
            info["cycles"] = total_cycles
        else:
            reward = -10000.0
            info["error"] = "no_cycles_reported"
            info.update(base_info)
            trace = dict(base_trace)
            trace.update({
                "simulated": 1,
                "surrogate_used": 0,
                "constraint_ok": 1,
                "constraint_violations": [],
                "correctness_passed": 0,
                "cycles": -1,
                "stalls": -1,
                "accel_busy": int(accel_busy),
                "accel_compute": int(accel_compute),
                "accel_stall": int(accel_stall),
                "reward": float(reward),
                "pe_util_est": -1.0,
            })
            append_trace(self.trace_path, trace)
            return self._get_obs(), reward, terminated, False, info

        total_stalls = sum(int(m.group(1)) for m in self.stall_regex.finditer(sim_log))
        raw_reward = float((1000000.0 / max(1, int(info["cycles"]))) - (total_stalls * 0.1))
        reward, norm_active, norm_z, norm_count, norm_mean, norm_std = self._normalize_reward(raw_reward)
        stall_dma_starvation = int(accel_stall)
        stall_mmio_control = int(max(0, total_stalls - stall_dma_starvation))
        pe_idle_cycles = int(max(0, accel_busy - accel_compute))
        mem_arb_wait = int(stall_dma_starvation)
        fifo_backpressure = int(max(0, stall_dma_starvation - pe_idle_cycles))
        info.update(base_info)
        info["stalls"] = total_stalls
        info["accel_busy"] = int(accel_busy)
        info["accel_compute"] = int(accel_compute)
        info["accel_stall"] = int(accel_stall)
        info["stall_dma_starvation"] = stall_dma_starvation
        info["stall_mmio_control"] = stall_mmio_control
        info["pe_idle_cycles"] = pe_idle_cycles
        info["mem_arb_wait_cycles"] = mem_arb_wait
        info["fifo_backpressure_cycles"] = fifo_backpressure
        info["reward_raw"] = float(raw_reward)
        info["reward_norm_active"] = int(norm_active)
        info["reward_norm_count"] = int(norm_count)
        info["reward_norm_mean"] = float(norm_mean)
        info["reward_norm_std"] = float(norm_std)
        info["reward_norm_z"] = float(norm_z)
        dense_equiv_macs_per_cycle = float(estimate_macs(self.M, self.N, self.K) / max(1, int(info["cycles"])))
        pe_util_est = float(estimate_pe_util(int(info["cycles"]), self.M, self.N, self.K))
        info["dense_equiv_macs_per_cycle"] = dense_equiv_macs_per_cycle
        info["pe_util_est"] = pe_util_est

        trace = dict(base_trace)
        trace.update({
            "simulated": 1,
            "surrogate_used": 0,
            "constraint_ok": 1,
            "constraint_violations": [],
            "correctness_passed": 1,
            "cycles": int(info["cycles"]),
            "stalls": int(total_stalls),
            "accel_busy": int(accel_busy),
            "accel_compute": int(accel_compute),
            "accel_stall": int(accel_stall),
            "stall_dma_starvation": stall_dma_starvation,
            "stall_mmio_control": stall_mmio_control,
            "pe_idle_cycles": pe_idle_cycles,
            "mem_arb_wait_cycles": mem_arb_wait,
            "fifo_backpressure_cycles": fifo_backpressure,
            "reward_raw": float(raw_reward),
            "reward_norm_active": int(norm_active),
            "reward_norm_count": int(norm_count),
            "reward_norm_mean": float(norm_mean),
            "reward_norm_std": float(norm_std),
            "reward_norm_z": float(norm_z),
            "reward": float(reward),
            "pe_util_est": pe_util_est,
            "dense_equiv_macs_per_cycle": dense_equiv_macs_per_cycle,
        })
        append_trace(self.trace_path, trace)

        return self._get_obs(), reward, terminated, False, info

    def render(self):
        pass
