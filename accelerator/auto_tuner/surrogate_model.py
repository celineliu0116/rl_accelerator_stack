#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from tuning_trace import FEATURE_ORDER, load_traces, vectorize_features

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False


@dataclass
class SurrogateMetrics:
    n_train: int
    n_val: int
    mae_cycles: float
    mape: float


class _TinyMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):  # type: ignore[override]
        return self.net(x)


class SurrogateModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.meta_path = model_path + ".meta.json"
        self.mtime = 0.0
        self.ready = False
        self.feature_order = list(FEATURE_ORDER)
        self.mean = np.zeros(len(self.feature_order), dtype=np.float32)
        self.std = np.ones(len(self.feature_order), dtype=np.float32)
        self._lin_w: np.ndarray | None = None
        self._net: _TinyMLP | None = None
        self._device = "cpu"
        self.meta_n_train = 0
        self.meta_n_val = 0
        self.meta_mape = 1.0
        self.meta_mae_cycles = float("inf")
        self.refresh_if_stale()

    def refresh_if_stale(self) -> None:
        if not os.path.exists(self.model_path) or not os.path.exists(self.meta_path):
            self.ready = False
            return
        m = os.path.getmtime(self.model_path)
        if m <= self.mtime and self.ready:
            return
        with open(self.meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.feature_order = list(meta.get("feature_order", FEATURE_ORDER))
        self.mean = np.asarray(meta.get("mean", self.mean.tolist()), dtype=np.float32)
        self.std = np.asarray(meta.get("std", self.std.tolist()), dtype=np.float32)
        self.std[self.std == 0] = 1.0
        backend = meta.get("backend", "linear")
        self.meta_n_train = int(meta.get("n_train", 0))
        self.meta_n_val = int(meta.get("n_val", 0))
        self.meta_mape = float(meta.get("mape", 1.0))
        self.meta_mae_cycles = float(meta.get("mae_cycles", float("inf")))

        if backend == "torch_mlp" and _HAS_TORCH:
            self._net = _TinyMLP(len(self.feature_order)).to(self._device)
            state = torch.load(self.model_path, map_location=self._device)
            self._net.load_state_dict(state)
            self._net.eval()
            self._lin_w = None
        else:
            data = np.load(self.model_path, allow_pickle=False)
            self._lin_w = data["w"]
            self._net = None

        self.mtime = m
        self.ready = True

    def is_trustworthy(self, min_train: int = 200, max_mape: float = 0.35) -> bool:
        self.refresh_if_stale()
        if not self.ready:
            return False
        if self.meta_n_train < int(min_train):
            return False
        if self.meta_mape > float(max_mape):
            return False
        return True

    def _prep(self, rec: Dict[str, Any]) -> np.ndarray:
        x = np.asarray(vectorize_features(rec), dtype=np.float32)
        x = (x - self.mean) / self.std
        return x

    def predict_cycles(self, rec: Dict[str, Any]) -> float:
        self.refresh_if_stale()
        if not self.ready:
            return -1.0
        x = self._prep(rec)
        if self._net is not None and _HAS_TORCH:
            with torch.no_grad():
                t = torch.tensor(x[None, :], dtype=torch.float32, device=self._device)
                y = self._net(t).cpu().numpy().reshape(-1)[0]
        elif self._lin_w is not None:
            x1 = np.concatenate([x, np.array([1.0], dtype=np.float32)], axis=0)
            y = float(np.dot(self._lin_w, x1))
        else:
            return -1.0
        y = float(np.clip(y, 0.0, 20.0))
        cycles = float(np.expm1(y))
        return max(1.0, cycles)

    @staticmethod
    def _build_xy(rows: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        xs = np.asarray([vectorize_features(r) for r in rows], dtype=np.float32)
        ys = np.asarray([math.log1p(float(r["cycles"])) for r in rows], dtype=np.float32)
        return xs, ys

    @staticmethod
    def _filter_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for r in rows:
            if int(r.get("simulated", 0)) != 1:
                continue
            if int(r.get("correctness_passed", 0)) != 1:
                continue
            c = int(r.get("cycles", -1))
            if c <= 0:
                continue
            out.append(r)
        return out

    @classmethod
    def train_from_trace(
        cls,
        trace_path: str,
        model_path: str,
        min_records: int = 200,
        epochs: int = 80,
        lr: float = 1e-3,
        seed: int = 7,
    ) -> SurrogateMetrics | None:
        rows = cls._filter_rows(load_traces(trace_path))
        if len(rows) < min_records:
            return None

        rng = np.random.default_rng(seed)
        idx = np.arange(len(rows))
        rng.shuffle(idx)
        split = max(1, int(0.9 * len(rows)))
        tr = [rows[i] for i in idx[:split]]
        va = [rows[i] for i in idx[split:]]
        if not va:
            va = tr[-1:]
            tr = tr[:-1]

        xtr, ytr = cls._build_xy(tr)
        xva, yva = cls._build_xy(va)
        mean = xtr.mean(axis=0)
        std = xtr.std(axis=0)
        std[std == 0] = 1.0
        xtrn = (xtr - mean) / std
        xvan = (xva - mean) / std

        backend = "linear"
        if _HAS_TORCH:
            torch.manual_seed(seed)
            net = _TinyMLP(xtrn.shape[1])
            opt = optim.Adam(net.parameters(), lr=lr)
            loss_fn = nn.MSELoss()
            xb = torch.tensor(xtrn, dtype=torch.float32)
            yb = torch.tensor(ytr[:, None], dtype=torch.float32)
            for _ in range(int(epochs)):
                opt.zero_grad()
                pred = net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
            torch.save(net.state_dict(), model_path)
            backend = "torch_mlp"
            with torch.no_grad():
                yhat = net(torch.tensor(xvan, dtype=torch.float32)).numpy().reshape(-1)
        else:
            lam = 1e-3
            x1 = np.concatenate([xtrn, np.ones((xtrn.shape[0], 1), dtype=np.float32)], axis=1)
            xtx = x1.T @ x1 + lam * np.eye(x1.shape[1], dtype=np.float32)
            w = np.linalg.solve(xtx, x1.T @ ytr)
            np.savez(model_path, w=w)
            x1v = np.concatenate([xvan, np.ones((xvan.shape[0], 1), dtype=np.float32)], axis=1)
            yhat = x1v @ w

        y_true = np.expm1(np.clip(yva, 0.0, 20.0))
        y_pred = np.expm1(np.clip(yhat, 0.0, 20.0))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(1.0, y_true))))

        meta = {
            "backend": backend,
            "feature_order": FEATURE_ORDER,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "mae_cycles": mae,
            "mape": mape,
        }
        with open(model_path + ".meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return SurrogateMetrics(
            n_train=int(len(tr)),
            n_val=int(len(va)),
            mae_cycles=mae,
            mape=mape,
        )
