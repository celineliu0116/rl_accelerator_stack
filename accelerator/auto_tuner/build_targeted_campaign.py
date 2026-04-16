#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _to_float(value: Any, default: float = -1.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _target_id(workload: str, shape: str, sp: int) -> str:
    return f"{workload}|{shape}|sp{sp}"


def _load_regret_rows(regret_csv: Path) -> List[Dict[str, Any]]:
    rows = _read_csv(regret_csv)
    for row in rows:
        row["workload_tag"] = str(row.get("workload_tag", "")).strip()
        row["shape_signature"] = str(row.get("shape_signature", "")).strip()
        row["sparsity_bucket"] = _to_int(row.get("sparsity_bucket", 0), 0)
        row["topk_regret_pct"] = _to_float(row.get("topk_regret_pct", -1.0), -1.0)
        row["within_regret_pct"] = _to_float(row.get("within_regret_pct", -1.0), -1.0)
        row["topk_matches"] = _to_int(row.get("topk_matches", 0), 0)
        row["within_matches"] = _to_int(row.get("within_matches", 0), 0)
        row["topk_cycles"] = _to_int(row.get("topk_cycles", -1), -1)
        row["within_cycles"] = _to_int(row.get("within_cycles", -1), -1)
    return rows


def _select_worst_regret(rows: List[Dict[str, Any]], policy: str, top_k: int) -> List[Dict[str, Any]]:
    col = "topk_regret_pct" if policy == "topk" else "within_regret_pct"
    filtered = [r for r in rows if _to_float(r.get(col), -1.0) >= 0.0]
    filtered.sort(key=lambda r: _to_float(r.get(col), -1.0), reverse=True)
    return filtered[: max(0, int(top_k))]


def _select_low_coverage(rows: List[Dict[str, Any]], policy: str, threshold: int) -> List[Dict[str, Any]]:
    col = "topk_matches" if policy == "topk" else "within_matches"
    out = [r for r in rows if 0 <= _to_int(r.get(col), 0) <= int(threshold)]
    out.sort(key=lambda r: (_to_int(r.get(col), 0), -_to_float(r.get("topk_regret_pct", -1.0), -1.0)))
    return out


def _select_disagreement(rows: List[Dict[str, Any]], max_count: int) -> List[Dict[str, Any]]:
    out = [
        r
        for r in rows
        if _to_int(r.get("topk_cycles"), -1) > 0
        and _to_int(r.get("within_cycles"), -1) > 0
        and _to_int(r.get("topk_cycles"), -1) != _to_int(r.get("within_cycles"), -1)
    ]
    out.sort(
        key=lambda r: abs(
            _to_float(r.get("topk_regret_pct", -1.0), -1.0)
            - _to_float(r.get("within_regret_pct", -1.0), -1.0)
        ),
        reverse=True,
    )
    return out[: max(0, int(max_count))]


def _underrepresented_buckets(trace_rows: List[Dict[str, Any]], min_rows: int, max_targets: int) -> List[Tuple[str, int, int]]:
    ctr: Counter[Tuple[str, int]] = Counter()
    for r in trace_rows:
        if _to_int(r.get("is_valid"), 0) != 1:
            continue
        workload = str(r.get("workload_tag", "")).strip()
        sp = _to_int(r.get("sparsity_bucket", 0), 0)
        if not workload:
            continue
        ctr[(workload, sp)] += 1
    sparse = [
        (workload, sp, count)
        for (workload, sp), count in ctr.items()
        if count <= int(min_rows)
    ]
    sparse.sort(key=lambda x: (x[2], x[0], x[1]))
    return sparse[: max(0, int(max_targets))]


def _add_target(
    targets: Dict[str, Dict[str, Any]],
    *,
    workload_tag: str,
    shape_signature: str,
    sparsity_bucket: int,
    reason: str,
    weight: float,
    evidence: Dict[str, Any],
) -> None:
    tid = _target_id(workload_tag, shape_signature, sparsity_bucket)
    rec = targets.setdefault(
        tid,
        {
            "target_id": tid,
            "workload_tag": workload_tag,
            "shape_signature": shape_signature,
            "sparsity_bucket": int(sparsity_bucket),
            "weight": 0.0,
            "reasons": [],
            "evidence": {},
        },
    )
    rec["weight"] = float(rec.get("weight", 0.0) + float(weight))
    if reason not in rec["reasons"]:
        rec["reasons"].append(reason)
    rec_evidence = rec.get("evidence")
    if not isinstance(rec_evidence, dict):
        rec_evidence = {}
    rec_evidence.update(evidence)
    rec["evidence"] = rec_evidence


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build targeted RL exploration campaign config from regret + coverage artifacts.")
    ap.add_argument("--regret-csv", type=str, default="data/audits/policy_ab_regret_34keys.csv")
    ap.add_argument("--trace-csv", type=str, default="data/trace_dataset_all.csv")
    ap.add_argument("--policy", type=str, default="topk", choices=["topk", "within_pct"])
    ap.add_argument("--top-regret-k", type=int, default=12)
    ap.add_argument("--low-coverage-threshold", type=int, default=1)
    ap.add_argument("--disagreement-k", type=int, default=12)
    ap.add_argument("--underrepresented-min-rows", type=int, default=8)
    ap.add_argument("--underrepresented-k", type=int, default=12)
    ap.add_argument("--weight-high-regret", type=float, default=3.0)
    ap.add_argument("--weight-low-coverage", type=float, default=2.0)
    ap.add_argument("--weight-disagreement", type=float, default=1.5)
    ap.add_argument("--weight-underrepresented", type=float, default=1.0)
    ap.add_argument("--output-json", type=str, default="data/campaigns/targeted_campaign_v1.json")
    args = ap.parse_args(argv)

    regret_rows = _load_regret_rows(Path(args.regret_csv).resolve())
    trace_rows = _read_csv(Path(args.trace_csv).resolve()) if str(args.trace_csv).strip() else []

    targets: Dict[str, Dict[str, Any]] = {}
    worst_regret = _select_worst_regret(regret_rows, policy=str(args.policy), top_k=int(args.top_regret_k))
    for row in worst_regret:
        workload_tag = str(row.get("workload_tag", "")).strip()
        shape_signature = str(row.get("shape_signature", "")).strip()
        sp = _to_int(row.get("sparsity_bucket", 0), 0)
        _add_target(
            targets,
            workload_tag=workload_tag,
            shape_signature=shape_signature,
            sparsity_bucket=sp,
            reason="high_regret",
            weight=float(args.weight_high_regret),
            evidence={
                "topk_regret_pct": _to_float(row.get("topk_regret_pct", -1.0), -1.0),
                "within_regret_pct": _to_float(row.get("within_regret_pct", -1.0), -1.0),
            },
        )

    low_cov = _select_low_coverage(
        regret_rows,
        policy=str(args.policy),
        threshold=int(args.low_coverage_threshold),
    )
    for row in low_cov:
        workload_tag = str(row.get("workload_tag", "")).strip()
        shape_signature = str(row.get("shape_signature", "")).strip()
        sp = _to_int(row.get("sparsity_bucket", 0), 0)
        _add_target(
            targets,
            workload_tag=workload_tag,
            shape_signature=shape_signature,
            sparsity_bucket=sp,
            reason="low_coverage",
            weight=float(args.weight_low_coverage),
            evidence={
                "topk_matches": _to_int(row.get("topk_matches", 0), 0),
                "within_matches": _to_int(row.get("within_matches", 0), 0),
            },
        )

    disagreements = _select_disagreement(regret_rows, max_count=int(args.disagreement_k))
    for row in disagreements:
        workload_tag = str(row.get("workload_tag", "")).strip()
        shape_signature = str(row.get("shape_signature", "")).strip()
        sp = _to_int(row.get("sparsity_bucket", 0), 0)
        _add_target(
            targets,
            workload_tag=workload_tag,
            shape_signature=shape_signature,
            sparsity_bucket=sp,
            reason="policy_disagreement",
            weight=float(args.weight_disagreement),
            evidence={
                "topk_cycles": _to_int(row.get("topk_cycles", -1), -1),
                "within_cycles": _to_int(row.get("within_cycles", -1), -1),
            },
        )

    underrep = _underrepresented_buckets(
        trace_rows,
        min_rows=int(args.underrepresented_min_rows),
        max_targets=int(args.underrepresented_k),
    )
    for workload_tag, sp, count in underrep:
        _add_target(
            targets,
            workload_tag=str(workload_tag),
            shape_signature="",
            sparsity_bucket=int(sp),
            reason="underrepresented",
            weight=float(args.weight_underrepresented),
            evidence={"valid_row_count": int(count)},
        )

    target_list = sorted(targets.values(), key=lambda x: (-float(x.get("weight", 0.0)), str(x.get("target_id", ""))))
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "generator": "auto_tuner/build_targeted_campaign.py",
        "generated_at_epoch_sec": float(time.time()),
        "policy": str(args.policy),
        "sources": {
            "regret_csv": str(Path(args.regret_csv).resolve()),
            "trace_csv": str(Path(args.trace_csv).resolve()),
        },
        "selection": {
            "top_regret_k": int(args.top_regret_k),
            "low_coverage_threshold": int(args.low_coverage_threshold),
            "disagreement_k": int(args.disagreement_k),
            "underrepresented_min_rows": int(args.underrepresented_min_rows),
            "underrepresented_k": int(args.underrepresented_k),
        },
        "weights": {
            "high_regret": float(args.weight_high_regret),
            "low_coverage": float(args.weight_low_coverage),
            "policy_disagreement": float(args.weight_disagreement),
            "underrepresented": float(args.weight_underrepresented),
        },
        "targets": target_list,
        "summary": {
            "target_count": int(len(target_list)),
            "high_regret_count": int(sum(1 for t in target_list if "high_regret" in t.get("reasons", []))),
            "low_coverage_count": int(sum(1 for t in target_list if "low_coverage" in t.get("reasons", []))),
            "policy_disagreement_count": int(sum(1 for t in target_list if "policy_disagreement" in t.get("reasons", []))),
            "underrepresented_count": int(sum(1 for t in target_list if "underrepresented" in t.get("reasons", []))),
        },
    }

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[TargetedCampaign] targets={len(target_list)} output={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
