#!/usr/bin/env python3
"""Collate Tier-1 / Tier-2 experiment outputs into the tables used by the paper.

Reads:
  * Tier-1: the ``*_agg.csv`` written by ``run_catboost_proxy.py``.
  * Tier-2: every ``experiment_record.json`` under a runs/ tree written by
    ``run_worldcereal_finetune.py``.

Produces tidy CSVs (and, if matplotlib is present, headline figures):
  * ``table_main.csv``      : f1_macro per (encoder, region, scenario, test_view).
  * ``table_delta.csv``     : Δf1 vs the baseline scenario, per cell — the
                              key "does outlier treatment help?" table.
  * ``fig_delta_by_scenario.png`` : bar plot of Δf1 on the CLEAN test view.

These are deliberately thin so you can drop the numbers straight into the
LaTeX result tables once the runs finish.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def load_tier2_records(runs_dir: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for rec_path in runs_dir.rglob("experiment_record.json"):
        try:
            rec = json.loads(rec_path.read_text())
        except Exception:
            continue
        flat = {k: v for k, v in rec.items() if k != "trainer_metrics"}
        # Best-effort extraction of an f1_macro from the trainer metrics blob.
        tm = rec.get("trainer_metrics", {})
        f1 = _find_metric(tm, ("f1_macro", "macro avg", "macro_f1", "f1-score"))
        flat["f1_macro"] = f1
        flat["tier"] = "finetune"
        rows.append(flat)
    return pd.DataFrame(rows)


def _find_metric(blob, keys):
    """Recursively search a nested dict for the first matching numeric metric."""
    if isinstance(blob, dict):
        for k, v in blob.items():
            if isinstance(k, str) and k.lower() in [x.lower() for x in keys]:
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, dict) and "f1-score" in v:
                    return float(v["f1-score"])
            found = _find_metric(v, keys)
            if found is not None:
                return found
    elif isinstance(blob, list):
        for v in blob:
            found = _find_metric(v, keys)
            if found is not None:
                return found
    return None


def build_delta_table(df: pd.DataFrame) -> pd.DataFrame:
    """Δf1 vs baseline within each (tier, encoder, region, test_view)."""
    group_cols = [c for c in ["tier", "encoder", "region", "test_view"] if c in df.columns]
    out = []
    for key, g in df.groupby(group_cols):
        base = g[g["scenario"] == "baseline"]["f1_macro_mean" if "f1_macro_mean" in g else "f1_macro"]
        if base.empty:
            continue
        base_val = float(base.mean())
        col = "f1_macro_mean" if "f1_macro_mean" in g else "f1_macro"
        gg = g.copy()
        gg["f1_macro_baseline"] = base_val
        gg["delta_f1_macro"] = gg[col] - base_val
        out.append(gg)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--proxy-agg-csv", type=Path, default=None,
                   help="Tier-1 *_agg.csv from run_catboost_proxy.py")
    p.add_argument("--finetune-runs-dir", type=Path, default=None,
                   help="Directory tree containing Tier-2 experiment_record.json files")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    if args.proxy_agg_csv and args.proxy_agg_csv.exists():
        t1 = pd.read_csv(args.proxy_agg_csv)
        t1["tier"] = "proxy"
        t1["encoder"] = "frozen"
        frames.append(t1)
    if args.finetune_runs_dir and args.finetune_runs_dir.exists():
        t2 = load_tier2_records(args.finetune_runs_dir)
        if not t2.empty:
            frames.append(t2)

    if not frames:
        raise SystemExit("No inputs found. Provide --proxy-agg-csv and/or --finetune-runs-dir.")

    main_tbl = pd.concat(frames, ignore_index=True)
    main_tbl.to_csv(args.out_dir / "table_main.csv", index=False)

    delta = build_delta_table(main_tbl)
    if not delta.empty:
        delta.to_csv(args.out_dir / "table_delta.csv", index=False)
        _maybe_plot(delta, args.out_dir)

    print(f"[aggregate] wrote tables to {args.out_dir}")
    print(main_tbl.head(40).to_string())


def _maybe_plot(delta: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    clean = delta[delta.get("test_view", "clean") == "clean"]
    if clean.empty:
        clean = delta
    piv = clean.pivot_table(index="scenario", values="delta_f1_macro", aggfunc="mean")
    piv = piv.sort_values("delta_f1_macro")
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in piv["delta_f1_macro"]]
    ax.barh(piv.index, piv["delta_f1_macro"], color=colors)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel(r"$\Delta$ F1-macro vs. baseline (clean test)")
    ax.set_title("Effect of outlier treatment on downstream performance")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_delta_by_scenario.png", dpi=150)
    print(f"[aggregate] wrote fig_delta_by_scenario.png")


if __name__ == "__main__":
    main()
