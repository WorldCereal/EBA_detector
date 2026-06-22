#!/usr/bin/env python3
"""Tier-1 experiment runner: fast CatBoost-on-embeddings proxy.

This is the cheap, high-throughput tier used to *sweep* the full scenario ×
region grid in minutes-to-hours, before committing GPU time to the real
WorldCereal fine-tuning (Tier 2, ``run_worldcereal_finetune.py``).  Because it
trains a CatBoost classifier directly on the frozen 128-d embeddings, it
answers the central question — *does treating outliers in the TRAIN set improve
classification of a fixed CLEAN test set?* — without any of the fine-tuning
machinery.

Honesty guarantees (see ``scenarios.py``):
  * Scenarios are applied to the TRAIN split only.
  * The test split is fixed across scenarios and the **clean** view is
    detector-independent (defined by annotation quality / gold mask).
  * Three test views (clean / full / minus_flagged) are reported so a metric
    change cannot be confused with the test set becoming easier.

Input parquet must contain: embeddings (``embedding_0..N`` or an ``embedding``
vector column), the label column, an anomaly-flag column, a
confidence-nonoutlier column, and ideally a region column and a quality column.

Example
-------
python run_catboost_proxy.py \
    --scored-parquet /data/outlier_scores.parquet \
    --label-col CTY24_cls --flag-col CTY24_anomaly_flag \
    --conf-col CTY24_confidence_nonoutlier \
    --region-col region --quality-col quality_score_ct \
    --group-col ref_id --seeds 0 1 2 \
    --out-csv results/proxy_results.csv

Or, against the real split artefacts (DuckDB embeddings + merged outlier parquet):

python run_catboost_proxy.py \
    --duckdb embeddings_cache.duckdb --merged-parquet merged_LC10_CTY24_flagged.parquet \
    --domain CTY24 --group-col ref_id --seeds 0 1 2 \
    --out-csv results/proxy_results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

# Allow running as a script from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from outlier_embeddings.experiments import (  # noqa: E402
    _extract_embeddings_matrix,
    _evaluate,
    _train_catboost_multiclass,
)
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for sibling 'scenarios'
from scenarios import (  # noqa: E402
    DEFAULT_SCENARIOS,
    Scenario,
    apply_scenario_to_train,
    build_test_views,
)


def _make_fixed_split(
    df: pd.DataFrame,
    *,
    group_col: Optional[str],
    test_frac: float,
    seed: int,
) -> pd.DataFrame:
    """Add a ``split`` column (train/test).

    When *group_col* is given (e.g. ``ref_id`` or an H3 cell) the split is made
    at the group level to prevent spatial leakage between train and test.
    """
    rng = np.random.RandomState(seed)
    df = df.copy()
    if group_col and group_col in df.columns:
        groups = df[group_col].astype(str).unique()
        rng.shuffle(groups)
        n_test = max(1, int(round(test_frac * len(groups))))
        test_groups = set(groups[:n_test])
        df["split"] = np.where(
            df[group_col].astype(str).isin(test_groups), "test", "train"
        )
    else:
        idx = df.index.to_numpy()
        rng.shuffle(idx)
        n_test = max(1, int(round(test_frac * len(idx))))
        test_idx = set(idx[:n_test].tolist())
        df["split"] = np.where(df.index.isin(test_idx), "test", "train")
    return df


def run(
    df: pd.DataFrame,
    *,
    label_col: str,
    flag_col: str,
    conf_col: str,
    region_col: Optional[str] = None,
    quality_col: Optional[str] = None,
    group_col: Optional[str] = None,
    scenarios: Sequence[Scenario] = DEFAULT_SCENARIOS,
    seeds: Sequence[int] = (0, 1, 2),
    test_frac: float = 0.2,
    quality_threshold: float = 0.9,
    min_class_count: int = 20,
    catboost_params: Optional[dict] = None,
) -> pd.DataFrame:
    """Run the full scenario × region × seed grid; return tidy results."""
    rows: List[dict] = []

    regions = ["ALL"]
    if region_col and region_col in df.columns:
        regions = ["ALL"] + sorted(df[region_col].dropna().astype(str).unique().tolist())

    for region in regions:
        sub = df if region == "ALL" else df[df[region_col].astype(str) == region]
        if len(sub) < 200:
            print(f"[proxy] skipping region {region}: too few samples ({len(sub)})")
            continue

        for seed in seeds:
            split_df = _make_fixed_split(
                sub, group_col=group_col, test_frac=test_frac, seed=seed
            )
            train_full = split_df[split_df["split"] == "train"].copy()
            eval_full = split_df[split_df["split"] == "test"].copy()

            # Fixed, detector-independent test views (built once per seed/region)
            views = build_test_views(
                eval_full,
                flag_col=flag_col,
                quality_col=quality_col,
                quality_threshold=quality_threshold,
            )

            # Keep only labels present with enough support in the clean view
            vc = views["clean"][label_col].value_counts()
            keep_labels = set(vc[vc >= min_class_count].index)
            if len(keep_labels) < 2:
                print(f"[proxy] region {region} seed {seed}: <2 usable classes, skip")
                continue

            for scen in scenarios:
                tr = apply_scenario_to_train(
                    train_full, scen, flag_col=flag_col, conf_col=conf_col
                )
                tr = tr[tr[label_col].isin(keep_labels)]
                if len(tr) < 100:
                    continue

                X_tr, _ = _extract_embeddings_matrix(tr)
                y_tr = tr[label_col].astype(str).to_numpy()
                w_tr = tr["sample_weight"].to_numpy()

                # Small internal val carved from train for early stopping
                rng = np.random.RandomState(seed)
                perm = rng.permutation(len(tr))
                n_val = max(50, int(0.1 * len(tr)))
                va_idx, tr_idx = perm[:n_val], perm[n_val:]

                model = _train_catboost_multiclass(
                    X_train=X_tr[tr_idx], y_train=y_tr[tr_idx],
                    X_val=X_tr[va_idx], y_val=y_tr[va_idx],
                    sample_weight=w_tr[tr_idx],
                    seed=int(seed), params=catboost_params,
                )

                for view_name, view_df in views.items():
                    vdf = view_df[view_df[label_col].isin(keep_labels)]
                    if len(vdf) < 50:
                        continue
                    X_te, _ = _extract_embeddings_matrix(vdf)
                    y_te = vdf[label_col].astype(str).to_numpy()
                    yhat = model.predict(X_te).astype(str).reshape(-1)
                    metrics = _evaluate(y_te, yhat)
                    rows.append({
                        "region": region,
                        "scenario": scen.name,
                        "drop_mode": scen.drop_mode,
                        "weight_mode": scen.weight_mode,
                        "test_view": view_name,
                        "seed": seed,
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(vdf)),
                        "accuracy": metrics["accuracy"],
                        "f1_macro": metrics["f1_macro"],
                        "f1_weighted": metrics["f1_weighted"],
                    })
                    print(f"[proxy] {region:12s} {scen.name:24s} {view_name:13s} "
                          f"seed={seed} f1_macro={metrics['f1_macro']:.4f}")

    return pd.DataFrame(rows)


def aggregate(results: pd.DataFrame) -> pd.DataFrame:
    """Mean ± std over seeds for each (region, scenario, test_view)."""
    return (
        results.groupby(["region", "scenario", "test_view"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            f1_macro_mean=("f1_macro", "mean"),
            f1_macro_std=("f1_macro", "std"),
            f1_weighted_mean=("f1_weighted", "mean"),
            acc_mean=("accuracy", "mean"),
            n_train_mean=("n_train", "mean"),
            n_test_mean=("n_test", "mean"),
        )
        .sort_values(["region", "test_view", "f1_macro_mean"], ascending=[True, True, False])
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- Source A: a single pre-scored parquet (legacy / combined artefact) ---
    p.add_argument("--scored-parquet", type=Path, default=None,
                   help="Single parquet with embeddings + label/flag/conf columns.")
    # --- Source B: real split artefacts (DuckDB embeddings + merged outlier parquet) ---
    p.add_argument("--duckdb", type=Path, default=None,
                   help="embeddings_cache DuckDB; joined to --merged-parquet on sample_id.")
    p.add_argument("--merged-parquet", type=Path, default=None,
                   help="Merged outlier parquet (outlier_{domain}_cls etc.).")
    p.add_argument("--domain", default="CTY24", choices=["CTY24", "LC10"],
                   help="Which label/flag/conf family to use from the merged parquet.")
    p.add_argument("--model-hash", default=None, help="Filter embeddings by model_hash.")
    p.add_argument("--ref-ids", nargs="+", default=None, help="Restrict to these ref_id values.")
    p.add_argument("--max-rows", type=int, default=None, help="Sub-sample N embedding rows.")
    # Column overrides (auto-filled from the loader when --duckdb is used) ----
    p.add_argument("--label-col", default=None)
    p.add_argument("--flag-col", default=None)
    p.add_argument("--conf-col", default=None)
    p.add_argument("--region-col", default=None)
    p.add_argument("--quality-col", default=None)
    p.add_argument("--group-col", default=None,
                   help="Group column for leakage-safe split (e.g. ref_id or h3_l3_cell).")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--quality-threshold", type=float, default=0.9)
    p.add_argument("--min-class-count", type=int, default=20)
    p.add_argument("--out-csv", required=True, type=Path)
    args = p.parse_args()

    label_col, flag_col, conf_col = args.label_col, args.flag_col, args.conf_col
    region_col = args.region_col
    if args.duckdb is not None:
        if args.merged_parquet is None:
            p.error("--duckdb requires --merged-parquet")
        from data_loader import load_unified
        df, names = load_unified(
            args.duckdb, args.merged_parquet,
            domain=args.domain, model_hash=args.model_hash,
            ref_ids=args.ref_ids, region_parquet=None,
            quality_parquet=None, max_rows=args.max_rows,
        )
        # Loader names win unless the user explicitly overrode them.
        label_col = label_col or names["label_col"]
        flag_col = flag_col or names["flag_col"]
        conf_col = conf_col or names["conf_col"]
        region_col = region_col or names.get("region_col")
        print(f"[proxy] loaded {len(df):,} rows via DuckDB+merged "
              f"(domain={args.domain}, label={label_col})")
    elif args.scored_parquet is not None:
        df = pd.read_parquet(args.scored_parquet)
        print(f"[proxy] loaded {len(df):,} rows from {args.scored_parquet}")
    else:
        p.error("provide either --scored-parquet or (--duckdb and --merged-parquet)")
    if not (label_col and flag_col and conf_col):
        p.error("label/flag/conf columns must be set (via args or the loader)")

    results = run(
        df,
        label_col=label_col,
        flag_col=flag_col,
        conf_col=conf_col,
        region_col=region_col,
        quality_col=args.quality_col,
        group_col=args.group_col,
        seeds=args.seeds,
        test_frac=args.test_frac,
        quality_threshold=args.quality_threshold,
        min_class_count=args.min_class_count,
    )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out_csv, index=False)
    agg = aggregate(results)
    agg.to_csv(args.out_csv.with_name(args.out_csv.stem + "_agg.csv"), index=False)
    print(f"[proxy] wrote {len(results)} rows -> {args.out_csv}")
    print(agg.to_string())


if __name__ == "__main__":
    main()
