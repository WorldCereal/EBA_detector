#!/usr/bin/env python3
"""Intrinsic detector validation via synthetic label-noise injection.

Loads an embeddings parquet (with a clean label column, an H3 cell column, and
optionally a parcel/group column), injects controlled label noise, re-runs the
detector on the corrupted labels, and reports how well it recovers the planted
errors (AUROC, average precision, precision@k).  This is the ground-truthed
evidence that the flagged points are genuine label errors — independent of any
downstream model.

Example
-------
python run_validation.py \
    --parquet /data/embeddings_sample.parquet \
    --label-col CTY24_cls --h3-col h3_l3_cell --group-col ref_id \
    --out-csv results/validation_sweep.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from outlier_embeddings.validation import sweep  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", required=True, type=Path)
    p.add_argument("--label-col", required=True)
    p.add_argument("--h3-col", default="h3_l3_cell")
    p.add_argument("--group-col", default=None, help="Parcel/field id (enables parcel-noise mode).")
    p.add_argument("--modes", nargs="+", default=["within_context", "random", "parcel"])
    p.add_argument("--rates", type=float, nargs="+", default=[0.02, 0.05, 0.10, 0.20])
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--score-col", default="S", choices=["S", "mean_score"])
    p.add_argument("--out-csv", required=True, type=Path)
    args = p.parse_args()

    df = pd.read_parquet(args.parquet)
    print(f"[validation] loaded {len(df):,} rows")

    res = sweep(
        df,
        modes=tuple(args.modes),
        rates=tuple(args.rates),
        seeds=tuple(args.seeds),
        label_col=args.label_col,
        h3_col=args.h3_col,
        group_col=args.group_col,
        score_col=args.score_col,
    )
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out_csv, index=False)

    agg = (
        res.groupby(["mode", "rate"], as_index=False)
        .agg(auroc=("auroc", "mean"),
             average_precision=("average_precision", "mean"),
             precision_at_0p05=("precision_at_0.05", "mean"),
             precision_at_0p1=("precision_at_0.1", "mean"),
             flag_precision=("flag_precision", "mean"),
             flag_recall=("flag_recall", "mean"))
    )
    agg.to_csv(args.out_csv.with_name(args.out_csv.stem + "_agg.csv"), index=False)
    print(agg.to_string())


if __name__ == "__main__":
    main()
