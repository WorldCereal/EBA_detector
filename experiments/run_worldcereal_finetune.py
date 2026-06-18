#!/usr/bin/env python3
"""Tier-2 experiment runner: real WorldCereal Presto fine-tuning / head training.

This drives the *actual* downstream training used by WorldCereal
(``worldcereal.train.downstream.TorchTrainer``) so the headline numbers in the
paper come from the operational model, not a proxy.  It sweeps the same
outlier-treatment scenarios as the Tier-1 proxy across regions, seeds, and —
crucially for this paper — across **encoders** (the vanilla, un-fine-tuned
Presto vs. a fine-tuned checkpoint), which is how we test whether outlier
removal still helps once the embedding space has seen the training data.

It does not modify the worldcereal repo: ``TorchTrainer`` already accepts
``outlier_col`` / ``outlier_score_col`` / ``outlier_drop_mode`` /
``presto_model_path`` / ``split_column``.  We map each :class:`Scenario` onto
those arguments (precomputing a per-sample weight column for the continuous
down-weighting scenarios, which TorchTrainer multiplies in via
``attach_sample_weights``).

Run it inside the worldcereal training environment (the one that can import
``worldcereal`` and load Presto).  Typically launched per grid-cell by the
SLURM array in ``slurm/finetune_array.slurm``.

Example (single cell of the grid)
---------------------------------
python run_worldcereal_finetune.py \
    --data-parquet /data/training_with_anomaly.parquet \
    --head-task croptype --season-id tc-s1 \
    --encoder vanilla --presto-vanilla-path /models/presto_vanilla.pt \
    --region Europe --scenario drop_suspect --seed 0 \
    --split-column split \
    --label-col downstream_class \
    --flag-col CTY24_anomaly_flag --conf-col CTY24_confidence_nonoutlier \
    --output-dir runs/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for sibling 'scenarios'
from scenarios import DEFAULT_SCENARIOS  # noqa: E402

SCENARIO_BY_NAME = {s.name: s for s in DEFAULT_SCENARIOS}


def _precompute_scenario_weight(
    df: pd.DataFrame, scenario, conf_col: str, out_col: str = "scenario_weight"
) -> pd.DataFrame:
    """Materialise a per-sample weight column for the continuous scenarios.

    TorchTrainer multiplies ``outlier_score_col`` into the final sample weight,
    so we can express *any* confidence transform (linear / power / filter) by
    precomputing it here and pointing ``outlier_score_col`` at *out_col*.
    Hard-drop scenarios leave the weight at 1.0 (removal is handled by
    ``outlier_drop_mode``).
    """
    df = df.copy()
    conf = pd.to_numeric(df.get(conf_col, 1.0), errors="coerce").fillna(1.0).clip(0, 1)
    if scenario.weight_mode == "none":
        w = np.ones(len(df))
    elif scenario.weight_mode == "conf_linear":
        w = conf.to_numpy()
    elif scenario.weight_mode == "conf_power":
        w = (conf ** float(scenario.weight_power)).to_numpy()
    elif scenario.weight_mode == "conf_filter":
        w = (conf >= float(scenario.conf_threshold)).astype(float).to_numpy()
    else:
        raise ValueError(f"Unknown weight_mode: {scenario.weight_mode}")
    df[out_col] = w
    return df


def run_one(args: argparse.Namespace) -> dict:
    """Run a single grid cell and return a metrics dict."""
    from worldcereal.train.downstream import TorchTrainer

    scenario = SCENARIO_BY_NAME[args.scenario]
    df = pd.read_parquet(args.data_parquet)

    # Optional region restriction (train pool only; keep the global fixed test
    # set if you want cross-region transfer — controlled by --region-col).
    if args.region and args.region != "ALL" and args.region_col in df.columns:
        if args.region_scope == "train_only":
            # keep all test rows, restrict only train rows to the region
            mask = (df[args.split_column] == "test") | (
                df[args.region_col].astype(str) == args.region
            )
        else:
            mask = df[args.region_col].astype(str) == args.region
        df = df[mask].copy()

    # Precompute the continuous-weight column for this scenario.
    df = _precompute_scenario_weight(df, scenario, args.conf_col)

    # Map scenario -> TorchTrainer outlier arguments.
    outlier_col = args.flag_col if scenario.drop_mode != "keep" else None
    outlier_score_col = (
        "scenario_weight" if scenario.weight_mode != "none" else None
    )

    presto_path = (
        args.presto_vanilla_path if args.encoder == "vanilla"
        else args.presto_finetuned_path
    )

    run_name = f"{args.encoder}__{args.region}__{args.scenario}__seed{args.seed}"
    out_dir = Path(args.output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = TorchTrainer(
        embeddings_df=df,
        split_column=args.split_column,
        head_type=args.head_type,
        head_task=args.head_task,
        season_id=args.season_id,
        output_dir=out_dir,
        modelversion=run_name,
        presto_model_path=presto_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        use_balancing=True,
        quality_col=args.quality_col,
        outlier_score_col=outlier_score_col,
        outlier_col=outlier_col,
        outlier_drop_mode=scenario.drop_mode,
        # Keep evaluation honest: do NOT let the detector filter the test set
        # inside the trainer.  The fixed 'test' split is used as-is; the
        # clean/minus_flagged views are produced offline by aggregate_results.py
        eval_weight_floor=None,
    )

    trainer.train()

    # TorchTrainer writes a manifest/metrics file; surface a compact record.
    record = {
        "run_name": run_name,
        "encoder": args.encoder,
        "region": args.region,
        "scenario": args.scenario,
        "drop_mode": scenario.drop_mode,
        "weight_mode": scenario.weight_mode,
        "seed": args.seed,
        "head_task": args.head_task,
        "season_id": args.season_id,
        "output_dir": str(out_dir),
        "n_rows": int(len(df)),
    }
    # Try to attach the metrics TorchTrainer saved (best-effort, schema-tolerant).
    for cand in out_dir.glob("*.json"):
        try:
            with open(cand) as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                record.setdefault("trainer_metrics", {})[cand.name] = payload
        except Exception:
            pass

    with open(out_dir / "experiment_record.json", "w") as f:
        json.dump(record, f, indent=2, default=str)
    print(f"[finetune] done: {run_name} -> {out_dir}")
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-parquet", required=True, type=Path)
    p.add_argument("--head-task", choices=["croptype", "landcover"], default="croptype")
    p.add_argument("--head-type", choices=["linear", "mlp"], default="mlp")
    p.add_argument("--season-id", default="tc-s1")
    p.add_argument("--encoder", choices=["vanilla", "finetuned"], default="vanilla")
    p.add_argument("--presto-vanilla-path", default=None)
    p.add_argument("--presto-finetuned-path", default=None)
    p.add_argument("--region", default="ALL")
    p.add_argument("--region-col", default="region")
    p.add_argument("--region-scope", choices=["train_only", "all"], default="train_only")
    p.add_argument("--scenario", default="baseline", choices=list(SCENARIO_BY_NAME))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-column", default="split")
    p.add_argument("--label-col", default="downstream_class")
    p.add_argument("--flag-col", default="CTY24_anomaly_flag")
    p.add_argument("--conf-col", default="CTY24_confidence_nonoutlier")
    p.add_argument("--quality-col", default="quality_score_ct")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--output-dir", default="runs", type=Path)
    args = p.parse_args()
    run_one(args)


if __name__ == "__main__":
    main()
