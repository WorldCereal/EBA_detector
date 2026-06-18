# Outlier-removal experiments

This directory contains the full experimental program that turns *"we can flag
outliers in embedding space"* into a publishable, evidence-backed claim. There
are three pillars, each answering a different reviewer question:

| Pillar | Question it answers | Entry point |
|---|---|---|
| **A. Intrinsic validation** | *Are the flagged points actually label errors?* | `run_validation.py` |
| **B. Downstream proxy (Tier 1)** | *Does treating outliers in TRAIN help a classifier on a CLEAN test set? (fast sweep)* | `run_catboost_proxy.py` |
| **C. Operational fine-tuning (Tier 2)** | *Does it help the real WorldCereal model — with the vanilla vs. fine-tuned encoder?* | `run_worldcereal_finetune.py` |

Results from all three are collated for the paper by `aggregate_results.py`.

---

## 0. Prerequisites

* The embeddings DuckDB cache built by `scripts/compute_anomaly_scores.py`.
* A **training parquet with the 6 anomaly columns written back** (run the CLI
  in `--mode rerun`), plus three extra columns you must ensure exist:
  * `region` — macro-region label (the balanced-splits builder in
    `worldcereal-classification/scripts/training/splits/create_balanced_splits.py`
    already assigns macro-regions; reuse that mapping).
  * `quality_score_*` — annotation quality (already present in WorldCereal RDM
    exports). Used to define the **detector-independent clean test set**.
  * `split` — a fixed `train`/`test` assignment (use the balanced-splits
    builder so the split is spatially disjoint and reused across all scenarios).

All paths and the grid live in `config.yaml`.

---

## 1. The scenario matrix (`scenarios.py`)

Every scenario is applied to the **training split only**. The test split is
fixed across scenarios; evaluation is reported on three *views*:

* `clean` — high-quality / gold subset, **defined independently of the
  detector** (primary metric).
* `full` — all test points (shows robustness to residual test-set noise).
* `minus_flagged` — test points the detector flags removed (diagnostic: how much
  of any metric change is the test set merely getting easier).

| Scenario | Train treatment |
|---|---|
| `baseline` | none (reference) |
| `drop_candidate` / `drop_suspect` / `drop_flagged` | hard removal at rising severity |
| `downweight_linear` / `_power2` / `_power4` | weight by `confidence_nonoutlier^p` |
| `filter_conf_0.90` / `_0.95` | drop train rows below a confidence threshold |
| `drop_suspect+downweight` | hybrid: remove worst, soft-weight the rest |

`python -c "from experiments.scenarios import scenario_grid_summary as s; print(s().to_string())"`
prints the matrix for the paper appendix.

---

## 2. Pillar A — intrinsic validation (synthetic noise)

```bash
python run_validation.py \
    --parquet /data/embeddings_sample.parquet \
    --label-col CTY24_cls --h3-col h3_l3_cell --group-col ref_id \
    --out-csv results/validation_sweep.csv
```

Injects three error models — `within_context` (confusable-class flip), `random`
(gross error), and `parcel` (whole-field systematic error) — at several rates
and seeds, then reports detection **AUROC**, **average precision**, and
**precision@k**. The `parcel` mode specifically stresses the masking failure
that point-wise kNN scoring is prone to, and demonstrates the value of the
robust-centroid + parcel-aware additions.

→ Paper table: *"Detection performance under controlled label noise."*

---

## 3. Pillar B — Tier-1 CatBoost proxy (fast)

```bash
export SCORED_PARQUET=/data/training_with_anomaly.parquet
sbatch slurm/catboost_proxy.slurm           # or run the python directly
```

Trains CatBoost on the frozen embeddings, sweeping the full
**scenario × region × seed** grid (minutes–hours, no GPU fine-tuning). Use it to
prune the scenario set before spending GPU time in Tier 2.

→ Paper table: *"Downstream F1 vs. outlier treatment (proxy), per region."*

---

## 4. Pillar C — Tier-2 WorldCereal fine-tuning (headline)

```bash
export SCORED_PARQUET=/data/training_with_anomaly.parquet
export PRESTO_VANILLA=/models/presto_vanilla.pt
export PRESTO_FINETUNED=/models/presto_worldcereal_ft.pt
sbatch --array=0-299%32 slurm/finetune_array.slurm
```

Each array task trains one grid cell via `worldcereal.train.downstream.TorchTrainer`
(the operational head trainer), mapping each scenario onto its native
`outlier_drop_mode` / `outlier_score_col` arguments. The `encoder` axis
(`vanilla` vs `finetuned`) is the crux of this paper: it shows whether outlier
removal helps **before** the encoder has seen the data (current setup) and
whether the benefit survives **after** fine-tuning — and motivates the
detect→clean→fine-tune→re-detect loop discussed in the paper.

→ Paper table: *"Operational crop-type F1 by encoder × region × treatment."*

---

## 5. Collate for the paper

```bash
python aggregate_results.py \
    --proxy-agg-csv results/proxy_<jobid>_agg.csv \
    --finetune-runs-dir runs/ \
    --out-dir results/paper_tables
```

Writes `table_main.csv`, `table_delta.csv` (Δf1 vs baseline — the headline
numbers), and `fig_delta_by_scenario.png`. Drop these straight into the LaTeX
result tables.

---

## Reproducibility notes

* All splits are spatial / group-based (no parcel leakage between train and test).
* The clean test view never uses the detector, so improvements cannot be an
  artefact of the test set becoming easier — the `minus_flagged` view makes that
  check explicit.
* `centroid_mode`, `apply_slice_trust`, and `gate_confidence_by_flag` are
  exposed in `run_pipeline` so the **ablations** (mean vs trimmed centroid,
  trust gate on/off, flag-gated vs rank-driven confidence) reuse the same grid.
