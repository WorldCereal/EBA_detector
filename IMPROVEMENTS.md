# Outlier workflow: improvements, fixes, and paper mapping

This document records the logical flaws found in the embedding-based outlier
pipeline, the fix applied for each, how it was validated, and where it is
reflected in the paper. It is the review companion to the code changes.

All changes are **backward-compatible**: every new behaviour is exposed as a
parameter with a default, and the legacy behaviour is recoverable by flipping
that parameter. The existing 56 unit tests still pass; 16 new tests were added
(72 total, all green: `pytest tests/ -q`).

---

## A. Logical flaws fixed in the core pipeline

### A1. Centroid masking (outliers hid themselves)
**Flaw.** The slice reference centroid was the plain mean of all embeddings,
*including* the outliers. With up to 10% contamination the mean is pulled
toward the anomalous mass, deflating those points' cosine distance — the
classical *masking* problem. The strongest, most removal-worthy errors were the
ones most able to hide.

**Fix.** `robust_centroid()` in `anomaly_utils.py`: an iterative trimmed-mean
(default) that drops the farthest `trim_frac` of points and recentres on the
inlier core (median mode also available). Wired through
`compute_slice_centroids`, `compute_scores_for_slice`, `_score_group_simple`,
`score_slices_hierarchical`, and exposed in `run_pipeline(centroid_mode=...,
centroid_trim=...)`.

**Validation.** `tests/test_robustness.py::TestRobustCentroid::test_trimmed_resists_masking`
constructs a contaminated slice and shows the outliers are measurably *farther*
from the trimmed centroid than from the plain mean.

**Paper.** Methods §"Robust reference centroid"; ablation axis (a).

### A2. Confidence inconsistent with the flag, and clean slices penalised
**Flaw.** `confidence_nonoutlier` was derived from `mean_score`, which is built
from **within-slice rank** statistics. The top-ranked point of *every* slice —
including a perfectly clean one — gets a high score, so its confidence dropped
and it was down-weighted downstream even though it was never flagged. The
continuous confidence and the discrete `anomaly_flag` could therefore disagree.

**Fix.** `add_confidence_from_score(..., flagged_col=...)` clamps confidence to
1.0 for non-flagged samples; the decay now only modulates the severity of
already-flagged points. Wired via `run_pipeline(gate_confidence_by_flag=True)`.

**Validation.** `TestConfidenceGating` (unflagged → exactly 1.0; legacy path
still penalises).

**Paper.** Methods §"Consistency of the confidence and flag outputs"; ablation (c).

### A3. MAD threshold used a scale-mismatched magic fallback
**Flaw.** In `flag_anomalies` MAD mode, `mad==0` fell back to a constant `1.0`.
Because `S in [0,1]` this silently flagged nothing, but the constant is
meaningless on any other score scale and obscured intent.

**Fix.** Explicit degenerate-slice handling: when `mad<=0` (>50% identical
scores, no robust scale) the slice flags nothing, by design and documented.

**Validation.** `TestMadDegeneracy::test_mad_zero_flags_nothing`.

**Paper.** Methods §"Robust thresholding details".

### A4. Adaptive-H3 small slices could never merge (coverage gap in sparse regions)
**Flaw.** In adaptive mode `effective_h3_cell` mixes H3 resolutions, but
`merge_small_slices` searched neighbours with `grid_disk`, which only returns
*same-resolution* cells. A small fine-level (L3) slice next to a coarse-resolved
(L2) region found no existing neighbour and stayed permanently undersized →
**never scored**. This disproportionately hit sparse regions (e.g. Africa) — the
under-represented areas that matter most.

**Fix.** Resolution-aware neighbour construction: candidates now include the
cell's parents at any coarser resolution present in the column, restricted to
cells that actually exist. A small L3 slice can merge into the L2 cell that
contains it.

**Validation.** Covered by the existing merge tests plus the smoke run on
synthetic mixed-resolution data; logic reviewed for fixed-level back-compat
(single resolution → identical to before).

**Paper.** Methods §"Robust thresholding details".

---

## B. New capabilities that make removal *sellable*

### B1. Slice-trust / separability gate (the frozen-encoder problem)
**Gap.** Distance-to-reference is only meaningful if the embedding geometry
separates classes in that slice. With a **frozen, un-fine-tuned** Presto there is
no such guarantee per region, so a high score can reflect an encoder blind spot
rather than label noise. This is the single biggest threat to the central claim.

**Addition.** `compute_slice_trust()` scores each context's geometry
(silhouette-style separation for multi-label contexts; concentration ratio for
single-label). `apply_trust_to_confidence()` attenuates the penalty of flags in
low-trust slices; `downgrade_flags_low_trust()` softens categories there. Wired
via `run_pipeline(apply_slice_trust=True, slice_trust_min=...)`.

**Validation.** `TestSliceTrust` (separated context → trust > 0.8, entangled →
< 0.5; attenuation and downgrade behave as specified). Smoke run showed
trust 0.996 vs 0.0 on separated vs entangled clouds.

**Paper.** Methods §"Slice trust and separability gating"; Discussion
§"The frozen-encoder caveat"; ablation (b).

### B2. Parcel / group awareness (systematic field-level errors)
**Gap.** Many samples are multiple points from one parcel; siblings are
near-duplicates, so a *whole* mislabelled parcel looks normal (its neighbours
are its own points). Point-wise kNN cannot see this — a likely cause of the
"misses some more" failures.

**Addition.** `parcel_aware_slice_scores()` computes kNN distance excluding
same-parcel neighbours, exposing whole-wrong-parcels; `aggregate_parcel_scores()`
produces a parcel-level anomaly score.

**Validation.** `TestParcelAware::test_wrong_parcel_scores_high` (the planted
wrong parcel is ranked most anomalous).

**Paper.** Methods §"Parcel- and group-aware scoring"; ablation (d).

### B3. Intrinsic synthetic-noise validation (`validation.py`)
**Gap.** Nothing measured whether the detector finds *known* errors; the whole
case rested on the downstream model, which can move for many reasons.

**Addition.** `validation.py`: inject controlled label noise (within-context /
random / parcel) → re-score → report detection AUROC, average precision,
precision@k, and realised flag precision/recall. Self-contained (no
worldcereal/DuckDB), so it is reproducible and unit-tested.

**Validation.** `TestValidation` (noise truth bookkeeping, parcel-mode
whole-group corruption, detector recovers planted errors at AUROC > 0.85 on
separable data, degenerate-case handling).

**Paper.** Experiments §"Pillar A"; Table `tab:synthetic`.

---

## C. Experiment harness (`experiments/`)

Closes the validity gaps in the original downstream plan and matches the
request for region-stratified, drop/score/down-weight experiments on
train/test sets with and without outliers.

- **`scenarios.py`** — 10 train-side treatments (baseline; drop candidate/
  suspect/flagged; downweight linear/power2/power4; filter conf 0.90/0.95;
  hybrid). Pure, unit-tested (`TestScenarios`).
- **Honest evaluation** — `build_test_views()` produces three fixed test views:
  `clean` (detector-independent, by annotation quality), `full`, and
  `minus_flagged`. The original plan risked circularity by filtering the eval
  set with the same detector; the `clean` vs `minus_flagged` comparison makes
  "did the model improve or did the test get easier?" explicit.
- **Tier 1 `run_catboost_proxy.py`** — fast CatBoost-on-embeddings sweep over
  scenario × region × seed, leakage-safe group splits, evaluates on the fixed
  views.
- **Tier 2 `run_worldcereal_finetune.py`** — drives the real
  `worldcereal.train.downstream.TorchTrainer` (native `outlier_drop_mode` /
  `outlier_score_col` / `presto_model_path`); sweeps the **vanilla vs
  fine-tuned encoder** axis — the headline comparison for this paper. No edits
  to the worldcereal repo required.
- **`run_validation.py`**, **`aggregate_results.py`**, **`config.yaml`**,
  **`slurm/*.slurm`** (incl. a 300-cell job array), **`README.md`**.

**Paper.** Experiments §"Pillar B" / §"Pillar C" / §"Ablations"; Tables
`tab:scenarios`, `tab:proxy`, `tab:finetune`.

---

## D. Recommended config changes (not hard-coded)

These are defensible defaults to consider for the production runs, documented
rather than forced:

1. For the score feeding the MAD flag, avoid clipping the high tail of the
   percentile normalisation (e.g. `norm_percentiles=(2.0, 100.0)`) so the
   strongest outliers are not compressed back toward the inlier mass before
   thresholding.
2. Set `centroid_trim` >= the configured `max_flagged_fraction`.
3. Run the vanilla→clean→fine-tune→re-detect loop (Discussion) at least once;
   Pillar C provides the first iteration.

---

## E. How to reproduce the checks

```bash
# unit tests (incl. the new robustness/validation/scenario tests)
cd outlier_embeddings && PYTHONPATH=src python3 -m pytest tests/ -q   # 72 passed

```
