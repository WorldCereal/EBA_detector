# EBA Detector <!-- omit in toc -->

**Embedding-Based Anomaly detection for cleaning crop-type and land-cover reference datasets.**

`EBA_detector` finds anomalous / potentially mislabelled samples in large Earth-observation
reference datasets (such as those used by [WorldCereal](https://github.com/WorldCereal/worldcereal-classification))
by operating directly on pre-computed **embedding vectors** from a pretrained geospatial encoder.
Rather than reasoning over raw multi-temporal signals, it scores each labelled sample against
other samples of the same class in the same locality, flags the ones that stand out, and emits a
per-sample confidence that downstream training can act on (remove or down-weight).

This repository accompanies the paper **"Embeddings-based Anomaly Detection for Cleaning Global
Crop-Type Reference Datasets"** ([arXiv:2607.23908](https://arxiv.org/abs/2607.23908), [HTML](https://arxiv.org/html/2607.23908v1), [PDF](https://arxiv.org/pdf/2607.23908))

## News: 
Accepted for spotlight oral presentation at **[TerraBytes II](https://terrabytes-workshop.github.io/spotlight)** workshop at ECCV 2026, Malmö, Sweden. (see [Citation](#citation)).

---

## Overview

A geospatial foundation model produces an embedding for every reference sample. On top of those
embeddings, this package provides a self-contained pipeline that:

1. **Loads** embeddings from a DuckDB cache (built once, updated incrementally).
2. **Groups** samples into local *slices* by H3 cell + label class (adaptive multi-resolution).
3. **Scores** each sample within its slice using cosine distance to the class centroid and mean
   distance to its k nearest neighbours.
4. **Flags** statistical outliers per slice (MAD / percentile / FDR), with robustness safeguards:
   a contamination-resistant trimmed centroid, a slice-trust gate that softens flags where the
   embedding geometry is uninformative, and optional group-level aggregation that surfaces whole
   datasets that are systematically off.
5. **Grades & scores** each sample into `normal / flagged / suspect / candidate` and a continuous
   `confidence_nonoutlier` in `[0, 1]`.
6. **Writes** the anomaly columns back to the reference parquet files.

An **incremental update** mode re-scores only the geographic impact zone of newly added datasets,
so the whole collection need not be reprocessed.

### Encoder-agnostic

The detector treats embeddings as opaque `d`-dimensional vectors and does not depend on the
architecture that produced them. In this work the primary embeddings come from a Presto-style
encoder (128-d) as used in WorldCereal, but the same pipeline runs unchanged on other
representations — we include tooling and notebooks for scoring
[AlphaEarth](https://arxiv.org/abs/2507.22291) embeddings (64-d) and **soon** to be included **TESSERA v1.1** embeddings,
and further encoders (e.g. Clay, Prithvi) are a natural extension.

---

## Repository layout

```
EBA_detector/
├── pyproject.toml
├── README.md
├── LICENSE
│
├── src/EBA_detector/
│   ├── anomaly_utils.py        # Stateless helpers: scoring, metrics, normalization,
│   │                           # flagging, adaptive H3, incremental utilities
│   ├── anomaly.py              # Pipeline orchestration: load -> map -> score -> flag -> write
│   ├── calibration.py          # Cross-slice null; absolute z-scores (the absolute gate)
│   ├── quality.py              # Embedding / encoder / H3 validation before scoring
│   ├── robust_extensions.py    # Trimmed centroid, slice-trust gate, group aggregation
│   ├── embeddings_cache.py     # DuckDB-backed embedding cache (Presto, 128-d)
│   ├── validation.py           # Synthetic label-noise injection & recovery metrics
│   └── experiments.py          # CatBoost-on-embeddings downstream experiments
│
├── scripts/
│   ├── compute_anomaly_scores.py    # Full CLI pipeline: parquets -> embeddings -> scores -> write-back
│   └── get_mappings_from_legend.py  # Fetch the legend / class-mapping table (SharePoint)
│
├── experiments/                # Evaluation harness used for the paper
│   ├── run_validation.py            # Synthetic-noise recovery (AUROC / AP / precision@k)
│   ├── run_catboost_proxy.py        # Fast CatBoost proxy: does cleaning the train set help?
│   ├── run_worldcereal_finetune.py  # Operational WorldCereal fine-tuning driver
│   ├── scenarios.py / data_loader.py / aggregate_results.py
│   └── slurm/                       # Example SLURM submit scripts
│
├── notebooks/
│   ├── compute_outlier_scores.ipynb            # Drive the full pipeline interactively
│   ├── compute_outlier_scores_newmodel.ipynb   # Same, on a fine-tuned encoder
│   ├── compute_outlier_scores_alphaearth.ipynb # Scoring on AlphaEarth embeddings
│   ├── alphaearth_outlier_scoring.ipynb        # AlphaEarth index build + scoring
│   └── explore_sample_embeddings.ipynb         # Exploration / visualisation
│
└── tests/                      # Unit tests (synthetic data, no DuckDB, no network)
```

---

## Dependencies

The package works **alongside** the main `worldcereal-classification` package (installed from its
`main` branch), from which it imports a few utilities:

| Import | Where used | Purpose |
|---|---|---|
| `worldcereal.utils.refdata.get_class_mappings` / `map_classes` | `anomaly.py` | Map `ewoc_code` -> label class |
| `worldcereal.train.datasets.WorldCerealTrainingDataset` | `embeddings_cache.py`| DataLoader for embedding inference |
| `worldcereal.utils.timeseries.process_parquet` | `compute_anomaly_scores.py` | Pivot long -> wide parquets |

Everything else (scoring, flagging, caching, incremental updates, validation) lives entirely
inside this package.

---

## Installation

```bash
# 1) WorldCereal (foundation utilities), from main
pip install "worldcereal[train] @ git+https://github.com/WorldCereal/worldcereal-classification.git@main"

# 2) EBA_detector (this package)
pip install "EBA_detector @ git+https://github.com/WorldCereal/EBA_detector.git"
# or, from a local checkout (editable install, recommended for development):
pip install -e /path/to/EBA_detector
```

Optional extras:

```bash
pip install "EBA_detector[experiments]"  # CatBoost experiments
pip install "EBA_detector[sharepoint]"   # SharePoint legend fetching
pip install "EBA_detector[notebooks]"    # Jupyter environment
pip install "EBA_detector[gee]"          # Google Earth Engine (AlphaEarth extraction)
pip install "EBA_detector[all]"          # Everything
pip install "EBA_detector[dev]"          # Tests + linting
```

> The Presto backbone is provided by `prometheo`:
> `pip install "prometheo @ git+https://github.com/WorldCereal/prometheo.git@v0.0.5"`

---

## Quick start

### Python API

```python
from EBA_detector.anomaly import run_pipeline

flagged_gdf, summary_df = run_pipeline(
    embeddings_db_path="/path/to/embeddings_cache.duckdb",
    h3_level=[2, 3],           # adaptive: try L2 first, fall back to L3
    group_cols=["ref_id"],
    min_slice_size=100,
    # Locality with a dispersion the slice cannot inflate
    threshold_mode="stable_mad", mad_k=3.3,
    # Cross-slice absolute gate (see "How a sample gets flagged")
    require_absolute=True, abs_z_k=3.3,
    # Scale measured from the clean left half, so contamination present in
    # every slice cannot inflate the null
    null_scale_estimator="left_tail",
    # Should be >= the largest contamination you expect in a slice
    centroid_mode="trimmed", centroid_trim=0.45,
    # Compare like with like in time — embeddings are season-specific
    time_col="year",
    gate_confidence_by_flag=True,
    output_samples_path="/path/to/outlier_scores.parquet",
    output_summary_path="/path/to/outlier_summary.parquet",
)
```

### Command line (full pipeline)

```bash
python scripts/compute_anomaly_scores.py \
    --mode rerun \
    --input-format geoparquet \
    --input-long-dir  /path/to/reference_parquets \
    --output-long-dir /path/to/reference_parquets_with_anomaly \
    --embeddings-db-path /path/to/embeddings_cache.duckdb \
    --wide-dir /path/to/cached_wide_parquets \
    --group-cols ref_id --time-col year \
    --abs-z-k 3.3 --strict-quality \
    --lc10-h3-levels 2 3 --lc10-min-slice-size 100 --lc10-mad-k 3.0 \
    --cty24-h3-levels 2 3 --cty24-min-slice-size 100 --cty24-mad-k 3.0

# Incremental: re-score only the impact zone of newly added datasets
python scripts/compute_anomaly_scores.py --mode update \
    --input-long-dir  /path/to/reference_parquets \
    --output-long-dir /path/to/reference_parquets_with_anomaly \
    --embeddings-db-path /path/to/embeddings_cache.duckdb
```

---

## Output columns

The pipeline appends per-sample anomaly columns to the reference parquet files, for each label
domain (LANDCOVER10 / CROPTYPE24):

| Column | Description |
|---|---|
| `LC10_confidence_nonoutlier` | Confidence that the sample is **not** an outlier under LC10 — float32 in `[0, 1]` |
| `LC10_anomaly_flag` | Category under LC10 (see below) |
| `outlier_LC10_cls` | Label class used for LC10 scoring |
| `CTY24_confidence_nonoutlier` | As above, under the CROPTYPE24 mapping |
| `CTY24_anomaly_flag` | Category under CTY24 |
| `outlier_CTY24_cls` | Label class used for CTY24 scoring |

### Flag values

| Value | Meaning |
|---|---|
| `normal` | Examined, no evidence of a problem |
| `flagged` | Unusual relative to its slice **and** in absolute terms |
| `suspect` | Corroborated by ≥2 independent signals |
| `candidate` | Strongly corroborated — the removal-worthy tier |
| `unscored` | Slice too small to score. **Not** the same as `normal` |
| `unscorable` | Embedding failed the quality gate (zero-norm, non-finite, duplicate id) |
| `unmapped` | `ewoc_code` absent from the legend — a coverage gap, not a data quirk |
| `skipped` | held out by `skip_classes` |

The last four are terminal states: they record that the detector *could not
form an opinion*. Treating them as `normal` (the previous behaviour) hides the
blind spot from downstream training and makes the incremental `--mode update`
pathway rediscover the same rows forever.

Alongside these, the review parquets keep the **evidence** behind each call —
`abs_z`, `cosine_distance`, `knn_distance`, `neighbourhood_offset`,
`knn_same_label_frac_ctx`, `alt_margin_ctx`, `escalation_votes`,
`weak_support`, `purity_veto`, `corroborated`, `quality_reason` — so a flag can
be audited against a basemap instead of taken on trust.

---

## How a sample gets flagged

Two conditions must both hold. This is the core of the method and the part
that changed most recently.

1. **Locally unusual.** Its distance exceeds the slice's own median by
   `mad_k` times a dispersion borrowed from the cross-slice null
   (`threshold_mode="stable_mad"`).
2. **Absolutely unusual.** Its `abs_z` exceeds `abs_z_k` robust sigma against
   a null pooled across many slices of the same class.

Why both are needed: every within-slice score is percentile-normalised per
slice, so on its own it cannot distinguish *the most unusual point of a clean
slice* from *a mislabelled point* — the top ~2 % of every slice clears a rank
quantile whether or not the slice contains a single error. And a within-slice
`median + k·MAD` gate is not a fix, because a contaminated slice inflates its
own median and MAD: measured on synthetic data the legacy rule flagged **0 %**
at both 2 % and 30 % true contamination, but 9 % at 10 %.

The null is built from **one summary statistic per slice**, so a contaminated
slice contributes a single observation and cannot calibrate its own errors
away.

Escalation to `suspect` / `candidate` requires agreement across signals that
measure genuinely different things — absolute distance, kNN **label purity**,
the **alt-class margin**, and within-slice rank. (The previous 2-of-3 vote over
`S_rank`, `S_rank_min` and `S_z` was not a consensus: the first two are the
mean and the min of the same two rank vectors.) Two safeguards apply on top:

- **Purity veto** — a point whose neighbours overwhelmingly share its label is
  capped at `flagged`. Being an unusual *example* of a class is not evidence of
  a wrong label when the neighbourhood agrees.
- **No corroboration, no strong claim** — where the context contains a single
  label there is no alternative class to have been confused with, so escalation
  is capped at `flagged`.

Set `require_absolute=False` (CLI: `--no-absolute-gate`) to reproduce the
legacy relative-only behaviour for ablations.

### Slice vs context

`group_cols` (default `[]`) defines the **slice**: the reference cloud a sample
is scored against. Pooling across source datasets is deliberate — comparing a
sample with the same class in the same locality *whoever digitised it* is the
assumption the method rests on. Splitting by `ref_id` discards that comparison,
fragments small datasets below `min_slice_size` into `unscored` while large ones
are fully scrutinised, and makes a wholly-mislabelled dataset undetectable
because it becomes its own self-consistent slice (measured: `abs_z` −0.15 and
0/360 flagged, versus `abs_z` 5.45 and 339/360 pooled).

`context_group_cols` (default `[]`) defines the **context**: the neighbourhood
the kNN-purity and alt-class-margin signals are measured over. It deliberately
does *not* inherit `group_cols` — "what else is on the ground around this point?"
is a geographic question, and restricting it to the same `ref_id` makes those
signals collapse for single-crop datasets (every point trivially has purity 1.0
and there is no alternative class centroid to measure a margin against).

### Null localisation

The cross-slice null asks *"how far from its own slice centroid does a typical
sample of this class sit?"*. Pooling that question **globally** is wrong for a
global collection: wheat in a uniform monoculture disperses far less around its
local centroid than wheat in a fragmented smallholder landscape. A globally
pooled null is set by whichever landscape contributes the most slices, so every
region that is legitimately more variable looks anomalous as a whole — and the
false positives land in exactly the places that are hardest to check on a
basemap.

The null is therefore conditioned, on two things, in this order:

1. **`h3_null_region`** — *where* the slice is. The slice cell's H3 parent at
   `region_level = max(slice_res - null_region_offset, null_region_min_level)`
   (offset 2, floor L1), so every slice is calibrated against roughly the same
   *number* of sibling cells (~49) rather than the same absolute area.
2. **`h3_null_res`** — *at what scale* it was sliced. The slice cell's own H3
   resolution.

The second one matters more than it looks. A distance distribution scales with
cell size — an L2 cell (86,802 km²) spans far more legitimate variation than an
L4 cell (1,770 km²) — so slices at different resolutions must not share a null.
The region key alone does not separate them: floored at L1, one region holds
both L2 and L3 slices. This is what made a three-level **CROPTYPE24** run
(`h3_level=[2, 3, 4]`) behave worse than a two-level **LANDCOVER10** run
(`[2, 3]`, almost all L3 in dense regions) — the tight L4 slices inherited a
scale inflated by the coarse L2 ones and their real errors fell under the gate.

Measured end to end on a co-located L2/L3/L4 croptype run with 15 % planted
errors:

| null keys | recall | clean FP |
|---|---|---|
| class only | 30.9 % | 0.368 % |
| class + fixed L1 region | 30.9 % | 0.368 % |
| class + relative region | 37.4 % | 0.150 % |
| **class + region + resolution** | **38.5 %** | **0.109 %** |

The two keys fix different levels, which is worth knowing when reading a run:

| | L2 rec/FP | L3 rec/FP | L4 rec/FP |
|---|---|---|---|
| class + fixed L1 region | 23 % / 0.90 % | 38 % / 0.20 % | 32 % / 0.00 % |
| class + relative region | 16 % / 0.37 % | 25 % / 0.08 % | 71 % / 0.00 % |
| class + region + resolution | 13 % / 0.20 % | 31 % / 0.12 % | 71 % / 0.00 % |

The **relative region** rescues L4 (32 % → 71 % recall): an L4 slice's region is
an L2 cell that holds only other L4 slices, so the region key separates
resolutions there by itself. The **resolution key** does the rest — L2 and L3
slices share one L1 region, and separating them lifts L3 recall 25 % → 31 %
while halving L2's false positives 0.37 % → 0.20 %.

L2 recall falls throughout (23 % → 13 %) as part of the same trade: the old
scheme was manufacturing detections there from an under-estimated scale, and
paying 0.90 % false positives for them.

For **LANDCOVER10** (`h3_level=[2, 3]`) the effect is smaller but points the
same way — same run, 15 % planted:

| null keys | recall | clean FP | L3 rec/FP |
|---|---|---|---|
| class only / fixed region / relative region | 24.8 % | 0.252 % | 28 % / 0.05 % |
| **class + region + resolution** | **25.0 %** | **0.112 %** | **33 % / 0.08 %** |

Both L2 and L3 floor to the same L1 region, so the region key alone changes
nothing there and only the resolution key bites: recall flat, false positives
halved. No landcover-specific setting is needed — the same defaults serve both
domains.

#### The ladder

`null_extra_keys` is read as a **nesting**, coarsest first. A null is estimated
at every prefix — `(class)`, `(class, region)`, `(class, region, resolution)` —
and each row is calibrated against the *finest group that exists for it*. Each
depth is shrunk toward its own parent, not toward the flat global:

```
w      = n_slices / (n_slices + null_shrink_k)      # 0 below min_slices
null_d = w * own_d + (1 - w) * null_(d-1)
```

with `null_shrink_k = 5`, so a group with 30 slices sits at `w = 0.86` and one
with 3 at `w = 0.38` — no threshold to trip over, and thin groups degrade
smoothly.

The backing-off is what makes the extra conditioner safe. Every conditioner
thins the groups, and for CROPTYPE24 most classes are rare somewhere. Without a
ladder a thin `(class, region, resolution)` group fell all the way back to the
flat global null — reinstating exactly the mixing the key was added to remove,
in precisely the rare-class-in-a-fine-cell case that needed it. Ordering region
*before* resolution means a thin group keeps its locality: measured over four
regions × three resolutions, region-then-resolution gave 0.46 % clean false
positives with a flat regional profile (0.36 % / 0.51 % between the tightest and
widest region), resolution-then-region 0.51 % with the regional gradient back
(0.07 % / 0.91 %).

The run prints the depth histogram, e.g.

```
[anomaly] Null ladder ['CROPTYPE24', 'h3_null_region', 'h3_null_res']:
          24 groups across 3 depths; rows calibrated at ->
          (CROPTYPE24, h3_null_region, h3_null_res) 100%
```

A large share of rows at depth 0 or 1 means the conditioners are too fine for
this collection's density — raise `null_region_offset` or drop a key.

Measured on clean data whose regions differ only in their legitimate spread:

| slices per region | pooled null | regional null |
|---|---|---|
| 3 | 1.99 % | 0.83 % |
| 5 | 2.35 % | 0.79 % |
| 12 | 2.03 % | 0.68 % |
| 25 | 2.06 % | 0.68 % |

…and the spread of the false-positive rate *across* regions falls from 3.85 %
to 0.78 %.

**This is a trade, not a free win.** At 10 % contamination precision rises
0.90 → 0.97 while recall falls 0.89 → 0.79 (five-seed average): a good share of
what the pooled null was "finding" was the regional bias rather than real
errors. The direction of the recall effect is scenario-dependent — it reverses
where tight regions dominate. Where regions genuinely do not differ,
conditioning costs ~2 % of recall and still lowers the false-positive rate.

Pass `null_extra_keys=[]` to pool globally, or
`null_extra_keys=["h3_null_region"]` to drop only the resolution key (sensible
for a single-resolution run, where it is a constant anyway). Set
`null_region_level` to pin an absolute region resolution instead of the
relative one — legacy behaviour, and wrong for any multi-resolution run. If you
have a real agro-ecological-zone or year column, add it to `null_extra_keys`
before the hexagon key — it is a better region proxy.

### Degenerate populations

If a class's pooled scale collapses — near-duplicate embeddings, e.g. grid-sampled
polygon interiors — the null scale is rejected rather than floored, and nothing in
that class is flagged. Flooring it would turn the absolute gate into a hair
trigger: a normal distance divided by a floored pseudo-scale yields z-scores in
the thousands, and measured on a duplicate-heavy synthetic that flagged 46 % of
the *ordinary* slices with zero planted errors.

### Heavy slice contamination

Two things decide whether a badly contaminated slice is still detectable:

* **The centroid** must resist the contaminant — `centroid_trim` should be at
  least the worst contamination you expect. At 40 % contamination a trim of
  0.20 leaves the centroid 0.954 cosine-similar to the true majority
  direction; 0.45 gets it to 0.999.
* **The scale** must not be inflated by the contaminant. This is the one that
  actually bit. The cross-slice null took the median of per-slice **MADs**, and
  a MAD is widened by the very right-side errors it is meant to measure
  against — so when *every* slice of a class carries 30–45 % errors, the null
  inflates with them and nothing clears the gate.
  `null_scale_estimator="left_tail"` measures the spread as `median − q25`,
  i.e. from the clean left half only. Under symmetry that is exactly the MAD,
  but it is blind to any right-side contamination below 50 %.

Measured end-to-end on synthetic slices, at a clean-data false-positive rate of
1.11 % (slightly *below* the previous defaults' 1.17 %):

| slice contamination | recall, previous defaults | recall, current defaults |
|---|---|---|
| 20 % | 0.992 | 0.994 |
| 30 % | 0.798 | 0.985 |
| 40 % | 0.140 | 0.902 |
| 45 % | 0.084 | 0.256 |

Precision stays above 0.98 throughout. Beyond ~45 % the problem becomes
genuinely unidentifiable (see below), not a tuning question.

### Known limit

Calibration removes the artificial blind spot; it does not remove the
**identifiability** limit. Once mislabelled points approach half of their own
class within a region, no purely geometric method can say which half is wrong.
That regime is what the per-`ref_id` rollup (written next to the summary as
`*_by_ref_id.parquet`) and the `time_col` split are for: the evidence lives at
the level of the dataset, not the point.

---

## Reproducing the evaluation

The `experiments/` directory holds the evaluation used in the paper:

- **Synthetic-noise recovery** (`run_validation.py`): inject controlled label noise and measure how
  strongly the detector concentrates the planted errors (AUROC, average precision, precision@k).
- **CatBoost proxy** (`run_catboost_proxy.py`): a fast, model-independent check of whether treating
  outliers in the *train* set improves classification of a fixed clean *test* set.
- **Operational fine-tuning** (`run_worldcereal_finetune.py`): the real downstream WorldCereal model.

---

## Running the tests

```bash
pytest tests/ -v      # synthetic data, no DuckDB, no network — runs in seconds
```

---

## Citation

If you use this code, please cite:

```bibtex
@article{alishah2026eba,
  title   = {Embeddings-based Anomaly Detection for Cleaning Global Crop-Type Reference Datasets},
  author  = {Ali Shah, Syed Roshaan and Van Tricht, Kristof and Butsko, Christina and
             Degerickx, Jeroen and Szantoi, Zoltan},
  year    = {2026},
  journal = {arXiv preprint arXiv:2607.23908},
  url     = {https://arxiv.org/abs/2607.23908}
}
```

Accepted for spotlight oral presentation at **[TerraBytes II](https://terrabytes-workshop.github.io/spotlight)** workshop at ECCV 2026, Malmö, Sweden.

---

## License

Released under the [MIT License](LICENSE).
