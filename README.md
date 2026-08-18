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
Accepted for spotlight oral presentation at **[TerraBytes II](https://terrabytes-workshop.github.io)** workshop at ECCV 2026, Malmö, Sweden. (see [Citation](#citation)).

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
    threshold_mode="mad",
    mad_k=4.0,
    centroid_mode="trimmed", centroid_trim=0.05,
    max_flagged_fraction=0.10,
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
| `LC10_anomaly_flag` | Category under LC10: `normal / flagged / suspect / candidate` |
| `outlier_LC10_cls` | Label class used for LC10 scoring |
| `CTY24_confidence_nonoutlier` | As above, under the CROPTYPE24 mapping |
| `CTY24_anomaly_flag` | Category under CTY24 |
| `outlier_CTY24_cls` | Label class used for CTY24 scoring |

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

Accepted for spotlight oral presentation at **[TerraBytes II](https://terrabytes-workshop.github.io)** workshop at ECCV 2026, Malmö, Sweden.

---

## License

Released under the [MIT License](LICENSE).
