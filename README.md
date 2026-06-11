# outlier_embeddings <!-- omit in toc -->

A private companion package to [WorldCereal classification](https://github.com/WorldCereal/worldcereal-classification) that detects anomalous / potentially mislabelled samples in WorldCereal reference datasets by operating directly on pre-computed **Presto embedding vectors**.

---

## Overview

The WorldCereal training pipeline produces Presto embeddings for every reference sample.
This package provides a self-contained pipeline that:

1. **Loads** embeddings from a DuckDB cache (built once, updated incrementally).
2. **Groups** samples spatially by H3 cell + label class (adaptive multi-resolution supported).
3. **Scores** each sample within its slice using cosine distance to the class centroid and kNN distance to neighbours.
4. **Flags** statistical outliers per slice (MAD, percentile, FDR).
5. **Assigns** per-sample confidence scores and anomaly categories (`normal / flagged / suspect / candidate`).
6. **Writes** the 6 anomaly columns (`LC10_confidence_nonoutlier`, `LC10_anomaly_flag`, `outlier_LC10_cls`, `CTY24_*`) back to the long-format parquet files.

An **incremental update** mode (`--mode update`) re-scores only the geographic impact zone of newly added datasets — no need to reprocess the entire collection.

---

## Repository layout

```
outlier_embeddings/
├── pyproject.toml                        # Package definition + dependencies
├── README.md
├── .gitignore
│
├── src/
│   └── outlier_embeddings/
│       ├── __init__.py
│       ├── anomaly_utils.py              # Stateless computation helpers (scoring, metrics,
│       │                                 # normalization, flagging, adaptive H3, incremental utils)
│       ├── anomaly.py                    # Pipeline orchestration: load → map → score → flag → write
│       ├── embeddings_cache.py           # DuckDB-backed Presto embedding cache (build + query)
│       └── experiments.py               # CatBoost downstream experiments on outlier scores
│
├── scripts/
│   ├── compute_anomaly_scores.py         # Full CLI pipeline: parquets → embeddings → scores → write-back
│   └── get_mappings_from_legend.py       # SharePoint utilities for fetching legend/class-mapping Excel
│
├── notebooks/
│   ├── compute_outlier_scores.ipynb      # Interactive notebook driving the full pipeline
│   └── explore_sample_embeddings.ipynb  # Exploration / visualisation of embedding distributions
│
└── tests/
    ├── test_outliers.py                  # Unit tests for anomaly_utils (synthetic data, no DuckDB)
    └── testresources/
        └── test_class_mappings.json      # Small mapping fixture used by tests
```

---

## Dependencies

This package is designed to work **alongside** the main `worldcereal-classification` package
installed from its `main` branch. It imports three things directly from `worldcereal`:

| Import | Where used | Purpose |
|---|---|---|
| `worldcereal.utils.refdata.get_class_mappings` / `map_classes` | `anomaly.py` | Map `ewoc_code` → label class names |
| `worldcereal.train.datasets.WorldCerealTrainingDataset` | `embeddings_cache.py` | Build the DataLoader for Presto inference |
| `worldcereal.utils.timeseries.process_parquet` | `compute_anomaly_scores.py` | Pivot long→wide parquets before embedding |

Everything else (scoring, flagging, caching, incremental updates) lives entirely inside this package.

---

## Installation

### Step 1 — Install worldcereal from main

```bash
pip install "worldcereal[train] @ git+https://github.com/WorldCereal/worldcereal-classification.git@main"
```

Or from a local clone:

```bash
pip install -e /path/to/worldcereal-classification
```

### Step 2 — Install outlier_embeddings (this package)

```bash
# From the private repo:
pip install "outlier_embeddings @ git+https://github.com/WorldCereal/outlier_embeddings.git"

# Or from a local checkout (editable install — recommended for development):
pip install -e /path/to/outlier_embeddings
```

### Optional extras

```bash
pip install "outlier_embeddings[experiments]"  # CatBoost experiments
pip install "outlier_embeddings[sharepoint]"   # SharePoint legend fetching
pip install "outlier_embeddings[notebooks]"    # Jupyter environment
pip install "outlier_embeddings[all]"          # Everything
pip install "outlier_embeddings[dev]"          # Tests + linting
```

> **Note:** `prometheo` (the Presto backbone) must also be installed:
> `pip install "prometheo @ git+https://github.com/WorldCereal/prometheo.git@v0.0.5"`

---

## Quick Start

### Python API

```python
from outlier_embeddings.anomaly import run_pipeline

flagged_gdf, summary_df = run_pipeline(
    embeddings_db_path="/data/embeddings_cache.duckdb",
    h3_level=[2, 3],           # adaptive: try L2 first, fall back to L3
    group_cols=["ref_id"],
    min_slice_size=100,
    max_slice_size=5000,
    threshold_mode="mad",
    mad_k=3.0,
    max_flagged_fraction=0.10,
    output_samples_path="/data/outlier_scores.parquet",
    output_summary_path="/data/outlier_summary.parquet",
)
```

### Command-line (full pipeline)

```bash
# Full rerun: long parquets → wide → embeddings → LC10 + CTY24 scores → write back
python scripts/compute_anomaly_scores.py \
    --mode rerun \
    --input-format geoparquet \
    --input-long-dir  /data/MERGED_PARQUETS_PHASEII \
    --output-long-dir /data/MERGED_PARQUETS_PHASEII_WITH_ANOMALY \
    --embeddings-db-path /data/embeddings_cache.duckdb \
    --wide-dir /data/cached_wide_parquets \
    --merged-wide-path /data/worldcereal_wide.parquet \
    --lc10-h3-levels 2 3 \
    --lc10-min-slice-size 100 \
    --lc10-mad-k 3.0 \
    --cty24-h3-levels 2 3 \
    --cty24-min-slice-size 100 \
    --cty24-mad-k 3.0

# Incremental update: re-score only the impact zone of newly added datasets
python scripts/compute_anomaly_scores.py \
    --mode update \
    --input-format geoparquet \
    --input-long-dir  /data/MERGED_PARQUETS_PHASEII \
    --output-long-dir /data/MERGED_PARQUETS_PHASEII_WITH_ANOMALY \
    --embeddings-db-path /data/embeddings_cache.duckdb \
    --sp-env-file ~/.sharepointenv
```

The `--sp-env-file` flag points to a file that exports SharePoint credentials as environment
variables (see `scripts/get_mappings_from_legend.py` for the expected variable names).

---

## Modules in detail

### `anomaly_utils.py` — Pure computation helpers

Stateless building blocks; no DuckDB, no disk I/O:

- **Scoring**: `compute_scores_for_slice`, `score_slices_hierarchical`
- **Spatial grouping**: `assign_adaptive_h3_level`, `merge_small_slices`
- **Metrics**: `add_alt_class_centroid_metrics`, `add_knn_label_purity_for_flagged`
- **Flagging**: `flag_anomalies` (supports `percentile`, `mad`, `absolute`, `fdr`)
- **Confidence**: `add_confidence_from_score`, `apply_confidence_fusion`
- **Incremental helpers**: `find_unscored_samples`, `compute_impact_zone`,
  `load_affected_embeddings_from_cache`, `merge_scores_to_long_parquets`

### `anomaly.py` — Pipeline orchestrator

`run_pipeline(...)` wires together all the helpers above into a complete run:

```
DuckDB load → class mapping → adaptive H3 → merge small slices →
hierarchical ref-class assignment → context centroid metrics →
scoring → flagging → confidence → kNN purity → anomaly categorisation →
output write
```

Supports **incremental mode** (`skip_existing_samples=True`) to avoid rescoring
already-processed samples.

### `embeddings_cache.py` — DuckDB embedding cache

- `init_cache(db_path)` — create / open the embeddings table
- `compute_embeddings(data_df, model, ...)` — run Presto inference and insert results
- `insert_embeddings / fetch_embeddings` — low-level batch I/O
- `get_model_hash(model)` — SHA-256 fingerprint of model weights
  (segments cache per model version so swapping the backbone is safe)

### `experiments.py` — CatBoost downstream training

Train a CatBoost classifier on top of outlier confidence scores to evaluate whether
the flagging is meaningful (requires `catboost` extra).

### `scripts/compute_anomaly_scores.py` — Full CLI

End-to-end pipeline script with full argument parsing. Run `python scripts/compute_anomaly_scores.py --help` for all options.

### `scripts/get_mappings_from_legend.py` — SharePoint utilities

Fetches the WorldCereal legend Excel from SharePoint and builds the class-mapping JSON
used by the pipeline. Can also be used standalone to regenerate `class_mappings.json`.

---

## Output columns

The pipeline appends six columns to the long-format parquet files:

| Column | Description |
|---|---|
| `LC10_confidence_nonoutlier` | P(not outlier) under LANDCOVER10 mapping — float32 in [0, 1] |
| `LC10_anomaly_flag` | Anomaly category under LC10: `normal / flagged / suspect / candidate` |
| `outlier_LC10_cls` | Label class used for LC10 scoring |
| `CTY24_confidence_nonoutlier` | P(not outlier) under CROPTYPE24 mapping — float32 in [0, 1] |
| `CTY24_anomaly_flag` | Anomaly category under CTY24: `normal / flagged / suspect / candidate` |
| `outlier_CTY24_cls` | Label class used for CTY24 scoring |

---

## Running the tests

```bash
# From the repo root (no DuckDB, no network access, runs in seconds):
pytest tests/ -v
```

---

## Relationship to worldcereal-classification

```
worldcereal-classification  (main branch — install this first)
│
│   provides:
│     • worldcereal.train.datasets.WorldCerealTrainingDataset
│     • worldcereal.utils.refdata.get_class_mappings / map_classes
│     • worldcereal.utils.timeseries.process_parquet
│     • worldcereal.utils.sharepoint.*
│
└── outlier_embeddings  (this repo — install on top)
      provides:
        • Presto embeddings cache (DuckDB)
        • Embedding-space outlier scoring pipeline
        • Adaptive H3 spatial grouping
        • Incremental update mode
        • CLI script + interactive notebooks
```

The branch `outliers-from-embeddings` of `worldcereal-classification` contains the integration
history (earlier exploratory versions of these modules). Going forward, all outlier-specific
development happens in this repo; `worldcereal-classification` stays on `main`.

---

## License

MIT
