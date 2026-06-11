"""outlier_embeddings — Embedding-based outlier scoring for WorldCereal reference data.

This package provides a self-contained pipeline for detecting anomalous / mislabelled
samples in WorldCereal reference datasets, operating entirely on pre-computed Presto
embedding vectors stored in a DuckDB cache.

Modules
-------
anomaly_utils
    Pure stateless computation helpers: scoring, metrics, normalization, flagging,
    adaptive H3 assignment, and incremental update utilities.
anomaly
    Pipeline orchestration: loads embeddings from DuckDB, applies class mapping,
    scores per slice, flags anomalies, and writes results.
embeddings_cache
    DuckDB-backed cache for Presto embedding vectors. Handles incremental insertion
    of new embeddings without recomputing existing ones.
experiments
    CatBoost-based downstream experiments: train classifiers on top of outlier
    confidence scores to validate and benchmark the flagging.

Typical usage
-------------
>>> from outlier_embeddings.anomaly import run_pipeline
>>> flagged_gdf, summary_df = run_pipeline(
...     embeddings_db_path="/path/to/embeddings_cache.duckdb",
...     h3_level=[2, 3],
...     threshold_mode="mad",
...     mad_k=3.0,
...     output_samples_path="/path/to/output_scores.parquet",
... )
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("outlier_embeddings")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__ = ["anomaly", "anomaly_utils", "embeddings_cache", "experiments"]
