#!/usr/bin/env python3
"""Build a unified per-sample table from the *separate* artefacts the WorldCereal
outlier pipeline produces, so the Pillar-A (synthetic noise) and Pillar-B
(CatBoost proxy) experiments can run without the 18-timestep training parquet.

The pieces and why they suffice here:

* **Embeddings DuckDB** (``embeddings_cache_*.duckdb``): one row per sample with
  ``sample_id, ref_id, ewoc_code, h3_l3_cell, lat, lon, embedding_0..127``.
  These frozen (vanilla) Presto vectors are exactly the features the detector
  itself uses; using them as classifier features is therefore a faithful,
  inexpensive proxy for the downstream effect (the *relative* baseline-vs-
  treatment difference is the signal, not the absolute accuracy).
* **Merged outlier parquet** (``merged_LC10_CTY24_flagged_gdf_*.parquet``): one
  row per sample with the anomaly columns and, importantly, the mapped label
  used for scoring (``outlier_LC10_cls`` / ``outlier_CTY24_cls``)---which we
  reuse as the classification target.
* (optional) a **region table** (``sample_id, region``) e.g. exported from the
  fine-tuning splits, to stratify by macro-region.

We never touch the long/wide 18-timestep parquet: the embeddings already encode
the time series, so no Presto forward pass is needed.

Returns a tidy DataFrame with ``embedding_0..N`` columns plus
``label / flag / conf / region / ref_id / sample_id / h3_l3_cell`` and a small
dict naming those columns, ready for ``run_catboost_proxy`` and ``run_validation``.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict
import pandas as pd

# domain prefix -> (label col, confidence col, flag col) in the merged parquet
_DOMAIN = {
    "CTY24": ("outlier_CTY24_cls", "CTY24_confidence_nonoutlier", "CTY24_anomaly_flag"),
    "LC10":  ("outlier_LC10_cls",  "LC10_confidence_nonoutlier",  "LC10_anomaly_flag"),
}


def load_embeddings_duckdb(
    duckdb_path: str,
    *,
    model_hash: Optional[str] = None,
    ref_ids: Optional[Sequence[str]] = None,
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, list]:
    """Read the embeddings cache into a wide DataFrame (embedding_0..N columns)."""
    import duckdb
    con = duckdb.connect(duckdb_path, read_only=True)
    try:
        cols = con.execute("PRAGMA table_info('embeddings_cache')").fetchdf()["name"].tolist()
        embed_cols = [c for c in cols if c.startswith("embedding_")]
        base = [c for c in ["sample_id", "ref_id", "ewoc_code", "h3_l3_cell", "lat", "lon"] if c in cols]
        where = []
        if model_hash:
            where.append(f"model_hash='{model_hash}'")
        if ref_ids:
            joined = ",".join("'" + str(r).replace("'", "''") + "'" for r in ref_ids)
            where.append(f"ref_id IN ({joined})")
        q = f"SELECT {', '.join(base + embed_cols)} FROM embeddings_cache"
        if where:
            q += " WHERE " + " AND ".join(where)
        if max_rows:
            q += f" USING SAMPLE {int(max_rows)} ROWS"
        df = con.execute(q).fetchdf()
    finally:
        con.close()
    df["sample_id"] = df["sample_id"].astype(str)
    return df, embed_cols


def load_unified(
    duckdb_path: str,
    merged_parquet: str,
    *,
    domain: str = "CTY24",
    model_hash: Optional[str] = None,
    ref_ids: Optional[Sequence[str]] = None,
    region_parquet: Optional[str] = None,
    quality_parquet: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Join DuckDB embeddings with the merged outlier parquet on ``sample_id``.

    Returns ``(df, names)`` where *names* gives the resolved
    ``label_col / flag_col / conf_col / region_col / group_col / h3_col`` for
    the chosen *domain* (``"CTY24"`` or ``"LC10"``).
    """
    if domain not in _DOMAIN:
        raise ValueError(f"domain must be one of {list(_DOMAIN)}")
    label_col, conf_col, flag_col = _DOMAIN[domain]

    emb, embed_cols = load_embeddings_duckdb(
        duckdb_path, model_hash=model_hash, ref_ids=ref_ids, max_rows=max_rows
    )

    # Only request columns that actually exist; flag/conf are optional, and
    # ref_id / h3 may live in the embeddings cache rather than the merged file.
    import pyarrow.parquet as _pq
    avail = set(_pq.ParquetFile(str(merged_parquet)).schema.names)
    want = [c for c in ("sample_id", label_col, conf_col, flag_col) if c in avail]
    if "sample_id" not in want:
        raise KeyError(f"merged parquet {merged_parquet} has no 'sample_id' column")
    if label_col not in want:
        raise KeyError(f"merged parquet {merged_parquet} lacks label column '{label_col}'")
    merged = pd.read_parquet(merged_parquet, columns=want)
    merged["sample_id"] = merged["sample_id"].astype(str)

    df = emb.merge(merged, on="sample_id", how="inner")

    # drop rows without a usable label
    df = df[df[label_col].notna()].copy()

    region_col = None
    if region_parquet:
        reg = pd.read_parquet(region_parquet)
        reg["sample_id"] = reg["sample_id"].astype(str)
        rcol = "region" if "region" in reg.columns else reg.columns[-1]
        df = df.merge(reg[["sample_id", rcol]].rename(columns={rcol: "region"}),
                      on="sample_id", how="left")
        region_col = "region"

    if quality_parquet:
        q = pd.read_parquet(quality_parquet)
        q["sample_id"] = q["sample_id"].astype(str)
        qcol = [c for c in q.columns if "quality" in c.lower()]
        if qcol:
            df = df.merge(q[["sample_id", qcol[0]]].rename(columns={qcol[0]: "quality"}),
                          on="sample_id", how="left")

    names = {
        "label_col": label_col, "flag_col": flag_col, "conf_col": conf_col,
        "h3_col": "h3_l3_cell", "group_col": "ref_id",
        "region_col": region_col, "embed_cols": embed_cols,
        "quality_col": "quality" if quality_parquet else None,
    }
    return df, names
