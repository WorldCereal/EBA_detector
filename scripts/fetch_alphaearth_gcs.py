#!/usr/bin/env python3
"""Fetch AlphaEarth Satellite Embedding vectors directly from Google's
official GCS bucket (``gs://alphaearth_foundations``), as documented at
https://developers.google.com/earth-engine/guides/aef_on_gcs_readme

Layout on GCS
-------------
``gs://alphaearth_foundations/satellite_embedding/v1/annual/{year}/{utm_zone}/{tile}.tiff``

An index is published alongside the data at
``gs://alphaearth_foundations/satellite_embedding/v1/annual/aef_index.parquet``
with one row per COG file: WGS84 footprint geometry, ``crs``, ``year``,
``utm_zone`` and UTM/WGS84 bounding boxes. As of writing the bucket is
configured "provider pays", so anonymous HTTPS reads work with no billing
project required.

Encoding (per Google's documentation — different from the linear scheme
used by the Source Cooperative re-release!):

    de_quantized = ((raw_int8 / 127.5) ** 2) * sign(raw_int8)

nodata = -128. Bands correspond to A00..A63 in that order.

Standalone / importable module:

    from fetch_alphaearth_gcs import fetch_region_to_parquet

fetch_region_to_parquet(
    region_parquet="/projects/worldcereal/data/hardnegatives/hardneg_eu_all_wide_v2.parquet",
    output_dir="/home/vito/shahs/TestFolder/wc_outliers/data_for_outlier/out_chunks",
    final_parquet="/home/vito/shahs/TestFolder/wc_outliers/data_for_outlier/hardneg_eu_all_wide_v2_AEF.parquet",
)

Or run as a script — see ``--help``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

from alphaearth_fetch_core import (
    build_tile_tasks,
    dequantize_quadratic,
    fetch_points,
    load_points_from_parquet,
    merge_chunks,
)

GCS_BUCKET = "alphaearth_foundations"
GCS_PREFIX = "satellite_embedding/v1/annual"
GCS_INDEX_URL = f"gs://{GCS_BUCKET}/{GCS_PREFIX}/aef_index.parquet"
GCS_HTTPS_BASE = f"https://storage.googleapis.com/{GCS_BUCKET}/{GCS_PREFIX}"

DEFAULT_INDEX_CACHE = Path(
    os.environ.get(
        "AEF_GCS_INDEX_CACHE",
        str(Path.home() / ".cache" / "alphaearth" / "aef_index.parquet"),
    )
)

# Candidate column names that might hold the per-file relative path / name
# across possible index schema variants — resolved dynamically at load time.
_URL_COL_CANDIDATES = ("href", "url", "gcs_url", "file", "filename", "path", "asset", "name", "key")


def _configure_gdal_env() -> None:
    """GDAL env for fast, anonymous, HTTPS-based COG reads via /vsicurl/.

    We deliberately use HTTPS (``/vsicurl/https://storage.googleapis.com/...``)
    rather than the GDAL ``/vsigs/`` driver so that no Google Cloud
    credentials / SDK are required — the bucket allows public anonymous
    reads and is configured "provider pays".
    """
    os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
    os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "200000000")  # 200 MB


def _download_index(cache_path: Path) -> Path:
    """Download ``aef_index.parquet`` from GCS (anonymous HTTPS) to a local
    cache file, if not already present.
    """
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{GCS_HTTPS_BASE}/aef_index.parquet"
    logger.info(f"Downloading AlphaEarth GCS index from {url} …")
    import requests

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    logger.info(f"Index cached at {cache_path}  ({len(resp.content) / 1e6:.1f} MB)")
    return cache_path


def _resolve_url_column(index: pd.DataFrame) -> str | None:
    for c in _URL_COL_CANDIDATES:
        if c in index.columns:
            return c
    return None


def _row_to_gcs_https_url(row: pd.Series, url_col: str | None) -> str:
    """Build a full ``/vsicurl/https://...`` URL for one index row.

    If the index exposes an explicit relative path / filename column, use
    it directly (joined onto the year/utm_zone directory). Otherwise fall
    back to the documented naming convention
    ``{year}/{utm_zone}/<basename-from-href>`` — if that also fails, the
    caller should inspect ``aef_index.parquet``'s columns directly.
    """
    year = int(row["year"])
    utm_zone = str(row["utm_zone"])
    if url_col is not None:
        raw = str(row[url_col])
        if raw.startswith("gs://"):
            raw = raw[len(f"gs://{GCS_BUCKET}/") :] if raw.startswith(f"gs://{GCS_BUCKET}/") else raw.split("/", 3)[-1]
            return f"/vsicurl/https://storage.googleapis.com/{GCS_BUCKET}/{raw}"
        if raw.startswith("http"):
            return f"/vsicurl/{raw}"
        # Bare filename / relative path under the year/zone directory
        basename = raw.rsplit("/", 1)[-1]
        return f"/vsicurl/{GCS_HTTPS_BASE}/{year}/{utm_zone}/{basename}"
    raise ValueError(
        "Could not find a URL/filename column in the AlphaEarth GCS index "
        f"(tried {_URL_COL_CANDIDATES}). Inspect the index's columns "
        "(`geopandas.read_parquet(...).columns`) and adapt `_row_to_gcs_https_url`."
    )


def load_index(cache_path: str | Path = DEFAULT_INDEX_CACHE) -> gpd.GeoDataFrame:
    """Download (if needed) and load the GCS ``aef_index.parquet``, then
    pre-process it into the generic schema expected by
    :func:`alphaearth_fetch_core.build_tile_tasks`: columns ``year_int``,
    ``cog_url``, ``epsg``, ``geometry``.
    """
    _configure_gdal_env()
    cache_path = Path(cache_path)
    _download_index(cache_path)

    index = gpd.read_parquet(cache_path)
    if index.geometry.name != "geometry":
        index = index.rename_geometry("geometry")
    if index.crs is None:
        index = index.set_crs("EPSG:4326")

    url_col = _resolve_url_column(index)
    if url_col is None:
        logger.warning(
            f"No obvious URL column found among {list(index.columns)} — "
            "will use documented year/utm_zone/<basename> convention with 'crs' for EPSG."
        )

    index["year_int"] = index["year"].astype(int)
    index["cog_url"] = index.apply(lambda r: _row_to_gcs_https_url(r, url_col), axis=1)
    index["epsg"] = index["crs"].apply(lambda s: int(str(s).split(":")[-1]))

    logger.info(f"GCS index loaded: {len(index):,} rows | years: {sorted(index['year_int'].unique())}")
    return index


def fetch_region_to_parquet(
    region_parquet: str | Path,
    output_dir: str | Path,
    final_parquet: str | Path,
    *,
    index_cache: str | Path = DEFAULT_INDEX_CACHE,
    heavy_threshold: int = 1_000,
    light_workers: int = 16,
    write_batch: int = 1_000,
) -> pd.DataFrame:
    """End-to-end: load points from ``region_parquet``, fetch AlphaEarth
    embeddings from the official GCS COGs (heavy/light split + resumable
    chunking), merge chunks, and write ``final_parquet``.
    """
    region_parquet = Path(region_parquet)
    output_dir = Path(output_dir)
    final_parquet = Path(final_parquet)

    index = load_index(index_cache)
    pts = load_points_from_parquet(region_parquet, output_dir)
    logger.info(f"Points to fetch: {len(pts):,}")

    if pts.empty:
        logger.info("Nothing to fetch — all sample_ids already present in output chunks.")
    else:
        tasks, unmatched = build_tile_tasks(pts, index, year_col="year_int", url_col="cog_url", epsg_col="epsg")
        logger.info(f"Tile tasks: {len(tasks):,}  |  unmatched points: {unmatched:,}")
        fetch_points(
            tasks,
            dequantize_quadratic,
            output_dir,
            heavy_threshold=heavy_threshold,
            light_workers=light_workers,
            write_batch=write_batch,
        )

    return merge_chunks(output_dir, final_parquet)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--region-parquet", required=True, help="Input parquet with sample_id, lat, lon, year")
    p.add_argument("--output-dir", required=True, help="Directory for resumable chunk parquet files")
    p.add_argument("--final-parquet", required=True, help="Path to write the merged final parquet")
    p.add_argument("--index-cache", default=str(DEFAULT_INDEX_CACHE), help="Local cache path for aef_index.parquet")
    p.add_argument("--heavy-threshold", type=int, default=1_000, help="Points/tile above which to download locally")
    p.add_argument("--light-workers", type=int, default=16, help="Thread pool size for remote (light) tiles")
    p.add_argument("--write-batch", type=int, default=1_000, help="Rows per output chunk file")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    df = fetch_region_to_parquet(
        region_parquet=args.region_parquet,
        output_dir=args.output_dir,
        final_parquet=args.final_parquet,
        index_cache=args.index_cache,
        heavy_threshold=args.heavy_threshold,
        light_workers=args.light_workers,
        write_batch=args.write_batch,
    )
    print(f"\n✓ Final parquet: {len(df):,} rows  ({df['valid'].sum():,} valid)  →  {args.final_parquet}")


if __name__ == "__main__":
    main()
