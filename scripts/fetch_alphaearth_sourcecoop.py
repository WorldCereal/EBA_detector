#!/usr/bin/env python3
"""Fetch AlphaEarth Satellite Embedding vectors from the Source Cooperative
public COG re-release (https://source.coop/tge-labs/aef/v1/annual).

Standalone / importable module:

    from fetch_alphaearth_sourcecoop import fetch_region_to_parquet

    fetch_region_to_parquet(
        region_parquet="/path/to/points.parquet",   # needs sample_id, lat, lon, year
        output_dir="/path/to/out_chunks",
        final_parquet="/path/to/out.parquet",
    )

Or run as a script:

    python fetch_alphaearth_sourcecoop.py \\
        --region-parquet /path/to/points.parquet \\
        --output-dir /path/to/out_chunks \\
        --final-parquet /path/to/out.parquet \\
        --heavy-threshold 1000 --light-workers 16

Encoding
--------
int8, nodata = -128, linear scale 1/127 → float32 in [-1, 1].
Band names A00..A63 match the DuckDB / parquet column names used
downstream in the outlier-scoring pipeline.

Environment
-----------
Requires GDAL configured for anonymous public S3 access (set automatically
by this module at import time — see ``_configure_gdal_env()``).
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
    dequantize_linear,
    fetch_points,
    load_points_from_parquet,
    merge_chunks,
)

DEFAULT_INDEX_CACHE = Path(
    os.environ.get(
        "AEF_SOURCECOOP_INDEX_CACHE",
        str(Path.home() / ".cache" / "alphaearth" / "aef_sourcecoop_index.parquet"),
    )
)

SOURCECOOP_INDEX_URL = "https://data.source.coop/tge-labs/aef/v1/annual/aef_index.parquet"


def _configure_gdal_env() -> None:
    """Set the GDAL/AWS env vars required to read public Source Cooperative
    COGs anonymously and efficiently. Safe to call multiple times.
    """
    os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-west-2")
    os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
    os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.environ.setdefault("CPL_VSIL_CURL_CACHE_SIZE", "200000000")  # 200 MB


def _download_index(cache_path: Path) -> Path:
    """Download the Source Cooperative ``aef_index.parquet`` (anonymous
    HTTPS) to a local cache file, if not already present.
    """
    if cache_path.exists():
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading AlphaEarth Source Cooperative index from {SOURCECOOP_INDEX_URL} …")
    import requests

    resp = requests.get(SOURCECOOP_INDEX_URL, timeout=180)
    resp.raise_for_status()
    cache_path.write_bytes(resp.content)
    logger.info(f"Index cached at {cache_path}  ({len(resp.content) / 1e6:.1f} MB)")
    return cache_path


def load_index(index_path: Path | str = DEFAULT_INDEX_CACHE) -> gpd.GeoDataFrame:
    """Download (if needed) and load the Source Cooperative
    ``aef_index.parquet`` and pre-process it into the generic schema
    expected by :func:`alphaearth_fetch_core.build_tile_tasks`: columns
    ``year_int``, ``cog_url``, ``epsg``, ``geometry``.

    Index schema (one row per COG file): ``path`` (``s3://...``), ``year``,
    ``utm_zone``, ``crs`` (e.g. ``EPSG:32610``), plus UTM/WGS84 bounding
    boxes and a WGS84 footprint geometry.
    """
    _configure_gdal_env()
    index_path = Path(index_path)
    _download_index(index_path)

    index = gpd.read_parquet(index_path)
    if index.geometry.name != "geometry":
        index = index.rename_geometry("geometry")
    if index.crs is None:
        index = index.set_crs("EPSG:4326")

    index["year_int"] = index["year"].astype(int)
    # s3://us-west-2.opendata.source.coop/…  →  /vsis3/us-west-2.opendata.source.coop/…
    index["cog_url"] = index["path"].apply(lambda s: "/vsis3/" + str(s)[len("s3://") :])
    index["epsg"] = index["crs"].apply(lambda s: int(str(s).split(":")[-1]))

    logger.info(f"Index loaded: {len(index):,} rows | years: {sorted(index['year_int'].unique())}")
    return index


def fetch_region_to_parquet(
    region_parquet: str | Path,
    output_dir: str | Path,
    final_parquet: str | Path,
    *,
    index_path: str | Path = DEFAULT_INDEX_CACHE,
    heavy_threshold: int = 1_000,
    light_workers: int = 16,
    write_batch: int = 1_000,
) -> pd.DataFrame:
    """End-to-end: load points from ``region_parquet``, fetch AlphaEarth
    embeddings from Source Cooperative COGs (heavy/light split + resumable
    chunking), merge chunks, and write ``final_parquet``.

    Returns the merged DataFrame.
    """
    region_parquet = Path(region_parquet)
    output_dir = Path(output_dir)
    final_parquet = Path(final_parquet)

    index = load_index(index_path)
    pts = load_points_from_parquet(region_parquet, output_dir)
    logger.info(f"Points to fetch: {len(pts):,}")

    if pts.empty:
        logger.info("Nothing to fetch — all sample_ids already present in output chunks.")
    else:
        tasks, unmatched = build_tile_tasks(pts, index, year_col="year_int", url_col="cog_url", epsg_col="epsg")
        logger.info(f"Tile tasks: {len(tasks):,}  |  unmatched points: {unmatched:,}")
        fetch_points(
            tasks,
            dequantize_linear,
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
    p.add_argument("--index-path", default=str(DEFAULT_INDEX_CACHE), help="Local cache path for the Source Cooperative aef_index.parquet")
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
        index_path=args.index_path,
        heavy_threshold=args.heavy_threshold,
        light_workers=args.light_workers,
        write_batch=args.write_batch,
    )
    print(f"\n✓ Final parquet: {len(df):,} rows  ({df['valid'].sum():,} valid)  →  {args.final_parquet}")


if __name__ == "__main__":
    main()
