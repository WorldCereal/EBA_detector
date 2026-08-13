"""Shared engine for fetching AlphaEarth Satellite Embedding vectors from COG tiles.

This module is source-agnostic: it doesn't care whether the COGs live on
Source Cooperative (S3) or Google Cloud Storage (GCS) — it just needs, for
every point to be sampled:

  * a "tile index" GeoDataFrame with one row per COG file, giving its
    footprint geometry (in EPSG:4326), the year it covers, its UTM EPSG
    code, and a GDAL-readable URL (``/vsis3/...``, ``/vsicurl/https://...``,
    etc.)
  * a de-quantization function that maps the raw int8 pixel values (as read
    by rasterio) to float32 embedding values.

Two public entry points are provided:

  * :func:`build_tile_tasks` — spatial-joins a point DataFrame against a
    tile index and groups points by the COG tile that covers them.
  * :func:`fetch_points` — the main parallel/heavy-light fetch loop. Given
    tile tasks, runs "light" tiles (few points) in a thread pool doing
    per-point HTTP range reads, and "heavy" tiles (many points, e.g. dense
    clusters) by downloading the whole COG locally once, sampling from
    disk, then deleting it — which is dramatically faster than thousands
    of tiny HTTP requests against the same file.

Both fetch scripts (``fetch_alphaearth_sourcecoop.py`` and
``fetch_alphaearth_gcs.py``) import this module and only provide the
source-specific index loading / URL construction / de-quantization glue.
"""

from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from pyproj import Transformer
from rasterio.shutil import copy as rio_copy
from shapely.geometry import Point
from tqdm.auto import tqdm

# ==============================================================
# AlphaEarth constants
# ==============================================================
AE_NODATA = -128
AE_N_DIMS = 64
AE_BAND_NAMES = [f"A{i:02d}" for i in range(AE_N_DIMS)]


def dequantize_linear(raw: np.ndarray) -> np.ndarray:
    """Simple linear de-quantization: ``raw / 127``  →  float32 in [-1, 1].

    This is the scheme used for the Source Cooperative re-release of the
    AlphaEarth COGs (verified empirically against known embedding ranges).
    """
    return raw.astype(np.float32) / 127.0


def dequantize_quadratic(raw: np.ndarray) -> np.ndarray:
    """Official AlphaEarth de-quantization (per Google's GCS documentation).

    ``((v / 127.5) ** 2) * sign(v)``  →  float32 in [-1, 1].
    """
    v = raw.astype(np.float32)
    return ((v / 127.5) ** 2) * np.sign(v)


DequantizeFn = Callable[[np.ndarray], np.ndarray]


# ==============================================================
# Tile task construction (spatial join points → COG tiles)
# ==============================================================
@dataclass
class TileTask:
    url: str
    points: pd.DataFrame  # columns: sample_id, lat, lon, year (+ any extra kept cols)
    epsg: int

    def __len__(self) -> int:
        return len(self.points)


def build_tile_tasks(
    points_df: pd.DataFrame,
    index_gdf: gpd.GeoDataFrame,
    year_col: str = "year_int",
    url_col: str = "cog_url",
    epsg_col: str = "epsg",
    point_cols: tuple[str, ...] = ("sample_id", "lat", "lon", "year"),
) -> tuple[list[TileTask], int]:
    """Spatial-join ``points_df`` (needs sample_id/lat/lon/year) against
    ``index_gdf`` and group matched points into one :class:`TileTask` per
    COG tile.  Falls back from ``within`` to ``intersects`` for points that
    land exactly on a tile boundary.

    Returns ``(tasks, n_unmatched)``.
    """
    gdf = gpd.GeoDataFrame(
        points_df,
        geometry=[Point(lon, lat) for lon, lat in zip(points_df["lon"], points_df["lat"])],
        crs=index_gdf.crs,
    )

    tasks: list[TileTask] = []
    unmatched = 0

    for year, grp in tqdm(gdf.groupby("year"), desc="Spatial join (years)", unit="yr"):
        idx_year = index_gdf[index_gdf[year_col] == year][[url_col, epsg_col, "geometry"]]
        if idx_year.empty:
            unmatched += len(grp)
            continue

        hits = gpd.sjoin(grp, idx_year, how="left", predicate="within")

        missed = hits[hits[url_col].isna()]
        if not missed.empty:
            m2 = missed.drop(columns=["index_right", url_col, epsg_col], errors="ignore")
            hits2 = gpd.sjoin(m2, idx_year, how="left", predicate="intersects")
            hits = pd.concat([hits[hits[url_col].notna()], hits2], ignore_index=True)

        unmatched += int(hits[url_col].isna().sum())

        keep_cols = [c for c in point_cols if c in hits.columns]
        for url, tile_grp in hits[hits[url_col].notna()].groupby(url_col):
            epsg = int(tile_grp[epsg_col].iloc[0])
            tasks.append(TileTask(url=url, points=tile_grp[keep_cols].copy(), epsg=epsg))

    return tasks, unmatched


# ==============================================================
# Per-tile sampling workers
# ==============================================================
def _record_from_raw(row: pd.Series, raw_vec: np.ndarray, dequantize: DequantizeFn) -> dict:
    is_nodata = bool(np.all(raw_vec == AE_NODATA))
    emb = [None] * AE_N_DIMS if is_nodata else dequantize(raw_vec).tolist()
    rec = {
        "sample_id": row["sample_id"],
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "year": int(row["year"]),
        "valid": not is_nodata,
    }
    rec.update(zip(AE_BAND_NAMES, emb))
    return rec


def _error_records(points: pd.DataFrame, error: Exception) -> list[dict]:
    out = []
    for _, row in points.iterrows():
        rec = {
            "sample_id": row["sample_id"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "year": int(row["year"]),
            "valid": False,
            "_error": str(error),
        }
        rec.update({n: None for n in AE_BAND_NAMES})
        out.append(rec)
    return out


def fetch_tile_remote(task: TileTask, dequantize: DequantizeFn) -> list[dict]:
    """Sample points directly from the remote COG via per-point HTTP range reads."""
    tx = Transformer.from_crs("EPSG:4326", task.epsg, always_xy=True)
    try:
        xy = [tx.transform(r.lon, r.lat) for _, r in task.points.iterrows()]
        with rasterio.open(task.url) as src:
            raw = np.array(list(src.sample(xy)), dtype=np.int8)
        return [
            _record_from_raw(row, raw[i], dequantize)
            for i, (_, row) in enumerate(task.points.iterrows())
        ]
    except Exception as e:  # noqa: BLE001
        return _error_records(task.points, e)


def fetch_tile_local(task: TileTask, dequantize: DequantizeFn, tmp_dir: Path) -> list[dict]:
    """Download the whole COG once into ``tmp_dir``, sample from local disk,
    then delete it.  Much faster than thousands of per-point remote reads
    when a tile has a very dense cluster of points.
    """
    tx = Transformer.from_crs("EPSG:4326", task.epsg, always_xy=True)
    local_path = tmp_dir / (task.url.rsplit("/", 1)[-1] or "tile.tif")
    out: list[dict] = []
    try:
        if not local_path.exists():
            rio_copy(task.url, str(local_path), driver="GTiff", compress="deflate")
        tqdm.write(f"  ↓ {local_path.name}  {local_path.stat().st_size / 1e6:.0f} MB  ({len(task)} pts)")
        xy = [tx.transform(r.lon, r.lat) for _, r in task.points.iterrows()]
        with rasterio.open(str(local_path)) as src:
            raw = np.array(list(src.sample(xy)), dtype=np.int8)
        out = [
            _record_from_raw(row, raw[i], dequantize)
            for i, (_, row) in enumerate(task.points.iterrows())
        ]
    except Exception as e:  # noqa: BLE001
        tqdm.write(f"  ✗ {local_path.name}: {e}")
        out = _error_records(task.points, e)
    finally:
        if local_path.exists():
            local_path.unlink()
    return out


# ==============================================================
# Main parallel / heavy-light orchestrator
# ==============================================================
@dataclass
class FetchStats:
    n_valid: int = 0
    n_nodata: int = 0
    n_errors: int = 0
    buffer: list = field(default_factory=list)


def fetch_points(
    tasks: list[TileTask],
    dequantize: DequantizeFn,
    output_dir: Path,
    *,
    heavy_threshold: int = 1_000,
    light_workers: int = 8,
    write_batch: int = 1_000,
    tmp_dir: Optional[Path] = None,
    chunk_prefix: str = "chunk",
) -> dict:
    """Fetch embeddings for all points in ``tasks``, writing resumable
    parquet chunks of ``write_batch`` rows to ``output_dir``.

    Tiles with fewer than ``heavy_threshold`` points are sampled remotely
    in parallel (``light_workers`` threads, one HTTP range-read per point).
    Tiles with ``>= heavy_threshold`` points are downloaded fully to
    ``tmp_dir`` (serially, one at a time), sampled from local disk, and
    the local copy is deleted immediately after — this avoids thousands of
    small HTTP requests hammering the same COG and is typically far
    faster for dense clusters.

    Returns a summary dict with counts and the list of chunk file paths
    written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = tmp_dir or (output_dir / "_tmp_heavy_tiles")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    heavy_tasks = [t for t in tasks if len(t) >= heavy_threshold]
    light_tasks = [t for t in tasks if len(t) < heavy_threshold]

    logger.info(
        f"Heavy tiles (>= {heavy_threshold} pts): {len(heavy_tasks)}  "
        f"({sum(len(t) for t in heavy_tasks):,} pts) → local download"
    )
    logger.info(
        f"Light tiles (< {heavy_threshold} pts): {len(light_tasks)}  "
        f"({sum(len(t) for t in light_tasks):,} pts) → parallel remote sampling "
        f"({light_workers} workers)"
    )

    existing_chunks = sorted(output_dir.glob(f"{chunk_prefix}_*.parquet"))
    chunk_idx = len(existing_chunks)
    stats = FetchStats()
    written_paths: list[Path] = []

    def flush():
        if not stats.buffer:
            return
        df = pd.DataFrame(stats.buffer)
        df.drop(columns=["_error"], errors="ignore", inplace=True)
        nonlocal chunk_idx
        p = output_dir / f"{chunk_prefix}_{chunk_idx:06d}.parquet"
        df.to_parquet(str(p), index=False)
        written_paths.append(p)
        tqdm.write(f"  ✓ {p.name}  ({len(df)} rows)")
        chunk_idx += 1
        stats.buffer.clear()

    def push(rows: list[dict]):
        stats.buffer.extend(rows)
        for r in rows:
            if r.get("_error"):
                stats.n_errors += 1
            elif r["valid"]:
                stats.n_valid += 1
            else:
                stats.n_nodata += 1
        if len(stats.buffer) >= write_batch:
            flush()

    # ── Heavy tiles: serial, one download at a time ────────────────────
    if heavy_tasks:
        for task in tqdm(heavy_tasks, desc="Heavy tiles (local dl)", unit="tile"):
            push(fetch_tile_local(task, dequantize, tmp_dir))

    # ── Light tiles: parallel remote sampling ──────────────────────────
    if light_tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=light_workers) as exe:
            futures = {exe.submit(fetch_tile_remote, t, dequantize): t for t in light_tasks}
            for fut in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(light_tasks),
                desc="Light tiles (parallel remote)",
                unit="tile",
            ):
                push(fut.result())

    flush()  # final remainder

    summary = {
        "n_valid": stats.n_valid,
        "n_nodata": stats.n_nodata,
        "n_errors": stats.n_errors,
        "chunks_written": written_paths,
    }
    logger.info(
        f"Done — valid: {stats.n_valid:,}  nodata: {stats.n_nodata:,}  errors: {stats.n_errors:,}"
    )
    return summary


def merge_chunks(
    output_dir: Path,
    final_parquet: Path,
    chunk_prefix: str = "chunk",
    rename_to_embedding_cols: bool = False,
) -> pd.DataFrame:
    """Concatenate all ``{chunk_prefix}_*.parquet`` files in ``output_dir``,
    drop duplicate sample_ids, optionally rename ``A00..A63`` →
    ``embedding_0..embedding_63``, and write the result to
    ``final_parquet``.
    """
    chunks = sorted(output_dir.glob(f"{chunk_prefix}_*.parquet"))
    if not chunks:
        raise FileNotFoundError(f"No chunk files found in {output_dir}")

    df = pd.concat([pd.read_parquet(str(f)) for f in chunks], ignore_index=True)
    df.drop_duplicates(subset="sample_id", inplace=True)

    if rename_to_embedding_cols:
        df.rename(columns={f"A{i:02d}": f"embedding_{i}" for i in range(AE_N_DIMS)}, inplace=True)

    final_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(final_parquet), index=False)
    logger.info(f"Merged {len(chunks)} chunks → {final_parquet}  ({len(df):,} rows)")
    return df


def load_points_from_parquet(
    parquet_path: Path,
    output_dir: Path,
    chunk_prefix: str = "chunk",
    columns: tuple[str, ...] = ("sample_id", "lat", "lon", "year"),
) -> pd.DataFrame:
    """Read unique (sample_id, lat, lon, year) points from a region parquet
    and drop any sample_ids already present in existing output chunks
    (resume support).
    """
    pts = pd.read_parquet(str(parquet_path), columns=list(columns))
    pts = pts.drop_duplicates(subset="sample_id").reset_index(drop=True)

    existing_chunks = sorted(output_dir.glob(f"{chunk_prefix}_*.parquet")) if output_dir.exists() else []
    if existing_chunks:
        done_ids = set(
            pd.concat(
                [pd.read_parquet(str(f), columns=["sample_id"]) for f in existing_chunks]
            )["sample_id"].dropna()
        )
        logger.info(f"Resuming — {len(done_ids):,} sample_ids already fetched in {len(existing_chunks)} chunks")
        pts = pts[~pts["sample_id"].isin(done_ids)].reset_index(drop=True)

    return pts
