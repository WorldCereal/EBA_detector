"""Quick smoke test for both AlphaEarth fetch scripts using a handful of
known-land points. Not a pytest module — run directly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd

TEST_POINTS = pd.DataFrame({
    "sample_id": ["p_nl", "p_in", "p_us", "p_br", "p_au"],
    "lat":       [52.10,  27.50,  40.71,  -15.60, -33.85],
    "lon":       [5.20,   80.20, -74.01,  -47.90,  151.20],
    "year":      [2023,   2022,   2023,   2021,    2024],
})

OUT_ROOT = Path("/tmp/alphaearth_smoke_test")


def run_sourcecoop():
    from fetch_alphaearth_sourcecoop import fetch_region_to_parquet

    region_pq = OUT_ROOT / "sourcecoop_points.parquet"
    region_pq.parent.mkdir(parents=True, exist_ok=True)
    TEST_POINTS.to_parquet(region_pq, index=False)

    df = fetch_region_to_parquet(
        region_parquet=region_pq,
        output_dir=OUT_ROOT / "sourcecoop_chunks",
        final_parquet=OUT_ROOT / "sourcecoop_final.parquet",
        heavy_threshold=1_000,
        light_workers=5,
    )
    print("\n=== Source Cooperative result ===")
    print(df[["sample_id", "lat", "lon", "year", "valid", "A00", "A01", "A02"]])
    return df


def run_gcs():
    from fetch_alphaearth_gcs import fetch_region_to_parquet

    region_pq = OUT_ROOT / "gcs_points.parquet"
    region_pq.parent.mkdir(parents=True, exist_ok=True)
    TEST_POINTS.to_parquet(region_pq, index=False)

    df = fetch_region_to_parquet(
        region_parquet=region_pq,
        output_dir=OUT_ROOT / "gcs_chunks",
        final_parquet=OUT_ROOT / "gcs_final.parquet",
        heavy_threshold=1_000,
        light_workers=5,
    )
    print("\n=== GCS result ===")
    print(df[["sample_id", "lat", "lon", "year", "valid", "A00", "A01", "A02"]])
    return df


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("sourcecoop", "both"):
        run_sourcecoop()
    if which in ("gcs", "both"):
        run_gcs()
