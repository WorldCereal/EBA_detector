#!/usr/bin/env python3
"""Populate the DuckDB Presto embeddings cache from wide-format parquet files.

Standalone CLI extracted from the interactive notebooks (e.g.
``notebooks/compute_outlier_scores_tessera.ipynb``) so the embedding pass can
be submitted as a SLURM GPU batch job instead of run cell-by-cell.

For each ``*{wide_suffix}.parquet`` file under ``--wide-dir``, streams the
file in Arrow batches and calls ``EBA_detector.embeddings_cache.compute_embeddings``
per batch, which skips any ``sample_id`` already cached for this model's hash
(so the script is safe to re-run / resume).

Example usage
-------------
python compute_embeddings.py \\
    --wide-dir /projects/worldcereal/data/cached_wide_merged/cached_wide_parquets \\
    --wide-suffix _ppq \\
    --embeddings-db-path /projects/worldcereal/data/cached_embeddings/embeddings_cache_LANDCOVER10.duckdb \\
    --presto-url /projects/worldcereal/models/.../model.pt \\
    --batch-size 4096 \\
    --num-workers 8

See ``compute_embeddings.sh`` for the sbatch launcher (edit the variables at
the top of that script, then ``sbatch compute_embeddings.sh``).
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Ensure worldcereal and EBA_detector are importable when running this script
# directly from a checkout (not an installed package).
# ---------------------------------------------------------------------------
def _ensure_packages_importable() -> None:
    here = Path(__file__).resolve()

    # EBA_detector/src/ — parent[1] of scripts/compute_embeddings.py
    oe_src = here.parents[1] / "src"
    if oe_src.exists() and str(oe_src) not in sys.path:
        sys.path.insert(0, str(oe_src))

    # worldcereal-classification/src/ — optional, only needed when running
    # from a local clone rather than an installed worldcereal package.
    wc_src = here.parents[1].parent / "worldcereal-classification" / "src"
    if wc_src.exists() and str(wc_src) not in sys.path:
        sys.path.insert(0, str(wc_src))


_ensure_packages_importable()

import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from prometheo.models import Presto  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from EBA_detector.embeddings_cache import compute_embeddings, get_model_hash  # noqa: E402

_DEFAULT_PRESTO_URL = (
    "https://artifactory.vgt.vito.be/artifactory/auxdata-public/worldcereal/models/"
    "PhaseII/presto-ss-wc_longparquet_random-window-cut_no-time-token_epoch96.pt"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Populate the DuckDB Presto embeddings cache from wide parquet files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--wide-dir", type=Path, required=True,
                    help="Directory containing wide-format parquet files.")
    p.add_argument("--wide-suffix", type=str, default="_ppq",
                    help="Filename suffix (before .parquet) identifying wide files to embed.")
    p.add_argument("--embeddings-db-path", type=Path, required=True,
                    help="Path to the DuckDB embeddings cache (created if missing).")
    p.add_argument("--presto-url", type=str, default=_DEFAULT_PRESTO_URL,
                    help="Presto checkpoint URL or local path.")
    p.add_argument("--batch-size", type=int, default=4096,
                    help="Model forward-pass batch size. Reduce to avoid GPU OOM.")
    p.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader worker processes.")
    p.add_argument("--parquet-batch-rows", type=int, default=300_000,
                    help="Rows per Arrow batch read from each wide parquet file.")
    p.add_argument("--force-recompute", action="store_true",
                    help="Delete and recompute embeddings already present in the cache.")
    p.add_argument("--log-level", type=str, default="WARNING",
                    help="Log level for embeddings-cache internals (e.g. DEBUG, INFO, WARNING).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Suppress debug/info noise from the embeddings cache internals by default.
    logger.remove()
    logger.add(sys.stderr, level=args.log_level)

    if not args.wide_dir.exists():
        raise FileNotFoundError(f"Wide directory not found: {args.wide_dir}")

    wide_files = sorted(args.wide_dir.glob(f"*{args.wide_suffix}.parquet"))
    if not wide_files:
        raise RuntimeError(f"No wide parquet files found under: {args.wide_dir}")
    print(f"Found {len(wide_files)} wide parquet files to embed.")

    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {torch_device}")

    model = Presto(pretrained_model_path=args.presto_url)
    model.eval().to(torch_device)
    model_hash = get_model_hash(model)
    print(f"Model hash: {model_hash}")

    for wide_file in tqdm(wide_files, desc="Embedding files", unit="file"):
        pf = pq.ParquetFile(str(wide_file))

        # Iterate ALL Arrow batches in the file.
        # (Using next() on the iterator would silently drop everything beyond
        # the first parquet_batch_rows rows.)
        for batch in pf.iter_batches(batch_size=args.parquet_batch_rows):
            tbl = pa.Table.from_batches([batch])
            df = tbl.to_pandas()
            compute_embeddings(
                df,
                model=model,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                embeddings_db_path=str(args.embeddings_db_path),
                force_recompute=args.force_recompute,
                show_progress=True,
            )
            del batch, tbl, df
            gc.collect()

    print("Embeddings cache population complete.")


if __name__ == "__main__":
    main()
