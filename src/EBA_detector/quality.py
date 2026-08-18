"""Input-quality gates for the embedding-based anomaly detector.

Nothing in the original pipeline checked the embeddings themselves before
scoring them.  That mattered because of how ``_cosine_similarity`` degrades:

>>> _cosine_similarity(np.zeros(128), centroid)
0.0

A zero-norm vector therefore gets ``cosine_distance = 1.0`` — the **maximum
possible value** — and is guaranteed to sit at the top of its slice and be
flagged.  Failed encoder inference, an all-cloud time series, or a sample with
no valid S1/S2 coverage all produce exactly this, so a whole class of "the
basemap clearly shows a healthy field" false positives came from missing data
rather than from wrong labels.

A second, quieter hazard: ``restrict_model_hash`` defaults to ``None``, so if
the DuckDB cache ever holds embeddings from two encoder versions they are
mixed inside a single slice and every distance is meaningless.

This module surfaces both as explicit, countable conditions.  Samples that
fail are given the ``unscorable`` flag state rather than being allowed to
compete for ``candidate``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "EmbeddingQualityReport",
    "validate_embeddings",
    "assert_single_model_hash",
    "assert_h3_matches_coordinates",
]


class EmbeddingQualityReport:
    """Counts of each rejection reason, with a human-readable summary."""

    def __init__(self, counts: Dict[str, int], n_total: int) -> None:
        self.counts = counts
        self.n_total = n_total

    @property
    def n_rejected(self) -> int:
        return int(sum(self.counts.values()))

    def __repr__(self) -> str:  # pragma: no cover - display only
        if not self.n_rejected:
            return f"<EmbeddingQualityReport ok n={self.n_total:,}>"
        parts = ", ".join(f"{k}={v:,}" for k, v in sorted(self.counts.items()) if v)
        return (
            f"<EmbeddingQualityReport rejected={self.n_rejected:,}/"
            f"{self.n_total:,} ({parts})>"
        )

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"reason": k, "n": v} for k, v in sorted(self.counts.items())]
        )


def validate_embeddings(
    df: pd.DataFrame,
    *,
    embedding_col: str = "embedding",
    id_col: str = "sample_id",
    min_norm: float = 1e-6,
    max_norm_ratio: float = 1e4,
    out_col: str = "quality_reason",
) -> tuple:
    """Split *df* into scorable and unscorable rows.

    Rejection reasons
    -----------------
    ``non_finite``
        Any NaN / ±inf component — the distance to it is undefined and
        ``np.nan_to_num`` downstream would silently turn it into 0.
    ``zero_norm``
        Norm below *min_norm*.  Cosine distance to such a vector is 1.0 by
        the zero-guard, i.e. maximally anomalous, purely because the encoder
        produced nothing.
    ``extreme_norm``
        Norm more than *max_norm_ratio* times the median norm — a numeric
        blow-up, usually a broken input window.
    ``duplicate_id``
        The same *id_col* appearing more than once.  Downstream write-back
        joins on the id, so duplicates silently drop scores for all but one.

    Returns
    -------
    (df_ok, df_bad, report)
        *df_bad* carries the reason in *out_col*.  Both frames preserve the
        input columns; no rows are lost.
    """
    if embedding_col not in df.columns:
        raise KeyError(f"validate_embeddings: missing embedding column {embedding_col!r}")

    n = len(df)
    if n == 0:
        return df.copy(), df.copy(), EmbeddingQualityReport({}, 0)

    reason = np.array([""] * n, dtype=object)

    X = np.vstack(df[embedding_col].to_numpy()).astype(np.float64, copy=False)

    finite_rows = np.isfinite(X).all(axis=1)
    reason[~finite_rows] = "non_finite"

    norms = np.full(n, np.nan, dtype=np.float64)
    norms[finite_rows] = np.linalg.norm(X[finite_rows], axis=1)

    zero = finite_rows & (norms < float(min_norm))
    reason[zero & (reason == "")] = "zero_norm"

    good_norms = norms[finite_rows & (norms >= float(min_norm))]
    if good_norms.size:
        med_norm = float(np.median(good_norms))
        if med_norm > 0:
            extreme = finite_rows & (norms > med_norm * float(max_norm_ratio))
            reason[extreme & (reason == "")] = "extreme_norm"

    if id_col in df.columns:
        dup = df[id_col].duplicated(keep="first").to_numpy()
        reason[dup & (reason == "")] = "duplicate_id"

    bad_mask = reason != ""
    counts: Dict[str, int] = {}
    for r in np.unique(reason[bad_mask]) if bad_mask.any() else []:
        counts[str(r)] = int((reason == r).sum())

    df_ok = df.loc[~bad_mask].copy()
    df_bad = df.loc[bad_mask].copy()
    if not df_bad.empty:
        df_bad[out_col] = reason[bad_mask]

    return df_ok, df_bad, EmbeddingQualityReport(counts, n)


def assert_single_model_hash(
    df: pd.DataFrame,
    *,
    model_hash_col: str = "model_hash",
    restrict_model_hash: Optional[str] = None,
    strict: bool = True,
) -> List[str]:
    """Fail loudly when a run mixes embeddings from different encoders.

    Cosine distances between vectors produced by two different encoders are
    not comparable, so a slice containing both is scored against a reference
    that does not exist in either space.  With ``restrict_model_hash=None``
    (the historical default) this happened silently.

    Returns the list of distinct hashes found.  Raises when *strict* and more
    than one is present and no restriction was requested.
    """
    if model_hash_col not in df.columns:
        return []
    hashes = sorted(str(h) for h in pd.unique(df[model_hash_col].dropna()))
    if len(hashes) > 1 and restrict_model_hash is None:
        msg = (
            f"Embeddings cache contains {len(hashes)} distinct {model_hash_col} "
            f"values {hashes[:5]}{'...' if len(hashes) > 5 else ''}. Distances "
            "between different encoders are not comparable — pass "
            "restrict_model_hash=... to select one."
        )
        if strict:
            raise ValueError(msg)
        print(f"[quality] WARNING: {msg}")
    return hashes


def assert_h3_matches_coordinates(
    df: pd.DataFrame,
    *,
    h3_col: str = "h3_l3_cell",
    lat_col: str = "lat",
    lon_col: str = "lon",
    resolution: int = 3,
    sample_n: int = 5000,
    tolerance: float = 0.001,
    strict: bool = True,
    seed: int = 0,
) -> float:
    """Check that the cached H3 cells actually match the coordinates.

    The whole spatial-slicing premise rests on ``h3_l3_cell`` being correct.
    If the cache was ever built with a different convention (the presence of
    ``recompute_h3_l3_cells_in_cache`` suggests it has been), every slice is
    wrong and no amount of scoring fixes it.  This is a cheap one-off check on
    a random sample.

    Returns the observed mismatch fraction; raises when it exceeds *tolerance*
    and *strict*.
    """
    needed = {h3_col, lat_col, lon_col}
    if not needed.issubset(df.columns):
        return 0.0
    try:
        import h3 as _h3
    except ImportError:  # pragma: no cover - h3 is a hard dep in practice
        return 0.0

    sub = df[[h3_col, lat_col, lon_col]].dropna()
    if sub.empty:
        return 0.0
    if len(sub) > sample_n:
        sub = sub.sample(n=sample_n, random_state=seed)

    expected = [
        _h3.latlng_to_cell(float(la), float(lo), int(resolution))
        for la, lo in zip(sub[lat_col].to_numpy(), sub[lon_col].to_numpy())
    ]
    actual = [str(c) for c in sub[h3_col].to_numpy()]
    mismatch = float(np.mean([e != a for e, a in zip(expected, actual)]))

    if mismatch > float(tolerance):
        msg = (
            f"{mismatch:.2%} of sampled rows have {h3_col} inconsistent with "
            f"({lat_col}, {lon_col}) at resolution {resolution}. Spatial slices "
            "would be built on the wrong cells."
        )
        if strict:
            raise ValueError(msg)
        print(f"[quality] WARNING: {msg}")
    return mismatch
