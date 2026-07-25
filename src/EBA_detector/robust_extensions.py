"""Robustness extensions for the embedding-based outlier pipeline.

This module adds two capabilities that close important logical holes in the
core pipeline and make embedding-space outlier removal defensible enough to
publish:

1. **Slice trust / separability gating** (:func:`compute_slice_trust`,
   :func:`apply_trust_to_confidence`, :func:`downgrade_flags_low_trust`).

   The detector scores samples by their distance to a within-slice reference.
   That is only meaningful if the *embedding geometry itself* is informative
   for the slice.  With a **vanilla (un-finetuned) Presto encoder** there is no
   guarantee the 128-d space separates crop classes in every region — so a
   "distance from the local mean" can reflect encoder blind spots rather than
   label noise.  This is the central threat to the claim that the flagged
   points are genuine errors.  We quantify, per slice / context, how
   well-structured the embedding cloud is (a silhouette-style separation when
   ≥2 labels co-occur, and a concentration ratio otherwise) and use that
   ``slice_trust`` ∈ [0, 1] to (a) attenuate confidence penalties and
   (b) downgrade anomaly categories where the geometry cannot be trusted.
   Untrustworthy slices therefore contribute *fewer, softer* flags instead of
   silently injecting noise into the training-data cleaning.

2. **Group / dataset-level aggregation** (:func:`aggregate_parcel_scores`,
   and the optional :func:`parcel_aware_slice_scores`).

   WorldCereal reference points are **field centroids — one point per field**,
   identified only by ``sample_id`` (unique) and ``ref_id`` (the source
   dataset).  There is therefore *no* within-field oversampling, and no field
   id.  The relevant systematic-error grouping is instead the **source dataset**
   (``ref_id``): an entire dataset can be digitised against the wrong legend or
   carry a regional labelling bias, which point-wise scoring dilutes across the
   slice.  :func:`aggregate_parcel_scores` rolls the per-point anomaly score up
   to any group key (typically ``ref_id``) so that systematically-off datasets
   surface as a whole.  :func:`parcel_aware_slice_scores` (same-group kNN
   exclusion) is retained only for the case where several points *do* share a
   field/group; it is a no-op advantage when each group is a single point and
   should be left disabled for the standard one-centroid-per-field setup.

All functions are stateless and operate on the same per-slice DataFrame /
embedding conventions as :mod:`EBA_detector.anomaly_utils`.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from .anomaly_utils import robust_centroid

__all__ = [
    "compute_slice_trust",
    "apply_trust_to_confidence",
    "downgrade_flags_low_trust",
    "parcel_aware_slice_scores",
    "aggregate_parcel_scores",
]


# ---------------------------------------------------------------------------
# 1. Slice trust / separability
# ---------------------------------------------------------------------------


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def compute_slice_trust(
    df: pd.DataFrame,
    *,
    label_col: str,
    context_cols: Sequence[str],
    embedding_col: str = "embedding",
    out_col: str = "slice_trust",
    trim_frac: float = 0.10,
    min_n: int = 20,
) -> pd.DataFrame:
    """Per-context *trust* in the embedding geometry, in ``[0, 1]``.

    For every context group (``context_cols`` — typically the H3 cell, i.e.
    everything in the slice key except the label) we estimate how informative
    the embedding space is.  Two regimes:

    **Multi-label context (≥ 2 labels present).**  Compute a silhouette-style
    separation::

        trust = mean_intra / (mean_intra + mean_nearest_inter)   ... inverted

    More precisely we use ``sep = inter / (inter + intra)`` where *intra* is the
    mean cosine distance of points to their own (trimmed) label centroid and
    *inter* is the mean cosine distance between each label centroid and its
    nearest other-label centroid.  ``sep → 1`` means classes form tight,
    well-separated clusters (geometry is trustworthy); ``sep → 0.5`` or below
    means classes are entangled (distance-based scores are unreliable).
    We rescale ``sep`` so that the chance level (0.5) maps to ~0.

    **Single-label context.**  There are no other classes to separate from, so
    trust is based on *concentration*: a cloud with a clear dense core and a
    thin tail (high ``median / (median + MAD)`` contrast is **not** what we
    want — we want a tight core so that the tail is genuinely unusual).  We use
    ``conc = 1 - clip(MAD / (median + eps), 0, 1)``: a tight cloud (small MAD
    relative to its median radius) yields high trust; a diffuse cloud yields
    low trust.

    Small contexts (``< min_n`` points) get ``trust = 0.5`` (neutral — we have
    no basis to either trust or distrust the geometry, so flags there are
    later treated cautiously by the gate).

    Returns a copy of *df* with the float32 column *out_col* added
    (broadcast to every row of the context).
    """
    df = df.copy()
    trust = np.full(len(df), 0.5, dtype=np.float64)
    pos = np.arange(len(df))
    df["_pos_trust"] = pos
    emb = df[embedding_col].to_numpy()

    for _, g in df.groupby(list(context_cols), dropna=False, sort=False):
        idx = g["_pos_trust"].to_numpy()
        n = len(idx)
        if n < min_n:
            continue

        X = _l2_normalize(np.vstack(emb[idx]).astype(np.float32))
        labels = g[label_col].to_numpy()
        uniq = pd.unique(labels)

        if len(uniq) >= 2:
            centroids = []
            intra_dists = []
            for lab in uniq:
                m = labels == lab
                if m.sum() < 2:
                    continue
                c = robust_centroid(X[m], mode="trimmed", trim_frac=trim_frac)
                c = c / (np.linalg.norm(c) + 1e-12)
                centroids.append(c)
                intra_dists.append(np.mean(1.0 - (X[m] @ c)))
            if len(centroids) < 2:
                continue
            C = np.vstack(centroids)
            cc = 1.0 - (C @ C.T)
            np.fill_diagonal(cc, np.inf)
            nearest_inter = cc.min(axis=1)
            inter = float(np.mean(nearest_inter))
            intra = float(np.mean(intra_dists))
            sep = inter / (inter + intra + 1e-12)          # 0.5 == chance-ish
            t = np.clip((sep - 0.5) / 0.5, 0.0, 1.0)        # rescale to [0,1]
        else:
            c = robust_centroid(X, mode="trimmed", trim_frac=trim_frac)
            c = c / (np.linalg.norm(c) + 1e-12)
            d = 1.0 - (X @ c)
            med = float(np.median(d))
            mad = float(np.median(np.abs(d - med)))
            conc = 1.0 - np.clip(mad / (med + 1e-9), 0.0, 1.0)
            t = float(conc)

        trust[idx] = t

    df[out_col] = trust.astype(np.float32)
    df = df.drop(columns=["_pos_trust"], errors="ignore")
    return df


def apply_trust_to_confidence(
    df: pd.DataFrame,
    *,
    conf_col: str = "confidence_nonoutlier",
    trust_col: str = "slice_trust",
    flagged_col: str = "flagged",
    out_col: Optional[str] = None,
    min_trust: float = 0.30,
    floor: float = 0.0,
) -> pd.DataFrame:
    """Pull a flagged sample's confidence back toward 1.0 when its slice
    geometry is untrustworthy.

    For flagged samples only::

        conf_out = 1 - (1 - conf) * trust_factor

    where ``trust_factor`` ramps from 0 at ``trust == 0`` to 1 at
    ``trust >= min_trust`` (clipped), so that a flag raised in a low-trust
    slice barely reduces the weight, while a flag in a high-trust slice keeps
    its full penalty.  Unflagged samples are untouched.

    If *out_col* is None the confidence column is updated in place.
    """
    out_col = out_col or conf_col
    df = df.copy()
    if conf_col not in df.columns or trust_col not in df.columns:
        df[out_col] = df.get(conf_col, 1.0)
        return df

    conf = pd.to_numeric(df[conf_col], errors="coerce").fillna(1.0).to_numpy()
    trust = pd.to_numeric(df[trust_col], errors="coerce").fillna(0.5).to_numpy()
    factor = np.clip(trust / max(min_trust, 1e-6), 0.0, 1.0)

    if flagged_col in df.columns:
        flagged = df[flagged_col].fillna(False).to_numpy(dtype=bool)
    else:
        flagged = np.ones(len(df), dtype=bool)

    new_conf = conf.copy()
    new_conf[flagged] = 1.0 - (1.0 - conf[flagged]) * factor[flagged]
    new_conf = np.clip(new_conf, floor, 1.0)
    df[out_col] = new_conf.astype(np.float32)
    return df


def downgrade_flags_low_trust(
    df: pd.DataFrame,
    *,
    flag_col: str = "anomaly_flag",
    trust_col: str = "slice_trust",
    suspect_min_trust: float = 0.30,
    candidate_min_trust: float = 0.50,
) -> pd.DataFrame:
    """Soften anomaly categories in low-trust slices.

    A ``candidate`` raised where ``slice_trust < candidate_min_trust`` is
    downgraded to ``suspect``; a ``suspect`` (or downgraded candidate) where
    ``slice_trust < suspect_min_trust`` is downgraded to ``flagged``.  This
    keeps the strong, removal-worthy categories reserved for slices whose
    geometry we can actually trust, while still surfacing the sample for
    inspection.
    """
    df = df.copy()
    if flag_col not in df.columns or trust_col not in df.columns:
        return df
    trust = pd.to_numeric(df[trust_col], errors="coerce").fillna(0.5).to_numpy()
    flags = df[flag_col].astype(object).to_numpy()

    cand = (flags == "candidate") & (trust < candidate_min_trust)
    flags[cand] = "suspect"
    susp = (flags == "suspect") & (trust < suspect_min_trust)
    flags[susp] = "flagged"

    df[flag_col] = flags
    return df


# ---------------------------------------------------------------------------
# 2. Parcel / group awareness
# ---------------------------------------------------------------------------


def parcel_aware_slice_scores(
    df_slice: pd.DataFrame,
    *,
    group_col: str,
    embedding_col: str = "embedding",
    knn_k: int = 10,
    centroid_trim: float = 0.10,
    norm_percentiles=(2.0, 98.0),
) -> pd.DataFrame:
    """Score a single slice with **same-parcel neighbours excluded** from the
    kNN distance.

    Standard kNN scoring lets a systematically mislabelled parcel hide: each of
    its near-duplicate points finds its siblings as nearest neighbours and
    therefore looks like an inlier.  Here, for every point, the *k* nearest
    neighbours are taken only from points belonging to a **different**
    ``group_col`` value (e.g. a different field).  A whole wrong parcel then has
    large kNN distance to every *other* parcel and is exposed.

    Returns the slice DataFrame (embeddings dropped) with columns
    ``cos_dist_parcel``, ``knn_dist_parcel``, ``S_parcel`` (the mean of the two
    per-slice min-max-normalised distances, in ``[0, 1]``).
    """
    X = np.vstack(df_slice[embedding_col].to_numpy()).astype(np.float32)
    Xn = _l2_normalize(X)
    n = X.shape[0]
    groups = df_slice[group_col].astype(str).to_numpy()

    c = robust_centroid(X, mode="trimmed", trim_frac=centroid_trim)
    c = c / (np.linalg.norm(c) + 1e-12)
    cos_dist = 1.0 - (Xn @ c)

    # full cosine-distance matrix, mask self and same-group neighbours
    D = 1.0 - (Xn @ Xn.T)
    same = groups[:, None] == groups[None, :]
    D[same] = np.inf  # excludes self and all siblings in the same parcel

    knn_dist = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        row = D[i]
        finite = row[np.isfinite(row)]
        if finite.size == 0:
            knn_dist[i] = 0.0
            continue
        k_i = min(knn_k, finite.size)
        knn_dist[i] = np.partition(finite, k_i - 1)[:k_i].mean()

    def _norm(v):
        lo, hi = np.percentile(v, norm_percentiles)
        denom = hi - lo if hi > lo else 1.0
        return np.clip((v - lo) / denom, 0.0, 1.0)

    cos_n = _norm(cos_dist)
    knn_n = _norm(knn_dist)
    S = 0.5 * (cos_n + knn_n)

    out = df_slice.copy()[[c for c in df_slice.columns if "embedding" not in c]]
    out["cos_dist_parcel"] = cos_dist.astype(np.float32)
    out["knn_dist_parcel"] = knn_dist.astype(np.float32)
    out["S_parcel"] = S.astype(np.float32)
    return out


def aggregate_parcel_scores(
    df: pd.DataFrame,
    *,
    group_col: str,
    score_col: str = "S",
    label_col: Optional[str] = None,
    agg: str = "median",
) -> pd.DataFrame:
    """Aggregate point-level scores to a **parcel-level** anomaly score.

    A parcel whose *aggregate* score is high is suspicious as a whole — the
    granularity at which systematic label errors usually occur (a field digitised
    against the wrong crop).  Returns one row per ``group_col`` with the
    aggregated score, point count, and (optionally) the parcel's majority label.

    Parameters
    ----------
    agg
        Aggregation for the per-parcel score: ``"median"`` (robust, default),
        ``"mean"``, or ``"max"``.
    """
    if agg not in {"median", "mean", "max"}:
        raise ValueError("agg must be one of {'median','mean','max'}")
    g = df.groupby(group_col)[score_col]
    parcel = getattr(g, agg)().rename(f"parcel_{agg}_{score_col}").reset_index()
    parcel["parcel_n"] = df.groupby(group_col).size().to_numpy()
    if label_col is not None and label_col in df.columns:
        maj = (
            df.groupby(group_col)[label_col]
            .agg(lambda s: s.value_counts().index[0])
            .rename("parcel_majority_label")
            .reset_index(drop=True)
        )
        parcel["parcel_majority_label"] = maj
    return parcel
