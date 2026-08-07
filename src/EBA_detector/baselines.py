"""Baseline outlier scorers for the EBA detector validation harness.

Reference points for the ablation questions raised in review:

* ``global``  : the EBA scoring with *no* H3 locality (one global slice per
  label), to test how much the local slicing contributes.
* ``iforest`` / ``lof`` : off-the-shelf detectors (IsolationForest / Local
  Outlier Factor) on the *same* embeddings, to test whether any distance-based
  method does as well as the locality-aware one.
* ``spatial`` : an embedding-free spatial label-disagreement baseline that flags
  points whose label disagrees with the majority label of their H3 cell.

Each scorer returns a DataFrame with ``sample_id``, ``mean_score`` (higher =
more anomalous), ``flagged`` (bool) and ``scored`` (bool), so it plugs directly
into :func:`EBA_detector.validation.evaluate_detection`.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

MIN_CELL = 5  # minimum neighbourhood size for a baseline to score a point


def _embed_matrix(df: pd.DataFrame, embedding_col: str = "embedding") -> np.ndarray:
    return np.vstack(df[embedding_col].to_numpy()).astype("float32", copy=False)


def _finish(df: pd.DataFrame, score, flagged, scored) -> pd.DataFrame:
    """Attach a min-max normalised ``mean_score`` plus flag/scored columns."""
    out = df[[c for c in df.columns if "embedding" not in c]].copy()
    s = np.asarray(score, dtype=float)
    finite = s[np.isfinite(s)]
    lo, hi = (finite.min(), finite.max()) if finite.size else (0.0, 1.0)
    out["mean_score"] = (s - lo) / (hi - lo + 1e-12)
    out["flagged"] = np.asarray(flagged, dtype=bool)
    out["scored"] = np.asarray(scored, dtype=bool)
    return out


def score_global_no_slice(df, *, label_col, h3_col="h3_l3_cell", group_cols=(),
                          embedding_col="embedding", **score_kwargs):
    """EBA scoring with locality removed: one global slice per label."""
    from .validation import score_embeddings_df  # lazy import to avoid cycle
    d = df.copy()
    d["_global_cell"] = "0"                      # collapse every H3 cell into one
    score_kwargs.pop("group_cols", None)
    return score_embeddings_df(
        d, label_col=label_col, h3_col="_global_cell", group_cols=(),
        embedding_col=embedding_col, **score_kwargs,
    )


def score_isolation_forest(df, *, embedding_col="embedding",
                           contamination=0.1, seed=0, **_):
    from sklearn.ensemble import IsolationForest
    X = _embed_matrix(df, embedding_col)
    clf = IsolationForest(contamination=contamination, random_state=seed, n_jobs=-1)
    pred = clf.fit_predict(X)          # -1 outlier, 1 inlier
    score = -clf.score_samples(X)      # higher = more anomalous
    return _finish(df, score, pred == -1, np.ones(len(df), bool))


def score_lof(df, *, embedding_col="embedding",
              n_neighbors=20, contamination=0.1, **_):
    from sklearn.neighbors import LocalOutlierFactor
    X = _embed_matrix(df, embedding_col)
    k = min(int(n_neighbors), max(2, len(df) - 1))
    lof = LocalOutlierFactor(n_neighbors=k, contamination=contamination, n_jobs=-1)
    pred = lof.fit_predict(X)
    score = -lof.negative_outlier_factor_     # higher = more anomalous
    return _finish(df, score, pred == -1, np.ones(len(df), bool))


def score_spatial_disagreement(df, *, label_col, h3_col="h3_l3_cell",
                               flag_threshold=0.5, **_):
    """Embedding-free: score a point by how much its label disagrees with the
    majority label of its H3 cell (1 = no cell-mate shares the label)."""
    d = df.reset_index(drop=True)
    scores = np.zeros(len(d)); scored = np.zeros(len(d), dtype=bool)
    for _, g in d.groupby(h3_col):
        n = len(g)
        if n < MIN_CELL:
            continue
        counts = g[label_col].value_counts()
        same_other = g[label_col].map(counts).to_numpy() - 1  # exclude self
        disagree = 1.0 - same_other / (n - 1)
        pos = g.index.to_numpy()
        scores[pos] = disagree
        scored[pos] = True
    flagged = (scores >= flag_threshold) & scored
    return _finish(d, scores, flagged, scored)


_BASELINES = {
    "global": score_global_no_slice,
    "iforest": score_isolation_forest,
    "lof": score_lof,
    "spatial": score_spatial_disagreement,
}

BASELINE_NAMES = tuple(_BASELINES)


def score_with_baseline(df, baseline, *, label_col, h3_col="h3_l3_cell",
                        group_cols=(), embedding_col="embedding",
                        score_kwargs=None, seed=0):
    """Dispatch to a named baseline; returns a scored DataFrame."""
    if baseline not in _BASELINES:
        raise ValueError(f"unknown baseline {baseline!r}; choose from {BASELINE_NAMES}")
    fn = _BASELINES[baseline]
    kw = dict(score_kwargs or {})
    if baseline == "global":
        return fn(df, label_col=label_col, h3_col=h3_col, group_cols=group_cols,
                  embedding_col=embedding_col, **kw)
    return fn(df, label_col=label_col, h3_col=h3_col,
              embedding_col=embedding_col, seed=seed)
