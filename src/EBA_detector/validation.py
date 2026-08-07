"""Synthetic label-noise validation for the embedding-based outlier detector.

The downstream crop-type experiments answer *"does removing flagged samples
help the model?"*, but on their own they cannot tell us *whether the detector
is finding genuine label errors* — a model can improve for many reasons.  To
make the outlier-removal claim defensible we need an **intrinsic, ground-truthed
evaluation**: corrupt a known set of labels, run the detector, and measure how
well the flags / scores recover the corrupted samples.

This module provides exactly that, with no dependency on ``worldcereal`` or a
DuckDB cache, so it runs on any pre-loaded embeddings DataFrame (real or
synthetic) and is fully unit-testable.

Pipeline
--------
1. :func:`inject_label_noise` — corrupt a controlled fraction of labels and
   record the ground truth (``noise_truth`` boolean column).  Three corruption
   models are supported, each mimicking a real WorldCereal error mode:

   * ``"within_context"`` — relabel a point to another class that co-occurs in
     the same H3 cell (confusable-class error, e.g. maize↔sunflower).
   * ``"random"`` — relabel to any other class present (gross error).
   * ``"group"`` (alias ``"parcel"``) — corrupt **all** points that share a
     source dataset (``ref_id``) or other group key at once, mimicking a whole
     dataset digitised against the wrong legend. (WorldCereal points are field
     centroids — one point per field — so this targets *dataset-level*
     systematic error, not within-field oversampling, which does not occur.)

2. :func:`score_embeddings_df` — a self-contained driver that runs the core
   scoring → flagging → confidence → (optional) slice-trust chain from
   :mod:`EBA_detector.anomaly_utils` and
   :mod:`EBA_detector.robust_extensions` on the corrupted DataFrame.

3. :func:`evaluate_detection` — score the recovery with detection AUROC,
   average precision (PR-AUC), precision@k / recall@k, and the realised
   precision/recall of the discrete ``flagged`` decision.

4. :func:`run_noise_experiment` / :func:`sweep` — convenience wrappers that
   tie the three steps together and return tidy result rows for plotting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .anomaly_utils import (
    MIN_SCORING_SLICE_SIZE,
    add_confidence_from_score,
    compute_scores_for_slice,
    compute_slice_centroids,
    flag_anomalies,
)
from .robust_extensions import (
    apply_trust_to_confidence,
    compute_slice_trust,
)

__all__ = [
    "inject_label_noise",
    "score_embeddings_df",
    "evaluate_detection",
    "run_noise_experiment",
    "sweep",
    "NoiseSpec",
]


# ---------------------------------------------------------------------------
# 1. Label-noise injection
# ---------------------------------------------------------------------------


@dataclass
class NoiseSpec:
    """Specification of a synthetic label-noise experiment."""

    mode: str = "within_context"     # within_context | random | group (alias: parcel)
    rate: float = 0.05               # fraction of points (or parcels) corrupted
    label_col: str = "label"
    h3_col: str = "h3_l3_cell"
    group_col: Optional[str] = None  # group key, e.g. ref_id/dataset (required for mode='group')
    seed: int = 0


def inject_label_noise(
    df: pd.DataFrame,
    spec: NoiseSpec,
    *,
    corrupt_label_col: str = "label_noisy",
    truth_col: str = "noise_truth",
) -> pd.DataFrame:
    """Return a copy of *df* with a corrupted label column and a ground-truth
    boolean (*truth_col*) marking which rows were corrupted.

    The original label column is left untouched; the corrupted labels are
    written to *corrupt_label_col* so the detector can be run on the noisy
    labels while the clean labels remain available for diagnostics.
    """
    rng = np.random.RandomState(spec.seed)
    out = df.copy().reset_index(drop=True)
    labels = out[spec.label_col].astype(object).to_numpy()
    noisy = labels.copy()
    truth = np.zeros(len(out), dtype=bool)

    if spec.mode in ("group", "parcel"):
        if not spec.group_col or spec.group_col not in out.columns:
            raise ValueError("mode='group'/'parcel' requires a valid group_col")
        groups = out[spec.group_col].astype(str).to_numpy()
        uniq_groups = pd.unique(groups)
        n_corrupt = max(1, int(round(spec.rate * len(uniq_groups))))
        chosen = rng.choice(uniq_groups, size=min(n_corrupt, len(uniq_groups)), replace=False)
        all_labels = pd.unique(labels)
        for grp in chosen:
            m = groups == grp
            cur = labels[m][0]
            alt = _pick_alternative(cur, all_labels, rng)
            if alt is None:
                continue
            noisy[m] = alt
            truth[m] = True

    elif spec.mode in {"within_context", "random"}:
        n = len(out)
        n_corrupt = max(1, int(round(spec.rate * n)))
        idx = rng.choice(n, size=min(n_corrupt, n), replace=False)
        h3 = out[spec.h3_col].astype(str).to_numpy() if spec.h3_col in out.columns else None
        all_labels = pd.unique(labels)
        for i in idx:
            cur = labels[i]
            if spec.mode == "within_context" and h3 is not None:
                # candidate alternatives = other labels in the same H3 cell
                same_cell = h3 == h3[i]
                cands = pd.unique(labels[same_cell])
                cands = [c for c in cands if c != cur]
                if not cands:
                    cands = [c for c in all_labels if c != cur]
            else:
                cands = [c for c in all_labels if c != cur]
            if not cands:
                continue
            noisy[i] = cands[rng.randint(len(cands))]
            truth[i] = True
    else:
        raise ValueError(f"Unknown noise mode: {spec.mode}")

    out[corrupt_label_col] = noisy
    out[truth_col] = truth
    return out


def _pick_alternative(cur, all_labels, rng) -> Optional[object]:
    cands = [c for c in all_labels if c != cur]
    if not cands:
        return None
    return cands[rng.randint(len(cands))]


# ---------------------------------------------------------------------------
# 2. Self-contained scoring driver (no worldcereal / DuckDB)
# ---------------------------------------------------------------------------


def score_embeddings_df(
    df: pd.DataFrame,
    *,
    label_col: str,
    h3_col: str = "h3_l3_cell",
    group_cols: Sequence[str] = (),
    embedding_col: str = "embedding",
    threshold_mode: str = "mad",
    mad_k: float = 4.0,
    percentile_q: float = 0.96,
    max_flagged_fraction: Optional[float] = None,
    norm_percentiles: Tuple[float, float] = (2.0, 98.0),
    centroid_mode: str = "trimmed",
    centroid_trim: float = 0.05,
    max_full_pairwise_n: Optional[int] = 0,
    gate_confidence_by_flag: bool = True,
    apply_slice_trust: bool = False,
    slice_trust_min: float = 0.05,
) -> pd.DataFrame:
    """Run the core scoring → flag → confidence → trust chain on a pre-loaded
    embeddings DataFrame.

    This mirrors the heart of :func:`EBA_detector.anomaly.run_pipeline`
    but skips the worldcereal-specific data loading and class-mapping layers,
    so it can be driven directly with controlled labels.  *df* must contain
    ``sample_id``, *label_col*, *h3_col*, and a vector *embedding_col*.

    Returns the scored DataFrame with at least: ``S``, ``mean_score``,
    ``flagged``, ``confidence``, and (if enabled) ``slice_trust``.
    """
    group_cols = list(group_cols)
    df = df.copy()
    if "sample_id" not in df.columns:
        df["sample_id"] = [f"s_{i}" for i in range(len(df))]

    slice_keys = [*group_cols, h3_col, label_col]

    # 1. centroids (robust by default)
    centroids = compute_slice_centroids(
        df,
        label_col=label_col,
        h3_level_name=h3_col,
        group_cols=group_cols,
        centroid_mode=centroid_mode,
        centroid_trim=centroid_trim,
    )
    df_c = df.merge(centroids, on=slice_keys, how="left")

    # 2. per-slice scoring
    results = []
    for _, g in df_c.groupby(slice_keys, group_keys=False):
        g = g.copy()
        g["slice_n"] = len(g)
        if len(g) < MIN_SCORING_SLICE_SIZE:
            for c in ["S", "mean_score", "rank_percentile",
                      "S_rank", "S_rank_min", "S_z"]:
                g[c] = 0.0
            results.append(g[[col for col in g.columns if "embedding" not in col
                              and col != "centroid"]])
            continue
        scored = compute_scores_for_slice(
            g,
            centroid=g["centroid"].iloc[0],
            norm_percentiles=norm_percentiles,
            max_full_pairwise_n=max_full_pairwise_n,
            force_knn=(max_full_pairwise_n == 0),
            knn_k=10,
            centroid_mode=centroid_mode,
            centroid_trim=centroid_trim,
        )
        scored["slice_n"] = len(g)
        results.append(scored)
    scored_df = pd.concat(results, ignore_index=True)
    # Rows in slices too small to define a centroid are never scored by the
    # production pipeline (they keep confidence 1.0); mark them so detection
    # metrics can be restricted to the population the detector operates on.
    scored_df["scored"] = scored_df["slice_n"] >= MIN_SCORING_SLICE_SIZE

    # 3. flagging
    flagged_df, _summary = flag_anomalies(
        scored_df,
        label_col=label_col,
        h3_level_name=h3_col,
        group_cols=group_cols,
        threshold_mode=threshold_mode,
        percentile_q=percentile_q,
        mad_k=mad_k,
        max_flagged_fraction=max_flagged_fraction,
    )

    # 4. confidence (flag-gated)
    flagged_df = add_confidence_from_score(
        flagged_df,
        score_col="mean_score",
        out_col="confidence",
        flagged_col="flagged" if gate_confidence_by_flag else None,
    )

    # 5. slice trust gating
    if apply_slice_trust:
        trust_df = compute_slice_trust(
            df[["sample_id", label_col, h3_col, *group_cols, embedding_col]],
            label_col=label_col,
            context_cols=[*group_cols, h3_col],
            embedding_col=embedding_col,
        )
        flagged_df = flagged_df.merge(
            trust_df[["sample_id", "slice_trust"]], on="sample_id", how="left"
        )
        flagged_df["slice_trust"] = flagged_df["slice_trust"].fillna(0.5)
        flagged_df = apply_trust_to_confidence(
            flagged_df, conf_col="confidence", trust_col="slice_trust",
            flagged_col="flagged", min_trust=slice_trust_min,
        )

    return flagged_df


# ---------------------------------------------------------------------------
# 3. Detection metrics
# ---------------------------------------------------------------------------


def evaluate_detection(
    scored_df: pd.DataFrame,
    *,
    truth_col: str = "noise_truth",
    score_col: str = "mean_score",
    flag_col: str = "flagged",
    k_list: Sequence[float] = (0.01, 0.05, 0.10),
    restrict_to_scored: bool = True,
    scored_col: str = "scored",
) -> Dict[str, float]:
    """Score how well the detector recovers the injected corruptions.

    Higher *score_col* must mean *more anomalous*.  Returns a metrics dict:

    - ``auroc``           : ranking quality of *score_col* vs ground truth.
    - ``average_precision``: area under the precision-recall curve (PR-AUC),
      the most informative single number under heavy class imbalance.
    - ``precision_at_{k}`` / ``recall_at_{k}`` : taking the top-``k`` fraction
      of points by score as the flagged set.
    - ``flag_precision`` / ``flag_recall`` / ``flag_f1`` : the realised quality
      of the discrete *flag_col* decision actually emitted by the pipeline.
    - ``n``, ``n_corrupt``, ``base_rate``.
    """
    from sklearn.metrics import (
        average_precision_score,
        roc_auc_score,
    )

    # Restrict to the population the detector can actually score: points in
    # slices too small to define a centroid are never assigned a score in
    # production (they keep confidence 1.0), so counting them as missed
    # detections would unfairly deflate ranking metrics.
    if restrict_to_scored and scored_col in scored_df.columns:
        scored_df = scored_df[scored_df[scored_col].fillna(False).astype(bool)]

    y = scored_df[truth_col].fillna(False).to_numpy(dtype=bool)
    s = pd.to_numeric(scored_df[score_col], errors="coerce").fillna(0.0).to_numpy()
    n = len(y)
    n_corrupt = int(y.sum())

    out: Dict[str, float] = {
        "n": int(n),
        "n_corrupt": n_corrupt,
        "base_rate": float(n_corrupt / n) if n else float("nan"),
    }

    if n_corrupt == 0 or n_corrupt == n:
        # degenerate: AUROC/AP undefined
        out["auroc"] = float("nan")
        out["average_precision"] = float("nan")
    else:
        out["auroc"] = float(roc_auc_score(y, s))
        out["average_precision"] = float(average_precision_score(y, s))

    order = np.argsort(-s)  # descending score
    y_sorted = y[order]
    for k in k_list:
        topk = max(1, int(round(k * n)))
        sel = y_sorted[:topk]
        tp = int(sel.sum())
        prec_k = tp / topk
        out[f"precision_at_{k:g}"] = float(prec_k)
        out[f"recall_at_{k:g}"] = float(tp / n_corrupt) if n_corrupt else float("nan")
        # Enrichment = how many times above the base rate planted errors are
        # concentrated in the top-k by score; >1 means the detector ranks them
        # above chance even when global AUROC is uninformative for this
        # per-slice, tail-gated detector.
        base = out["base_rate"]
        out[f"lift_at_{k:g}"] = float(prec_k / base) if base else float("nan")

    if flag_col in scored_df.columns:
        f = scored_df[flag_col].fillna(False).to_numpy(dtype=bool)
        tp = int((f & y).sum())
        fp = int((f & ~y).sum())
        fn = int((~f & y).sum())
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        out["flag_precision"] = float(prec)
        out["flag_recall"] = float(rec)
        out["flag_enrichment"] = float(prec / out["base_rate"]) if out["base_rate"] else float("nan")
        if prec and rec and np.isfinite(prec) and np.isfinite(rec) and (prec + rec) > 0:
            out["flag_f1"] = float(2 * prec * rec / (prec + rec))
        else:
            out["flag_f1"] = float("nan")
        out["n_flagged"] = int(f.sum())

    return out


# ---------------------------------------------------------------------------
# 4. Convenience runners
# ---------------------------------------------------------------------------


def run_noise_experiment(
    df: pd.DataFrame,
    spec: NoiseSpec,
    *,
    score_kwargs: Optional[dict] = None,
    eval_kwargs: Optional[dict] = None,
    score_col: str = "mean_score",
    baseline: Optional[str] = None,
) -> Dict[str, float]:
    """Inject noise → score on the noisy labels → evaluate recovery.

    Returns a single tidy row (dict) combining the noise spec and the metrics,
    suitable for appending into a results DataFrame.
    """
    score_kwargs = dict(score_kwargs or {})
    eval_kwargs = dict(eval_kwargs or {})

    noisy = inject_label_noise(df, spec)
    if baseline:
        from .baselines import score_with_baseline
        scored = score_with_baseline(
            noisy, baseline,
            label_col="label_noisy",
            h3_col=spec.h3_col,
            group_cols=score_kwargs.get("group_cols", ()),
            score_kwargs={k: v for k, v in score_kwargs.items()
                          if k != "group_cols"},
            seed=spec.seed,
        )
    else:
        scored = score_embeddings_df(
            noisy,
            label_col="label_noisy",
            h3_col=spec.h3_col,
            group_cols=score_kwargs.pop("group_cols", ()),
            **score_kwargs,
        )
    # carry ground truth onto the scored frame via sample_id
    truth_map = noisy.set_index("sample_id")["noise_truth"]
    scored["noise_truth"] = scored["sample_id"].map(truth_map).fillna(False)

    metrics = evaluate_detection(
        scored, truth_col="noise_truth", score_col=score_col, **eval_kwargs
    )
    row = {
        "detector": baseline or "eba",
        "mode": spec.mode,
        "rate": spec.rate,
        "seed": spec.seed,
        **metrics,
    }
    return row


def sweep(
    df: pd.DataFrame,
    *,
    modes: Sequence[str] = ("within_context", "random", "group"),
    rates: Sequence[float] = (0.02, 0.05, 0.10, 0.20),
    seeds: Sequence[int] = (0, 1, 2),
    label_col: str = "label",
    h3_col: str = "h3_l3_cell",
    group_col: Optional[str] = None,
    score_kwargs: Optional[dict] = None,
    score_col: str = "mean_score",
    baseline: Optional[str] = None,
) -> pd.DataFrame:
    """Sweep noise modes × rates × seeds and return a tidy results DataFrame
    (one row per run) ready for aggregation / plotting in the paper.

    Pass *baseline* in {"global","iforest","lof","spatial"} to score with a
    reference detector instead of the locality-aware EBA scorer."""
    rows: List[Dict[str, float]] = []
    for mode in modes:
        print(f"Running mode: {mode}")
        if mode in ("group", "parcel") and not group_col:
            continue
        for rate in rates:
            print(f"  Running rate: {rate}")
            for seed in seeds:
                print(f"    Running seed: {seed}")
                spec = NoiseSpec(
                    mode=mode, rate=rate, label_col=label_col,
                    h3_col=h3_col, group_col=group_col, seed=seed,
                )
                rows.append(
                    run_noise_experiment(
                        df, spec, score_kwargs=score_kwargs, score_col=score_col,
                        baseline=baseline,
                    )
                )
    return pd.DataFrame(rows)
