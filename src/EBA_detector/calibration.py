"""Absolute-scale calibration for the embedding-based anomaly detector.

Why this module exists
----------------------
The original scoring path was **purely relative within a slice**:
``cosine_distance`` and ``knn_distance`` were min–max normalised between the
2nd and 98th within-slice percentile and clipped to ``[0, 1]``, and the
escalation thresholds were fixed rank quantiles (``rank_percentile >= 0.98``
→ ``suspect``, ``>= 0.99`` → ``candidate``).

That construction forces **every** slice onto the same score range, so a
perfectly clean slice and a 30 %-mislabelled slice produce the same-looking
output.  Two failure modes follow directly:

* **False positives on clean slices.**  The top ~2 % / ~1 % of *every* slice
  reaches the ``suspect`` / ``candidate`` thresholds by construction, whether
  or not the slice contains a single genuine error.
* **False negatives on heavily contaminated slices.**  The within-slice
  ``median + k·MAD`` gate is evaluated on a bounded score, so once the
  contaminant is a large enough sub-population it inflates the slice's own
  median and MAD until the threshold exceeds the score ceiling — and the
  worst slices flag *nothing*.

The cure is a reference scale that a single slice cannot move.  This module
estimates a **pooled null distribution of raw distances** from the *consensus
across many slices of the same class*, then expresses every point's distance
as a z-score against that null.

Because the pool is built from **one summary statistic per slice** (the slice
median and the slice MAD), a contaminated slice contributes a single
observation and cannot drag the null toward itself.  Its points therefore keep
a genuinely large ``abs_z`` and *do* get flagged, while a clean slice's
relatively-highest point sits at an ordinary ``abs_z`` and does *not*.

Known limit
-----------
Calibration removes the *artificial* blind spot — the one created by letting a
slice set its own scale.  It does not remove the **identifiability** limit: once
the mislabelled points approach half of their own class within a region, no
purely geometric method can say which half is wrong, because "the wrong ones"
and "the right ones" are simply two modes of equal standing.  Measured on the
synthetic harness, detection stays strong while the errors are up to roughly a
third of their class slice and degrades beyond that.

That regime is exactly what ``ref_id``-level aggregation
(:func:`EBA_detector.robust_extensions.aggregate_parcel_scores`) and the
``time_col`` split are for: if half of one dataset's records in a region are
wrong, the evidence lives at the level of *the dataset*, not the point.  Report
it there rather than pretending the point-wise scores can resolve it.

Usage
-----
>>> null_ref = compute_null_reference(scored_df, null_keys=["label"])
>>> scored_df = add_absolute_scores(scored_df, null_ref, null_keys=["label"])
>>> # scored_df now has cos_abs_z, knn_abs_z, abs_z  (units: robust sigma)
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "MAD_TO_SIGMA",
    "compute_null_reference",
    "add_absolute_scores",
    "suggest_abs_z_threshold",
]

# Consistency constant: for a Gaussian, MAD = sigma / 1.4826.  Multiplying the
# MAD by this puts ``abs_z`` in familiar sigma units so the thresholds below
# read the way a reader expects (3.0 ~ a one-sided 0.1 % tail under normality).
MAD_TO_SIGMA: float = 1.4826

# Cosine distances live in [0, 2].  A pooled scale at or below this is not a
# small dispersion, it is *no* dispersion — the population is degenerate
# (near-duplicate embeddings, e.g. grid-sampled polygon interiors that the
# encoder maps to almost the same vector).
#
# Such a scale must NOT be floored and used as a divisor.  Flooring it silently
# turns the absolute gate into a hair trigger: dividing a normal distance by
# 1e-4 yields z-scores in the thousands, so every point of every *ordinary*
# slice in that class clears the gate.  Measured on 5 duplicate-heavy slices
# plus 2 ordinary ones, flooring flagged 46% of both ordinary slices.
#
# Instead the scale becomes NaN, which propagates to abs_z (never flagged) and
# to null_scale_sigma (so `stable_mad` sees a non-finite sigma and thresholds
# at +inf).  This matches the long-standing policy of the legacy `mad` mode:
# when the robust scale collapses there is nothing to threshold on, so flag
# nothing.
_DEGENERATE_SCALE: float = 1e-6


def _slice_stats(values: np.ndarray, estimator: str = "left_tail") -> tuple:
    """Robust (location, scale) of one slice's raw distances.

    ``estimator``
        ``"left_tail"`` (default)
            ``scale = median - q25``.  Under symmetry this is *identical* to the
            MAD (for a Gaussian both equal 0.6745 sigma), but it is computed
            **only from the left half** of the distribution.

            That matters because label errors push distances to the *right*.
            The MAD uses both sides, so a slice that is 40 % contaminated has a
            hugely inflated MAD — and since the cross-slice null is the median
            of the per-slice scales, contamination present in *every* slice
            inflates the null itself.  The z-scores then shrink and nothing
            clears the gate: measured recall collapsed from 0.93 at 30 %
            contamination to 0.06 at 40 % and 0.01 at 45 %.

            Estimating the scale from the clean left half keeps the null honest
            for any right-side contamination below 50 %.  Measured at a matched
            (in fact slightly lower) clean false-positive rate, this lifts
            recall at 40 % contamination from 0.009 to 0.42, and at 30 % from
            0.71 to 0.93.

        ``"mad"``
            The classic median-absolute-deviation.  Kept for ablations; do not
            use it when slices may carry more than ~20 % errors.

    Returns the scale in MAD units, so callers can keep multiplying by
    :data:`MAD_TO_SIGMA` regardless of which estimator produced it.
    """
    med = float(np.nanmedian(values))
    if estimator == "mad":
        return med, float(np.nanmedian(np.abs(values - med)))
    if estimator != "left_tail":
        raise ValueError("estimator must be one of {'left_tail','mad'}")
    q25 = float(np.nanquantile(values, 0.25))
    return med, max(med - q25, 0.0)


def compute_null_reference(
    df_scores: pd.DataFrame,
    *,
    null_keys: Sequence[str],
    slice_key_cols: Sequence[str],
    metric_cols: Sequence[str] = (
        "cosine_distance",
        "knn_distance_fixed",
        "knn_distance",
        "neighbourhood_offset",
    ),
    min_slice_n: int = 30,
    min_slices: int = 2,
    scored_mask_col: Optional[str] = None,
    scale_estimator: str = "left_tail",
    shrink_k: float = 5.0,
) -> pd.DataFrame:
    """Estimate a pooled null location/scale of raw distances per *null_keys*.

    The estimator is deliberately a **median of per-slice statistics**, not a
    pooled median over points:

    1. For every slice (``slice_key_cols``) with at least *min_slice_n* scored
       points, compute the slice's median and MAD of each raw distance.
    2. For every ``null_keys`` group (typically the label class), take the
       median of those slice medians and the median of those slice MADs.

    Step 2 gives every slice equal weight, so a large contaminated slice
    contributes exactly one (median, MAD) pair and cannot inflate the
    reference.  This is what lets a uniformly-bad slice be detected as bad
    instead of self-calibrating its own errors away.

    Groups with fewer than *min_slices* contributing slices fall back to the
    global (all-data) null, which is returned as the ``__GLOBAL__`` row.

    Parameters
    ----------
    df_scores
        Scored frame containing *metric_cols*, *null_keys* and *slice_key_cols*.
    null_keys
        Grouping for the null — usually ``[label_col]``.  Add a coarse spatial
        or temporal key (e.g. a continent or year column) when distance scales
        differ systematically across them.
    slice_key_cols
        The columns identifying a scoring slice.
    metric_cols
        Raw (un-normalised) distance columns to calibrate.
    min_slice_n
        Slices with fewer scored points than this do not contribute to the null.
    min_slices
        Minimum contributing slices before a group is given a local estimate at
        all.  With shrinkage (below) this is only a floor, not a cliff.
    shrink_k
        Strength of the shrinkage toward the global null::

            w    = n_slices / (n_slices + shrink_k)
            null = w * local_null + (1 - w) * global_null

        Localising the null is necessary — a class's legitimate dispersion
        differs region to region, and a globally pooled null makes every
        more-variable region look anomalous as a whole.  But a *hard* local null
        trades that bias for variance: estimated from a handful of slices it is
        noisy, and the noise becomes false positives of its own.  Measured on
        four real-geography regions with 5-6 slices each, a hard local null
        raised the clean false-positive rate from 1.03 % to 3.54 % (7.70 % in
        the tightest region) — worse than pooling.

        Shrinkage spends locality in proportion to the evidence for it: a region
        with 50 contributing slices sits at w = 0.83 and is essentially local; a
        region with 3 sits at w = 0.23 and is essentially global.  There is no
        threshold to trip over, and sparse regions degrade smoothly instead of
        falling off a cliff at *min_slices*.
    scored_mask_col
        Optional boolean column; when given only rows where it is True are used.
    scale_estimator
        ``"left_tail"`` (default) or ``"mad"`` — see :func:`_slice_stats`.  The
        default is what keeps the null usable when contamination is present in
        every slice of a class.

    Returns
    -------
    pd.DataFrame
        One row per null group with columns
        ``{metric}_null_loc``, ``{metric}_null_scale``, ``n_slices``,
        plus the *null_keys*.  A sentinel row with ``__is_global__ = True``
        carries the fallback.
    """
    # Drop conditioners that are not columns here.  The default null key
    # ``h3_effective_level`` only exists in adaptive H3 mode, so a fixed-level
    # run (or a caller passing a column this frame does not carry) must degrade
    # to a coarser null rather than raising deep inside a groupby.
    requested_null_keys = list(null_keys)
    null_keys = [k for k in requested_null_keys if k in df_scores.columns]
    dropped = [k for k in requested_null_keys if k not in df_scores.columns]
    if dropped:
        print(
            f"[calibration] NOTE: null conditioner(s) {dropped} not present; "
            "the null is pooled without them."
        )
    slice_key_cols = [c for c in slice_key_cols if c in df_scores.columns]
    metric_cols = [c for c in metric_cols if c in df_scores.columns]
    # Prefer the fixed-k variant, which removes the k-dependence of the
    # adaptive knn_distance.  It does NOT remove the density component of the
    # size effect (see compute_scores_for_slice for the measured numbers), but
    # that residual is inert: abs_z is min(centroid, neighbourhood) and the
    # near-unbiased centroid term governs the gate.  knn_distance remains as a
    # fallback for legacy scored frames predating knn_distance_fixed.
    if "knn_distance_fixed" in metric_cols and "knn_distance" in metric_cols:
        metric_cols = [c for c in metric_cols if c != "knn_distance"]
    if not metric_cols:
        raise ValueError(
            "compute_null_reference: none of the requested metric_cols are present"
        )

    df = df_scores
    if scored_mask_col is not None and scored_mask_col in df.columns:
        df = df[df[scored_mask_col].fillna(False).astype(bool)]

    if df.empty:
        raise ValueError("compute_null_reference: no scored rows to calibrate on")

    # ---- Step 1: one (median, MAD) pair per slice -------------------------
    group_cols = list(dict.fromkeys([*null_keys, *slice_key_cols]))
    rows: List[dict] = []
    for key, g in df.groupby(group_cols, dropna=False, sort=True):
        if len(g) < min_slice_n:
            continue
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        ok = False
        for m in metric_cols:
            vals = pd.to_numeric(g[m], errors="coerce").to_numpy(dtype="float64")
            vals = vals[np.isfinite(vals)]
            if vals.size < min_slice_n:
                continue
            med, mad = _slice_stats(vals, estimator=scale_estimator)
            row[f"{m}__med"] = med
            row[f"{m}__mad"] = mad
            ok = True
        if ok:
            rows.append(row)

    stat_cols = [f"{m}__{s}" for m in metric_cols for s in ("med", "mad")]

    if not rows:
        # Nothing large enough to calibrate on — degrade to a global null over
        # all points rather than silently disabling the absolute gate.
        glob = {"__is_global__": True, "n_slices": 0}
        for m in metric_cols:
            vals = pd.to_numeric(df[m], errors="coerce").to_numpy(dtype="float64")
            vals = vals[np.isfinite(vals)]
            med, mad = (
                _slice_stats(vals, estimator=scale_estimator)
                if vals.size
                else (0.0, float("nan"))
            )
            glob[f"{m}_null_loc"] = med
            glob[f"{m}_null_scale"] = mad if mad > _DEGENERATE_SCALE else float("nan")
        out = pd.DataFrame([glob])
        for k in null_keys:
            out[k] = None
        return out

    slice_stats = pd.DataFrame(rows)

    # ---- Step 2: median of the per-slice statistics, per null group -------
    agg_map = {c: "median" for c in stat_cols if c in slice_stats.columns}
    grouped = slice_stats.groupby(null_keys, dropna=False, sort=True)
    null_df = grouped.agg(agg_map).reset_index()
    null_df["n_slices"] = grouped.size().to_numpy()

    rename = {}
    for m in metric_cols:
        rename[f"{m}__med"] = f"{m}_null_loc"
        rename[f"{m}__mad"] = f"{m}_null_scale"
    null_df = null_df.rename(columns=rename)

    contributing = [m for m in metric_cols if f"{m}__med" in slice_stats.columns]
    for m in contributing:
        sc = f"{m}_null_scale"
        if sc in null_df.columns:
            # Degenerate -> NaN, never floored.  See _DEGENERATE_SCALE.
            null_df[sc] = null_df[sc].where(null_df[sc] > _DEGENERATE_SCALE)

    # Groups with too little support use the global null instead of a noisy own one
    # --- shrink each local null toward the global one ---------------------
    # w = n/(n+shrink_k): a well-supported region keeps its own null, a
    # thinly-supported one borrows the global.  Degenerate (NaN) local scales
    # are NOT rescued by the global value - see _DEGENERATE_SCALE.
    _g_loc, _g_scale = {}, {}
    for m in contributing:
        _g_loc[m] = float(np.nanmedian(slice_stats[f"{m}__med"].to_numpy()))
        _gs = float(np.nanmedian(slice_stats[f"{m}__mad"].to_numpy()))
        _g_scale[m] = _gs if _gs > _DEGENERATE_SCALE else float("nan")

    if shrink_k and shrink_k > 0 and not null_df.empty:
        _w = (
            null_df["n_slices"].to_numpy(dtype="float64")
            / (null_df["n_slices"].to_numpy(dtype="float64") + float(shrink_k))
        )
        null_df["null_shrink_w"] = _w.astype(np.float32)
        for m in contributing:
            lc, sc = f"{m}_null_loc", f"{m}_null_scale"
            if lc in null_df.columns and np.isfinite(_g_loc[m]):
                loc = null_df[lc].to_numpy(dtype="float64")
                null_df[lc] = np.where(
                    np.isfinite(loc), _w * loc + (1.0 - _w) * _g_loc[m], _g_loc[m]
                )
            if sc in null_df.columns and np.isfinite(_g_scale[m]):
                scl = null_df[sc].to_numpy(dtype="float64")
                # NaN local scale = degenerate; leave it NaN.
                null_df[sc] = np.where(
                    np.isfinite(scl), _w * scl + (1.0 - _w) * _g_scale[m], np.nan
                )
    else:
        null_df["null_shrink_w"] = np.float32(1.0)

    # Only metrics that actually contributed a per-slice statistic can appear in
    # the global row.  Indexing every requested metric here raised a KeyError
    # whenever a column was present on the frame but never reached min_slice_n
    # finite values in any slice — which is exactly what happens when a legacy
    # scored frame lacking `neighbourhood_offset` data is passed in.
    global_row = {"__is_global__": True, "n_slices": int(len(slice_stats))}
    for m in contributing:
        global_row[f"{m}_null_loc"] = float(
            np.nanmedian(slice_stats[f"{m}__med"].to_numpy())
        )
        g_scale = float(np.nanmedian(slice_stats[f"{m}__mad"].to_numpy()))
        global_row[f"{m}_null_scale"] = (
            g_scale if g_scale > _DEGENERATE_SCALE else float("nan")
        )
    for k in null_keys:
        global_row[k] = None

    null_df["__is_global__"] = False
    null_df = null_df[null_df["n_slices"] >= int(min_slices)].copy()
    # Marks "this group has its own null" so add_absolute_scores can tell a
    # group that is ABSENT (too few slices -> legitimately borrow the global
    # null) from one that is PRESENT but degenerate (must not be rescued).
    null_df["__has_own_null__"] = True

    out = pd.concat([null_df, pd.DataFrame([global_row])], ignore_index=True)
    return out


def add_absolute_scores(
    df_scores: pd.DataFrame,
    null_ref: pd.DataFrame,
    *,
    null_keys: Sequence[str],
    metric_cols: Sequence[str] = (
        "cosine_distance",
        "knn_distance_fixed",
        "knn_distance",
        "neighbourhood_offset",
    ),
    out_prefixes: Sequence[str] = ("cos", "knn", "nbr"),
    combine: str = "min",
) -> pd.DataFrame:
    """Attach absolute z-scores (in robust sigma units) using *null_ref*.

    ``abs_z`` combines a **centroid** term and a **neighbourhood** term with
    *combine*:

    * ``"min"`` (default) — a point must look anomalous on both.  This is what
      suppresses false positives: a point far from the centroid but sitting
      inside a dense, well-placed same-class neighbourhood (a legitimate
      sub-type — a late-planted field, an unusual cultivar) scores low.
    * ``"max"`` — either term suffices (higher recall, more false positives).
    * ``"mean"`` — average of the two.

    Masking-aware neighbourhood evidence
    ------------------------------------
    A small kNN distance normally means "well supported by same-class
    neighbours".  That inference fails when the errors are **coherent** — a
    whole dataset, region or season mislabelled the same way forms its own
    dense cluster, so every member finds its fellow errors as neighbours and
    looks like a perfect inlier.  Measured on a 30 %-contaminated slice, the
    planted errors have a median ``knn_abs_z`` of ~0.45 while their
    ``cos_abs_z`` is ~5.3: under a naive ``min`` the kNN term vetoes the
    correct centroid evidence and recall collapses to ~1 %.

    The neighbourhood term is therefore **not** the raw kNN z-score but::

        neighbour_evidence = max(knn_abs_z, nbr_abs_z)

    where ``nbr_abs_z`` comes from ``neighbourhood_offset`` — how far the
    point's *own neighbourhood* sits from the class centroid.  A genuine
    inlier has both small; a member of a coherent wrong cluster has a tiny kNN
    distance but a large neighbourhood offset, so it can no longer hide behind
    its accomplices.

    Rows whose null group is absent from *null_ref* fall back to the global
    null row.  Rows with non-finite distances get ``abs_z = NaN`` (they are
    never flagged and are surfaced by the quality gate instead).
    """
    if combine not in {"min", "max", "mean"}:
        raise ValueError("combine must be one of {'min','max','mean'}")

    # Mirror compute_null_reference: a conditioner absent from either side is
    # dropped, so the null_ref built above and the merge below agree on keys.
    null_keys = [
        k for k in null_keys if k in df_scores.columns and k in null_ref.columns
    ]
    # Only calibrate metrics that exist both on the data and in the null
    # reference — callers may legitimately pass a null built for fewer metrics
    # (e.g. an older cached reference without neighbourhood_offset).
    metric_cols = [
        m
        for m in metric_cols
        if m in df_scores.columns and f"{m}_null_loc" in null_ref.columns
    ]
    # Prefer the size-independent fixed-k metric; see compute_null_reference.
    if "knn_distance_fixed" in metric_cols and "knn_distance" in metric_cols:
        metric_cols = [m for m in metric_cols if m != "knn_distance"]
    if not metric_cols:
        raise ValueError(
            "add_absolute_scores: no metric is present in both df_scores and null_ref"
        )
    out = df_scores.copy()

    global_rows = null_ref[null_ref.get("__is_global__", False) == True]  # noqa: E712
    if global_rows.empty:
        raise ValueError("null_ref has no global fallback row")
    global_row = global_rows.iloc[0]

    per_group = null_ref[null_ref.get("__is_global__", False) != True]  # noqa: E712

    loc_cols = [f"{m}_null_loc" for m in metric_cols]
    scale_cols = [f"{m}_null_scale" for m in metric_cols]
    join_cols = [c for c in (loc_cols + scale_cols) if c in null_ref.columns]

    if not per_group.empty and all(k in out.columns for k in null_keys):
        _pg = per_group[null_keys + join_cols].copy()
        _pg["__has_own_null__"] = True
        out = out.merge(_pg, on=null_keys, how="left")
        has_own = out["__has_own_null__"].fillna(False).to_numpy(dtype=bool)
        out = out.drop(columns=["__has_own_null__"])
    else:
        for c in join_cols:
            out[c] = np.nan
        has_own = np.zeros(len(out), dtype=bool)

    # Re-derive prefixes only when the caller left the default in place; an
    # explicit out_prefixes used to be accepted and then silently overwritten,
    # so a caller reading e.g. "a_abs_z" got a KeyError.
    _DEFAULT_PREFIXES = ("cos", "knn", "nbr")
    prefix_for = {
        "cosine_distance": "cos",
        "knn_distance": "knn",
        "knn_distance_fixed": "knn",
        "neighbourhood_offset": "nbr",
    }
    if tuple(out_prefixes) == _DEFAULT_PREFIXES:
        out_prefixes = [prefix_for.get(m, m) for m in metric_cols]
    else:
        out_prefixes = list(out_prefixes)
        if len(out_prefixes) != len(metric_cols):
            raise ValueError(
                f"out_prefixes has {len(out_prefixes)} entries but "
                f"{len(metric_cols)} metric columns are usable: {metric_cols}"
            )

    z_cols: List[str] = []
    for m, pref in zip(metric_cols, out_prefixes):
        loc_c, scale_c = f"{m}_null_loc", f"{m}_null_scale"
        if loc_c not in out.columns:
            out[loc_c] = np.nan
        if scale_c not in out.columns:
            out[scale_c] = np.nan
        # A group that is ABSENT from null_ref (too few slices) legitimately
        # borrows the global null.  A group that is PRESENT but whose scale is
        # NaN is *degenerate* and must NOT be rescued by the global scale —
        # doing so would resurrect the hair-trigger described at
        # _DEGENERATE_SCALE.  `has_own` separates the two cases.
        g_loc = float(global_row[loc_c]) if loc_c in global_row else float("nan")
        g_scale = float(global_row[scale_c]) if scale_c in global_row else float("nan")

        loc = out[loc_c].to_numpy(dtype="float64").copy()
        scale = out[scale_c].to_numpy(dtype="float64").copy()
        borrow = ~has_own
        loc[borrow & ~np.isfinite(loc)] = g_loc
        scale[borrow & ~np.isfinite(scale)] = g_scale
        # Degenerate (or still-missing) scales stay NaN and propagate.
        scale = np.where(scale > _DEGENERATE_SCALE, scale, np.nan) * MAD_TO_SIGMA

        x = pd.to_numeric(out[m], errors="coerce").to_numpy(dtype="float64")
        z = (x - loc) / scale
        zc = f"{pref}_abs_z"
        out[zc] = z.astype(np.float32)
        z_cols.append(zc)

    # Collapse the two neighbourhood-based metrics into a single term BEFORE
    # combining with the centroid term, so that a self-consistent error cluster
    # (small kNN distance, large neighbourhood offset) cannot veto the centroid
    # evidence.  See the docstring.
    centroid_z_col = z_cols[0]
    neighbour_z_cols = [c for c in z_cols[1:]]

    # All-NaN rows are expected (unscored slices, quarantined embeddings) and
    # are handled explicitly below, so numpy's "All-NaN slice" notice is noise.
    import warnings as _warnings

    with np.errstate(invalid="ignore"), _warnings.catch_warnings():
        _warnings.filterwarnings(
            "ignore", message="All-NaN slice encountered", category=RuntimeWarning
        )
        if neighbour_z_cols:
            NB = out[neighbour_z_cols].to_numpy(dtype="float64")
            nb_all_nan = ~np.isfinite(NB).any(axis=1)
            neighbour_z = np.where(nb_all_nan, np.nan, np.nanmax(NB, axis=1))
            out["neighbour_abs_z"] = neighbour_z.astype(np.float32)
            terms = np.column_stack(
                [out[centroid_z_col].to_numpy(dtype="float64"), neighbour_z]
            )
        else:
            terms = out[[centroid_z_col]].to_numpy(dtype="float64")

        if combine == "min":
            abs_z = np.nanmin(terms, axis=1)
        elif combine == "max":
            abs_z = np.nanmax(terms, axis=1)
        else:
            abs_z = np.nanmean(terms, axis=1)

    # all-NaN rows -> NaN (never flagged)
    all_nan = ~np.isfinite(terms).any(axis=1)
    abs_z = np.where(all_nan, np.nan, abs_z)

    out["abs_z"] = abs_z.astype(np.float32)
    # How many of the (centroid, neighbourhood) terms were actually finite.
    # Under combine="min" a row with only one finite term silently degrades to
    # single-metric evidence — the "must agree on both" guarantee no longer
    # holds for it.  Recording the count makes that degradation auditable
    # instead of invisible.
    out["abs_z_n_terms"] = np.isfinite(terms).sum(axis=1).astype(np.int8)
    if combine == "min":
        _n_degraded = int(((out["abs_z_n_terms"].to_numpy() == 1)).sum())
        if _n_degraded:
            print(
                f"[calibration] NOTE: {_n_degraded:,} rows have only one finite "
                "abs_z term; combine='min' is single-metric evidence for them "
                "(see abs_z_n_terms)."
            )

    # Retain the null *scale* of the primary metric.  ``flag_anomalies``'s
    # ``stable_mad`` mode uses it as the dispersion for the within-slice test,
    # keeping the local reference (the slice median) while borrowing a scale
    # the slice cannot inflate.  Without this the local MAD gate stays a
    # knife-edge and silently suppresses detection in exactly the slices that
    # need it most.
    primary = metric_cols[0]
    prim_scale_c = f"{primary}_null_scale"
    prim_scale = out[prim_scale_c].to_numpy(dtype="float64").copy()
    _g_prim = (
        float(global_row[prim_scale_c]) if prim_scale_c in global_row else float("nan")
    )
    prim_scale[(~has_own) & ~np.isfinite(prim_scale)] = _g_prim
    out["null_scale_sigma"] = (
        np.where(prim_scale > _DEGENERATE_SCALE, prim_scale, np.nan) * MAD_TO_SIGMA
    ).astype(np.float32)

    out = out.drop(columns=[c for c in (loc_cols + scale_cols) if c in out.columns])
    return out


def suggest_abs_z_threshold(
    df_scores: pd.DataFrame,
    *,
    target_flag_fraction: float = 0.02,
    abs_z_col: str = "abs_z",
    scored_mask_col: Optional[str] = None,
) -> float:
    """Return the ``abs_z`` cut that would flag *target_flag_fraction* overall.

    This is a **diagnostic**, not the default policy — the whole point of the
    absolute scale is that the realised flag rate should be allowed to differ
    between clean and dirty populations.  Use it to sanity-check that your
    chosen ``abs_z_k`` is in a sensible neighbourhood, or to report "the
    threshold equivalent to a 2 % budget was z = 3.4" in a paper.
    """
    df = df_scores
    if scored_mask_col is not None and scored_mask_col in df.columns:
        df = df[df[scored_mask_col].fillna(False).astype(bool)]
    z = pd.to_numeric(df[abs_z_col], errors="coerce").to_numpy(dtype="float64")
    z = z[np.isfinite(z)]
    if z.size == 0:
        return float("nan")
    q = 1.0 - float(np.clip(target_flag_fraction, 1e-6, 0.999))
    return float(np.quantile(z, q))
