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


# Separator for composite null-group keys.  ASCII unit separator: it cannot
# occur in an H3 cell id, a WorldCereal class name or a resolution, so
# "a" + SEP + "bc" can never collide with "ab" + SEP + "c".
_NULL_KEY_SEP: str = "\x1f"
# Explicit token for a missing key component.  ASCII group separator: it can
# never be produced by `astype(str)` on an H3 cell id, a class name or an
# integer resolution.  Deliberately NOT a NUL — pandas' object-dtype string
# hashtable truncates at NUL, so a NUL-bearing sentinel made
# `Series.nunique()` under-count (12 distinct keys reported as 3).  The Index
# hashtable used for the actual lookup handled it correctly, but relying on
# that distinction is not worth the risk.
_NULL_KEY_NA: str = "\x1d"


def _null_key_array(frame: pd.DataFrame, cols) -> np.ndarray:
    """Vectorised composite key for *cols*, as an object array of strings.

    Used instead of a pandas merge so that the null can be looked up at
    several prefix depths without materialising a join per depth.

    Missing components are materialised as *_NULL_KEY_NA* BEFORE the pieces are
    joined.  This is not cosmetic.  Under pandas' string dtype ``astype(str)``
    preserves NA rather than rendering it ``"nan"``, and NA propagates through
    concatenation — so a single missing region collapsed the WHOLE composite
    key to NA.  Every row with an underivable region, of every class, then
    shared one key, matched whichever group the de-duplication happened to keep
    first, and was calibrated against another class's null.  Measured on a
    two-class repro that gave class B a mean ``cos_abs_z`` of 34 against class
    A's null, with nothing raised and nothing printed.  A row with no region is
    a legitimate group of its own ("unknown region"), and the ladder lets it
    back off to the class-level null if that group is thin.
    """
    cols = list(cols)
    if not cols:
        return np.full(len(frame), "", dtype=object)
    out = None
    for c in cols:
        col = frame[c]
        piece = col.astype(str).astype(object).mask(col.isna(), _NULL_KEY_NA)
        out = piece if out is None else out + _NULL_KEY_SEP + piece
    return out.to_numpy(dtype=object)


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
            null = w * own_estimate + (1 - w) * parent_null

        Localising the null is necessary — a class's legitimate dispersion
        differs region to region, and a globally pooled null makes every
        more-variable region look anomalous as a whole.  But a *hard* local null
        trades that bias for variance: estimated from a handful of slices it is
        noisy, and the noise becomes false positives of its own.  Measured on
        four real-geography regions with 5-6 slices each, a hard local null
        raised the clean false-positive rate from 1.03 % to 3.54 % (7.70 % in
        the tightest region) — worse than pooling.

        The target is the group's PARENT in the ladder, not the flat global
        null — that is what keeps a thin (class, region, resolution) group on
        (class, region) instead of stranding it.

        Shrinkage spends locality in proportion to the evidence for it: a region
        with 50 contributing slices sits at w = 0.91 and is essentially local; a
        region with 3 sits at w = 0.38 and is essentially its parent.  There is no
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
    _req_slice_keys = list(slice_key_cols)
    slice_key_cols = [c for c in _req_slice_keys if c in df_scores.columns]
    if len(slice_key_cols) != len(_req_slice_keys):
        # Symmetric with the null_keys note above.  A misspelled slice key
        # coarsens every "slice", which silently thins the null and changes
        # what "one observation" means, so it must not pass in silence.
        print(
            "[calibration] NOTE: slice key(s) "
            f"{[c for c in _req_slice_keys if c not in slice_key_cols]} not "
            "present; slices are defined without them."
        )
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

    # ---- Step 2: a HIERARCHICAL ladder of nulls ---------------------------
    #
    # `null_keys` is read as a nesting, coarsest conditioner first, e.g.
    #
    #     [label, h3_null_region, h3_null_res]
    #
    # and a null is estimated at every prefix depth:
    #
    #     depth 0 : global            (all slices)
    #     depth 1 : (label,)
    #     depth 2 : (label, region)
    #     depth 3 : (label, region, res)
    #
    # Each depth is shrunk toward its own PARENT, not toward the flat global::
    #
    #     w      = n_slices / (n_slices + shrink_k)      (0 below min_slices)
    #     null_d = w * own_d + (1 - w) * null_{d-1}
    #
    # Why the cascade matters.  Conditioning the null on the slice's H3
    # resolution is necessary — a distance distribution scales with cell size,
    # so pooling L2 (86,802 km2) with L4 (1,770 km2) slices hands the tight L4
    # slices a scale inflated by the coarse ones and their real outliers fall
    # under the gate.  But every extra conditioner thins the groups, and a
    # thinly-supported group used to fall all the way back to the flat global
    # null — reinstating exactly the mixing the conditioner was added to
    # remove, and doing it for the rare classes in the finest cells, which for
    # CROPTYPE24 is most of them.  Backing off one level at a time keeps the
    # resolution conditioning even when the region-level group is empty.
    #
    # All depths are returned; `add_absolute_scores` matches deepest-first.
    # ------------------------------------------------------------------
    contributing = [m for m in metric_cols if f"{m}__med" in slice_stats.columns]

    # depth 0 — the flat global null, the root of the cascade.
    _g_loc, _g_scale = {}, {}
    for m in contributing:
        _g_loc[m] = float(np.nanmedian(slice_stats[f"{m}__med"].to_numpy()))
        _gs = float(np.nanmedian(slice_stats[f"{m}__mad"].to_numpy()))
        _g_scale[m] = _gs if _gs > _DEGENERATE_SCALE else float("nan")

    agg_map = {c: "median" for c in stat_cols if c in slice_stats.columns}
    rename = {}
    for m in metric_cols:
        rename[f"{m}__med"] = f"{m}_null_loc"
        rename[f"{m}__mad"] = f"{m}_null_scale"

    ladder: List[pd.DataFrame] = []
    parent_frame: Optional[pd.DataFrame] = None
    parent_keys: List[str] = []
    parent_depth = 0
    for depth in range(1, len(null_keys) + 1):
        keys_d = list(null_keys[:depth])
        grouped = slice_stats.groupby(keys_d, dropna=False, sort=True)
        nd = grouped.agg(agg_map).reset_index().rename(columns=rename)
        nd["n_slices"] = grouped.size().to_numpy()

        # A rung that partitions exactly like its parent carries no
        # information — and is not merely redundant, it is harmful.  Two
        # identical rungs compose their shrinkage: w_eff = 1 - (1 - w)^2, so a
        # thin group's weight on its own noisy estimate silently rises (0.29 ->
        # 0.49 at n = 2), i.e. adding a zero-information key halves the
        # insurance the shrinkage was there to provide.  This bites on any
        # FIXED single-resolution run, where `h3_null_res` is a constant.
        # Equal group counts prove identical partitions: every parent group
        # holds at least one child group, so equal totals force exactly one.
        if parent_frame is not None and len(nd) == len(parent_frame):
            print(
                f"[calibration] NOTE: null key {null_keys[depth - 1]!r} does not "
                f"subdivide {parent_keys} ({len(nd)} groups either way); "
                "skipping that rung."
            )
            continue

        for m in contributing:
            sc = f"{m}_null_scale"
            if sc in nd.columns:
                # Degenerate -> NaN, never floored.  See _DEGENERATE_SCALE.
                nd[sc] = nd[sc].where(nd[sc] > _DEGENERATE_SCALE)

        # --- the parent estimate this depth shrinks toward -----------------
        par_loc, par_scale = {}, {}
        par_n = None
        if parent_frame is None:
            par_eff = np.zeros(len(nd), dtype="float64")
            for m in contributing:
                par_loc[m] = np.full(len(nd), _g_loc[m], dtype="float64")
                par_scale[m] = np.full(len(nd), _g_scale[m], dtype="float64")
        else:
            lut = parent_frame.copy()
            lut.index = pd.Index(_null_key_array(lut, parent_keys))
            lut = lut[~lut.index.duplicated(keep="first")]
            want = pd.Index(_null_key_array(nd, parent_keys))
            for m in contributing:
                lc, sc = f"{m}_null_loc", f"{m}_null_scale"
                par_loc[m] = (
                    lut[lc].reindex(want).to_numpy(dtype="float64")
                    if lc in lut.columns
                    else np.full(len(nd), np.nan)
                )
                par_scale[m] = (
                    lut[sc].reindex(want).to_numpy(dtype="float64")
                    if sc in lut.columns
                    else np.full(len(nd), np.nan)
                )
            par_eff = (
                lut["__eff_depth__"].reindex(want).to_numpy(dtype="float64")
                if "__eff_depth__" in lut.columns
                else np.full(len(nd), float(parent_depth))
            )
            par_eff = np.where(np.isfinite(par_eff), par_eff, 0.0)
            par_n = lut["n_slices"].reindex(want).to_numpy(dtype="float64")

        n = nd["n_slices"].to_numpy(dtype="float64")
        if shrink_k and shrink_k > 0:
            w = n / (n + float(shrink_k))
        else:
            w = np.ones_like(n)
        # Below min_slices a group keeps no opinion of its own; it *is* its
        # parent.  This is the floor, not a cliff — shrinkage already fades
        # thin groups out smoothly above it.
        supported = n >= float(min_slices)
        w = np.where(supported, w, 0.0)
        # A group that is its parent's ONLY child adds no information: it was
        # estimated from exactly the same slices.  Blending it toward the
        # parent would compose the two shrinkages (w_eff = 1 - (1 - w)^2) and
        # pull it AWAY from the pooled estimate rather than toward it.  Equal
        # slice counts prove sole-childhood: a parent that splits gives every
        # child strictly fewer slices.  (The whole-rung skip above is the case
        # where this holds for every group at once.)
        if par_n is not None:
            w = np.where(np.isfinite(par_n) & (n >= par_n), 0.0, w)
        nd["null_shrink_w"] = w.astype(np.float32)
        # The deepest rung that actually contributed anything.  A w = 0 group
        # equals its parent exactly, so stamping it with its own depth would
        # make the localisation diagnostic claim a locality the numbers do not
        # have — and that diagnostic is the only instrument for "are these keys
        # too fine for this collection's density?".
        nd["__eff_depth__"] = np.where(w > 0.0, float(depth), par_eff)

        for m in contributing:
            lc, sc = f"{m}_null_loc", f"{m}_null_scale"
            if lc in nd.columns:
                own = nd[lc].to_numpy(dtype="float64")
                par = par_loc[m]
                blend = np.where(np.isfinite(par), w * own + (1.0 - w) * par, own)
                # No own estimate -> inherit the parent's location outright.
                nd[lc] = np.where(np.isfinite(own), blend, par)
            if sc in nd.columns:
                own = nd[sc].to_numpy(dtype="float64")
                par = par_scale[m]
                blend = np.where(np.isfinite(par), w * own + (1.0 - w) * par, own)
                # A NaN own scale means *degenerate* — every contributing slice
                # identical — and must NOT be rescued by the parent, or the
                # absolute gate becomes the hair trigger described at
                # _DEGENERATE_SCALE (measured: 46 % of ordinary slices flagged).
                #
                # But that verdict needs evidence.  Below min_slices the group
                # has no standing to make any claim of its own — that is what
                # w = 0 means — so a single degenerate slice must not condemn
                # the whole group to an unflaggable NaN scale.  It inherits the
                # parent instead, which is what the pre-ladder code did by
                # dropping the group entirely.
                nd[sc] = np.where(
                    np.isfinite(own), blend, np.where(supported, np.nan, par)
                )

        parent_frame = nd.copy()
        parent_keys = keys_d
        parent_depth = depth
        emitted = nd.copy()
        emitted["__null_depth__"] = np.int16(depth)
        # Carry the composite key as a STRING computed here, while `nd` still
        # has the source dtypes.  Concatenating ladder depths widens an integer
        # key column to float (the shallower depths have no value for it), so
        # rebuilding the key from the concatenated frame would produce "3.0"
        # where the data frame produces "3" and nothing would ever match.
        emitted["__null_key__"] = _null_key_array(nd, keys_d)
        ladder.append(emitted)

    # Only metrics that actually contributed a per-slice statistic can appear in
    # the global row.  Indexing every requested metric here raised a KeyError
    # whenever a column was present on the frame but never reached min_slice_n
    # finite values in any slice — which is exactly what happens when a legacy
    # scored frame lacking `neighbourhood_offset` data is passed in.
    global_row = {"__is_global__": True, "n_slices": int(len(slice_stats))}
    for m in contributing:
        global_row[f"{m}_null_loc"] = _g_loc[m]
        global_row[f"{m}_null_scale"] = _g_scale[m]
    for k in null_keys:
        global_row[k] = None
    global_row["__null_depth__"] = np.int16(0)
    global_row["__null_key__"] = None
    global_row["__eff_depth__"] = 0.0
    # Record WHICH key list built this ladder.  `add_absolute_scores` slices
    # its own `null_keys` by depth, so a caller that passes a different list —
    # reordered, or built against a different frame — would index the wrong
    # prefix, match nothing, and silently calibrate everything against a
    # coarser null while the run completed normally.  Provenance turns that
    # into an error.
    global_row["__null_keys__"] = _NULL_KEY_SEP.join(map(str, null_keys))

    if ladder:
        null_df = pd.concat(ladder, ignore_index=True)
    else:
        null_df = pd.DataFrame(
            columns=[*null_keys, "n_slices", "__null_depth__", "__null_key__"]
        )
    null_df["__is_global__"] = False
    # Marks "this group has its own null" so add_absolute_scores can tell a
    # group that is ABSENT from every depth (borrow the global null) from one
    # that is PRESENT but degenerate (must not be rescued).
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

    # `null_ref.get(col, False) == True` collapses to the scalar False when the
    # column is missing, and `null_ref[False]` then raises KeyError instead of
    # the intended message.  Test membership explicitly.
    if "__is_global__" not in null_ref.columns:
        raise ValueError(
            "null_ref has no __is_global__ column — it was not produced by "
            "compute_null_reference"
        )
    _is_glob = null_ref["__is_global__"].fillna(False).astype(bool).to_numpy()
    global_rows = null_ref[_is_glob]
    if global_rows.empty:
        raise ValueError("null_ref has no global fallback row")
    global_row = global_rows.iloc[0]

    per_group = null_ref[~_is_glob]

    loc_cols = [f"{m}_null_loc" for m in metric_cols]
    scale_cols = [f"{m}_null_scale" for m in metric_cols]
    join_cols = [c for c in (loc_cols + scale_cols) if c in null_ref.columns]

    # ------------------------------------------------------------------
    # Deepest-first lookup down the null ladder.
    #
    # `null_keys` is a nesting (coarsest first).  A row is calibrated against
    # the FINEST null group that exists for it: (label, region, res) if that
    # group was estimated, else (label, res), else (label,), else the flat
    # global null.  A plain merge on the full key could only do the first and
    # the last, which is what sent rare classes in fine H3 cells — most of
    # CROPTYPE24 — straight to a resolution-blind global null.
    # ------------------------------------------------------------------
    _keys = list(null_keys)

    # Provenance check.  The ladder's depth d was built on `built_keys[:d]`, so
    # slicing a DIFFERENT list by depth would index the wrong prefix, match
    # nothing, and calibrate everything against a coarser null — while the run
    # completed and still produced flags.  Refuse instead.
    _built = None
    if "__null_keys__" in null_ref.columns:
        _bs = null_ref["__null_keys__"].dropna()
        if len(_bs):
            _built = [k for k in str(_bs.iloc[0]).split(_NULL_KEY_SEP) if k]
    if _built is not None and _keys[: len(_built)] != _built:
        raise ValueError(
            "add_absolute_scores: null_ref was built on null_keys="
            f"{_built} but {_keys} was passed. The ladder is indexed by "
            "prefix depth, so these must agree (the passed list may only "
            "extend the built one). Rebuild the reference, or pass the same "
            "keys."
        )

    resolved = {c: np.full(len(out), np.nan, dtype="float64") for c in join_cols}
    depth_used = np.full(len(out), -1.0, dtype="float64")
    has_own = np.zeros(len(out), dtype=bool)

    if not per_group.empty and all(k in out.columns for k in _keys):
        if "__null_depth__" in per_group.columns:
            depths = sorted(
                {int(d) for d in per_group["__null_depth__"].dropna().tolist()},
                reverse=True,
            )
        else:
            # A reference predating the ladder: a single depth on the full key.
            # Honoured, but the key is rebuilt from the frame's own columns, and
            # a dtype that shifted between building and use ("3" vs "3.0") makes
            # that silently miss — hence the coverage warning below.
            print(
                "[calibration] NOTE: null_ref predates the null ladder "
                "(no __null_key__); matching on the full key only."
            )
            depths = [len(_keys)]
        pending = np.ones(len(out), dtype=bool)
        for d in depths:
            d = int(d)
            if d <= 0 or d > len(_keys) or not pending.any():
                continue
            keys_d = _keys[:d]
            if "__null_depth__" in per_group.columns:
                sub = per_group[per_group["__null_depth__"].astype("int64") == d]
            else:
                sub = per_group
            if sub.empty:
                continue
            lut = sub.copy()
            if "__null_key__" in lut.columns:
                lut.index = pd.Index(lut["__null_key__"].to_numpy(dtype=object))
            else:  # legacy reference without a stored key
                lut.index = pd.Index(_null_key_array(lut, keys_d))
            lut = lut[~lut.index.duplicated(keep="first")]
            avail = [c for c in join_cols if c in lut.columns]
            if not avail:
                continue
            # One hashed reindex over the whole block, rather than an `isin`
            # pass plus one reindex per join column: `row_key` used to be hashed
            # six to eight times per depth.
            row_key = pd.Index(_null_key_array(out, keys_d))
            take = [
                c for c in ("__eff_depth__", "__null_depth__") if c in lut.columns
            ]
            blk = lut[avail + take].reindex(row_key)
            # Membership must come from a column that is non-null for EVERY
            # group row, not from the values: a group whose loc and scale are
            # both NaN is still a match, and treating it as a miss would let it
            # fall through to the parent — quietly undoing the degenerate-scale
            # guard for it.  `__null_depth__` is that marker; only a reference
            # predating the ladder lacks it, and there the values are the only
            # signal available.
            if "__null_depth__" in take:
                found = np.asarray(blk["__null_depth__"].notna().to_numpy())
            else:
                found = np.zeros(len(out), dtype=bool)
                for c in avail:
                    found |= np.asarray(blk[c].notna().to_numpy())
            hit = pending & found
            if not hit.any():
                continue
            pos = np.flatnonzero(hit)
            for c in avail:
                resolved[c][pos] = blk[c].to_numpy(dtype="float64")[pos]
            # Report the deepest rung that actually CONTRIBUTED, not the
            # deepest that merely exists: a group below min_slices sits at
            # w = 0 and equals its parent exactly, so crediting it with its own
            # depth would make the localisation histogram claim a locality the
            # numbers do not have.
            if take:
                _eff = blk["__eff_depth__"].to_numpy(dtype="float64")[pos]
                depth_used[pos] = np.where(np.isfinite(_eff), _eff, float(d))
            else:
                depth_used[pos] = float(d)
            has_own |= hit
            pending &= ~hit

        # A reference that exists but matches almost nothing is the signature
        # of a key mismatch (a dtype that shifted, a renamed column), and it is
        # otherwise invisible: the run completes and still flags things, just
        # against the wrong null.
        #
        # Measure it over rows that COULD have matched.  The null is built from
        # scored slices only, so unscored rows (slice below the scoring size —
        # the long tail of rare classes in a CROPTYPE24 run) are absent from
        # every group by construction.  Counting them made the warning fire at
        # 71 % on a perfectly healthy frame while asserting that sparsity was
        # not the explanation.  They carry NaN distances, so finiteness of the
        # primary metric is the mask.
        _prim = pd.to_numeric(out[metric_cols[0]], errors="coerce").to_numpy(
            dtype="float64"
        )
        _cand = np.isfinite(_prim)
        _n_cand = int(_cand.sum())
        _miss = float(np.mean(~has_own[_cand])) if _n_cand else 0.0
        if _miss > 0.5:
            print(
                f"[calibration] WARNING: {_miss:.0%} of the {_n_cand:,} "
                f"scorable rows matched NO null group on {_keys} and fell back "
                "to the flat global null. That usually means the null "
                "reference and this frame disagree on a key column (dtype or "
                "spelling) — check before trusting these scores."
            )

    for c in join_cols:
        out[c] = resolved[c]
    # Small integer, -1 for "matched no group": an int round-trips through
    # parquet predictably, a float32 carrying 3.0 / 0.0 / -1.0 does not.
    out["abs_z_null_depth"] = np.where(
        np.isfinite(depth_used), depth_used, -1.0
    ).astype(np.int16)

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
