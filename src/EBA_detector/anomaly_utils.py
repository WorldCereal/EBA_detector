"""Anomaly detection utilities — pure computation helpers, scoring, and metrics.

Extracted from anomaly.py to improve maintainability.  Every public function
here is a stateless building block consumed by the orchestration layer in
``anomaly.py``.

Sections
--------
1. Constants
2. Math / distance helpers
3. Normalization & rank helpers
4. Label-domain & mapping helpers
5. Slice operations (merge, centroids)
6. Scoring (per-slice, hierarchical)
7. Context-aware metrics (centroid margins, kNN purity)
8. Confidence computation & fusion
9. Flagging / thresholding
10. Incremental update helpers (impact zone, unscored sample detection)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------

MIN_SCORING_SLICE_SIZE: int = 50
"""Slices smaller than this get zero scores (not enough data to be meaningful)."""

# The 6 anomaly columns appended to long-format parquets.
# These are the final user-facing columns after the LC10 + CTY24 pipeline runs.
ANOMALY_COLUMNS: List[str] = [
    "LC10_confidence_nonoutlier",
    "LC10_anomaly_flag",
    "outlier_LC10_cls",
    "CTY24_confidence_nonoutlier",
    "CTY24_anomaly_flag",
    "outlier_CTY24_cls",
]

# Per-domain subsets — used by the incremental update pathway to detect
# unscored samples for each domain independently.  Rows mapped to a
# skip_class (e.g. "ignore") will have NaN in one domain's columns but
# valid scores in the other; checking all 6 would incorrectly flag them.
LC10_ANOMALY_COLUMNS: List[str] = [
    "LC10_confidence_nonoutlier",
    "LC10_anomaly_flag",
    "outlier_LC10_cls",
]

CTY24_ANOMALY_COLUMNS: List[str] = [
    "CTY24_confidence_nonoutlier",
    "CTY24_anomaly_flag",
    "outlier_CTY24_cls",
]

_SCORE_COLS: List[str] = [
    "cosine_distance",
    "knn_distance",
    "neighbourhood_offset",
    "cos_norm",
    "knn_norm",
    "S",
    "rank_percentile",
    "cos_rank",
    "knn_rank",
    "S_rank",
    "S_rank_min",
    "cos_z",
    "knn_z",
    "S_z",
    "mean_score",
]

_EXCEL_SUFFIXES = {
    ".xls",
    ".xlsx",
    ".xlsm",
    ".xlsb",
    ".ods",
}

# ---------------------------------------------------------------------------
# 2. Math / distance helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (0 when either is zero-norm)."""
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0.0 or b_norm == 0.0:
        return 0.0
    return float(np.dot(a, b) / (a_norm * b_norm))


def _cosine_distance_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Full NxN cosine-distance matrix (1 – cosine-similarity)."""
    normed = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
    sim = normed @ normed.T
    return 1.0 - sim


def robust_centroid(
    embeddings: np.ndarray,
    mode: str = "trimmed",
    trim_frac: float = 0.05,
    n_iter: int = 2,
    normalize: bool = True,
) -> np.ndarray:
    """Compute a centroid that resists contamination by the very outliers we
    are trying to detect.

    .. note:: **Spherical mean.**  Every distance in this package is a *cosine*
       distance, but the historical implementation averaged the **raw**
       vectors, so the reference point was pulled toward whichever samples
       happened to have the largest L2 norm rather than toward the angular
       centre of the cloud.  With ``normalize=True`` (the default) the
       embeddings are L2-normalised before averaging, which is the correct
       centroid for cosine geometry and matches what
       :func:`EBA_detector.robust_extensions.compute_slice_trust` already did.
       Pass ``normalize=False`` for the legacy Euclidean behaviour.

    The plain mean is *masked* by outliers: with a 10 % contamination rate the
    mean is pulled toward the anomalous mass, which deflates the cosine
    distance of those points and hides them (the classic *masking* problem in
    outlier detection).  This helper offers contamination-resistant
    alternatives.

    Parameters
    ----------
    embeddings
        ``(N, D)`` array of slice embeddings.
    mode
        - ``"mean"``    : plain mean (legacy behaviour, no robustification).
        - ``"median"``  : per-dimension median (very robust, cheap).
        - ``"trimmed"`` : iterative trimmed mean.  Compute the mean, measure
          cosine distance of every point to it, drop the farthest
          ``trim_frac`` of points, and recompute the mean on the retained
          (inlier) set.  Repeated *n_iter* times.  This is the recommended
          default: it removes the masking effect while staying close to the
          dense inlier core.
    trim_frac
        Fraction of farthest points discarded per iteration in ``"trimmed"``
        mode.  Should be >= the maximum plausible outlier fraction.
    n_iter
        Number of trimming iterations (``"trimmed"`` mode only).

    Returns
    -------
    np.ndarray
        ``(D,)`` centroid vector (float32).
    """
    X = np.asarray(embeddings, dtype=np.float32)
    n = X.shape[0]
    if n == 0:
        raise ValueError("Cannot compute a centroid of an empty slice")

    if mode not in {"mean", "median", "trimmed"}:
        raise ValueError("centroid mode must be one of {'mean','median','trimmed'}")

    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    # Below n == 3 a trimmed estimate has no room to trim; fall back to the
    # plain mean.  (The previous cut-off of 5 silently disabled robustification
    # on slices small enough that a single contaminant dominates, which is
    # exactly where it was needed most.)
    if mode == "mean" or n < 3:
        return X.mean(axis=0).astype(np.float32, copy=False)

    if mode == "median":
        return np.median(X, axis=0).astype(np.float32, copy=False)

    trim_frac = float(np.clip(trim_frac, 0.0, 0.49))
    centroid = X.mean(axis=0)
    keep_n = min(max(int(np.ceil((1.0 - trim_frac) * n)), 3), n)
    for _ in range(max(int(n_iter), 1)):
        c_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        dist = 1.0 - (Xn @ c_norm)
        # indices of the closest keep_n points (the inlier core)
        keep_idx = np.argpartition(dist, keep_n - 1)[:keep_n]
        new_centroid = X[keep_idx].mean(axis=0)
        if np.allclose(new_centroid, centroid, atol=1e-7):
            centroid = new_centroid
            break
        centroid = new_centroid
    return centroid.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# 3. Normalization & rank helpers
# ---------------------------------------------------------------------------


def _normalize_percentile_minmax(
    metric: np.ndarray, norm_percentiles: Tuple[float, float] = (5.0, 95.0)
) -> np.ndarray:
    """Robust-ish min-max normalization using slice percentiles; output clipped to [0,1]."""
    lo, hi = norm_percentiles
    p_lo, p_hi = np.percentile(metric, [lo, hi])
    denom = p_hi - p_lo if p_hi > p_lo else 1.0
    return np.clip((metric - p_lo) / denom, 0.0, 1.0)


def _rank_pct(metric: np.ndarray) -> np.ndarray:
    """Rank-percentile in [0,1].  Higher metric => higher rank."""
    if metric.size == 0:
        return metric
    return pd.Series(metric).rank(pct=True, method="max").to_numpy(dtype=np.float32)


def _robust_z(metric: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Robust z-score using median/MAD (not scaled)."""
    if metric.size == 0:
        return metric
    med = np.median(metric)
    mad = np.median(np.abs(metric - med))
    denom = mad if mad > 0 else 1.0
    return (metric - med) / (denom + eps)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable-ish sigmoid for moderate x ranges."""
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------------------
# 4. Label-domain & mapping helpers
# ---------------------------------------------------------------------------


def _as_label_levels(label_domain: Union[str, Sequence[str]]) -> List[str]:
    """Normalize *label_domain* into an ordered list of label columns (fine -> coarse)."""
    if isinstance(label_domain, (list, tuple)):
        return [str(x) for x in label_domain]
    return [str(label_domain)]


def _require_label_columns(df: pd.DataFrame, label_cols: Sequence[str]) -> None:
    """Raise if any of the requested label columns are missing from *df*."""
    missing = [c for c in label_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Requested label column(s) not found after mapping: {missing}")


def _load_mapping_df(
    mapping_file: Union[str, dict],
    *,
    label_cols: Sequence[str],
    class_mappings_name: str,
) -> pd.DataFrame:
    """Load a mapping into a DataFrame with columns: ``ewoc_code`` + *label_cols*.

    *mapping_file* can be:

    * A **dict** (in-memory CLASS_MAPPINGS, e.g. from SharePoint) — processed
      directly without any file I/O.
    * A **file path string** pointing to an Excel or JSON file (legacy).

    JSON / dict formats supported
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    1) ``{"LANDCOVER10": {"110...": "temporary_crops", ...}, "CROPTYPE25": {...}}``
       — uses *class_mappings_name* to select the inner mapping.
    2) ``{"110...": "label", ...}``  — single label column.
    3) ``{"110...": {"lvl0": "...", "lvl1": "..."}, ...}``  — hierarchical.
    4) ``{"110...": ["lvl0", "lvl1", ...], ...}``  — hierarchical by position.
    5) ``[{"ewoc_code": "110...", "lvl0": "...", ...}, ...]``  — table.
    """
    # ------------------------------------------------------------------
    # In-memory dict path (e.g. CLASS_MAPPINGS built from SharePoint)
    # ------------------------------------------------------------------
    if isinstance(mapping_file, dict):
        data: Union[dict, list] = mapping_file
    else:
        # ------------------------------------------------------------------
        # File path path (legacy)
        # ------------------------------------------------------------------
        p = Path(mapping_file)
        suf = p.suffix.lower()

        if suf in _EXCEL_SUFFIXES:
            return pd.read_excel(mapping_file)

        if suf != ".json":
            raise ValueError(
                f"Unsupported mapping_file type '{suf}'. "
                f"Use an Excel file ({sorted(_EXCEL_SUFFIXES)}) or a .json file."
            )

        with open(mapping_file, "r", encoding="utf-8") as f:
            data = json.load(f)

    # Table-like JSON
    if isinstance(data, list):
        return pd.DataFrame(data)

    if not isinstance(data, dict):
        raise ValueError("mapping_file JSON must be a dict or a list of records")

    # Multi-mapping JSON (e.g. class_mappings.json) — select the named mapping
    if class_mappings_name in data and isinstance(data[class_mappings_name], dict):
        data = data[class_mappings_name]

    rows: list = []
    for ewoc_code, v in data.items():
        row: dict = {"ewoc_code": ewoc_code}

        if isinstance(v, (str, int, float)) or v is None:
            if len(label_cols) != 1:
                raise ValueError(
                    "mapping_file JSON maps ewoc_code to a single value, but label_domain "
                    f"requests multiple label columns.  Expected columns: {list(label_cols)}"
                )
            row[label_cols[0]] = v

        elif isinstance(v, dict):
            for lc in label_cols:
                if lc in v:
                    row[lc] = v[lc]

        elif isinstance(v, (list, tuple)):
            if len(v) < len(label_cols):
                raise ValueError(
                    f"mapping_file JSON list for ewoc_code={ewoc_code} has {len(v)} values "
                    f"but {len(label_cols)} label columns were requested: {list(label_cols)}"
                )
            for lc, vv in zip(label_cols, v):
                row[lc] = vv

        else:
            raise ValueError(
                f"Unsupported JSON mapping value type for ewoc_code={ewoc_code}: {type(v)}"
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 5. Slice operations (merge, centroids, adaptive H3)
# ---------------------------------------------------------------------------


def assign_adaptive_h3_level(
    df: pd.DataFrame,
    h3_levels: Sequence[int],
    label_col: str = "ewoc_code",
    group_cols: Optional[Sequence[str]] = None,
    min_slice_size: int = 100,
    max_slice_size: Optional[int] = None,
) -> pd.DataFrame:
    """Assign each point an effective H3 cell based on point density.

    Iterates from **coarsest** to **finest** H3 resolution.  For each
    ``(group_cols, label, h3_cell)`` slice at the current level:

    - If the slice has **≤ max_slice_size** points (or *max_slice_size* is
      None) → resolve those points at this level, regardless of whether the
      slice is small or large.  Small slices are handled later by
      ``merge_small_slices``.
    - If the slice **exceeds max_slice_size** → leave those points unresolved
      and push them to the next finer level where the geographic cell is
      smaller and the slice will naturally shrink.
    - At the finest level all remaining unresolved points are resolved
      unconditionally (every point must end up somewhere).

    After the loop, any still-unresolved points are assigned the finest
    requested level unconditionally.

    Example with h3_levels=[1, 2, 3], min_slice_size=100, max_slice_size=4000
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    - Dense Europe cell at L1 with 12 000 points in a slice → too big,
      pushed to L2.
    - At L2 the cell splits; each sub-cell has ~1 500 points → resolved ✓
    - Sparse Africa cell at L1 with 200 points → within bounds, resolved ✓
    - Very sparse cell: only 40 points even at L3 → resolved at L3
      unconditionally (handled later by merge_small_slices as undersized).

    New columns added
    ~~~~~~~~~~~~~~~~~
    - ``effective_h3_cell`` : the H3 cell index used for this point
    - ``h3_effective_level``: the H3 resolution that was selected (int)

    Parameters
    ----------
    h3_levels
        H3 resolutions to try, e.g. ``[1, 2, 3]`` or ``[3, 2, 1]``.
        Internally always sorted **coarsest → finest** (ascending by H3
        resolution number, i.e. smallest number first).
    min_slice_size
        Minimum number of points required to resolve a slice at a given level.
        Slices below this are pushed to a finer level.
    max_slice_size
        Maximum number of points allowed in a slice at a given level.  Slices
        exceeding this are pushed to a finer level.  At the finest level this
        cap is not applied — every remaining point is resolved unconditionally.
        If None, no upper cap is applied and all slices >= min_slice_size are
        resolved at the coarsest level where they first meet the minimum.
    """
    import h3 as _h3

    # Ensure coarsest → finest ordering (lowest H3 number = coarsest)
    h3_levels = sorted(h3_levels, reverse=False)

    df = df.copy()
    group_cols = list(group_cols or [])

    # Make sure h3_l3_cell exists (source column)
    if "h3_l3_cell" not in df.columns:
        raise ValueError("DataFrame must contain an 'h3_l3_cell' column")

    # Pre-compute H3 cells at every requested level from h3_l3_cell.
    # - Levels < 3 (coarser): use cell_to_parent — h3_l3_cell is the child.
    # - Level == 3: direct copy.
    # - Levels > 3 (finer): use cell_to_children — h3_l3_cell is the parent,
    #   so each L3 row maps to ONE of its children that covers the original
    #   point.  We use h3.cell_to_center_child which gives the single child
    #   at the target resolution whose centre is closest to the L3 cell centre
    #   (deterministic, no explosion of rows).
    h3_col_map: dict[int, str] = {}
    for lvl in h3_levels:
        col = f"_h3_l{lvl}_cell"
        if lvl == 3:
            df[col] = df["h3_l3_cell"]
        elif lvl < 3:
            df[col] = df["h3_l3_cell"].apply(
                lambda h, _lvl=lvl: _h3.cell_to_parent(h, _lvl)
            )
        else:  # lvl > 3 — derive finer cell from lat/lon
            if "lat" not in df.columns or "lon" not in df.columns:
                raise ValueError(
                    f"H3 level {lvl} is finer than the cached L3 cells. "
                    "DataFrame must contain 'lat' and 'lon' columns to derive "
                    "finer H3 cells via lat/lon coordinates."
                )
            df[col] = df.apply(
                lambda row, _lvl=lvl: _h3.latlng_to_cell(row["lat"], row["lon"], _lvl),
                axis=1,
            )
        h3_col_map[lvl] = col

    # Track which rows are resolved
    resolved = np.zeros(len(df), dtype=bool)
    effective_cell = np.empty(len(df), dtype=object)
    effective_level = np.full(len(df), -1, dtype=np.int8)

    finest_level = h3_levels[-1]  # last in coarsest→finest order

    for lvl in h3_levels:
        if resolved.all():
            break

        h3_col = h3_col_map[lvl]
        unresolved_idx = np.where(~resolved)[0]
        if len(unresolved_idx) == 0:
            break

        is_finest = (lvl == finest_level)

        # Build slice keys for unresolved rows at this level
        sub = df.iloc[unresolved_idx]
        key_cols = [*group_cols, label_col, h3_col]
        counts = sub.groupby(key_cols).size()

        if is_finest:
            # At the finest level: resolve ALL remaining points unconditionally.
            # No max_slice_size cap — every point must be assigned somewhere.
            resolve_keys = set(counts.index.tolist())
        else:
            # Push a slice to a finer level only when it is TOO BIG
            # (> max_slice_size); it will split into smaller sub-cells there.
            #
            # A slice that is too SMALL (< min_slice_size) is resolved HERE at
            # the current (coarser) level — going finer would only shrink it
            # further.  ``merge_small_slices`` absorbs the remainder afterwards.
            #
            # ``min_slice_size`` is used to *report* how many slices are being
            # resolved below the target support.  It was previously accepted,
            # documented at length in both this docstring and run_pipeline's,
            # and then never referenced in the body — so tuning it appeared to
            # do nothing.  It now drives the diagnostic below.  (Slices that
            # stay undersized after merging are marked by ``merge_small_slices``
            # via ``undersized_slice``, which is what downstream consumes.)
            keep = counts <= max_slice_size if max_slice_size is not None else counts == counts
            resolve_keys = set(counts[keep].index.tolist())
            n_below_min = int((counts[keep] < int(min_slice_size)).sum())
            if n_below_min:
                print(
                    f"[adaptive_h3]   L{lvl}: {n_below_min} slices resolved below "
                    f"min_slice_size={min_slice_size} (will be merged or marked "
                    "undersized)"
                )

        if not resolve_keys:
            if not is_finest:
                n_oversized = len(counts) - len(resolve_keys)
                print(
                    f"[adaptive_h3]   L{lvl}: 0 slices resolved, "
                    f"{n_oversized} slices too big → pushing to next level"
                )
            continue

        # Mark matching unresolved rows as resolved at this level
        sub_indexed = sub.set_index(key_cols)
        match_mask = sub_indexed.index.isin(resolve_keys)
        matched_positions = unresolved_idx[match_mask]

        resolved[matched_positions] = True
        effective_cell[matched_positions] = df.iloc[matched_positions][h3_col].to_numpy()
        effective_level[matched_positions] = np.int8(lvl)

        if not is_finest:
            n_resolved = len(resolve_keys)
            n_oversized = len(counts) - n_resolved
            n_pts_resolved = int(match_mask.sum())
            n_pts_oversized = int(len(unresolved_idx) - n_pts_resolved)
            print(
                f"[adaptive_h3]   L{lvl}: {n_resolved} slices resolved "
                f"({n_pts_resolved:,} pts), "
                f"{n_oversized} slices too big ({n_pts_oversized:,} pts) → next level"
            )

    # Safety: assign any still-unresolved points to the finest level
    # (should only happen if h3_levels has a single entry)
    still_unresolved = ~resolved
    if still_unresolved.any():
        finest_col = h3_col_map[finest_level]
        effective_cell[still_unresolved] = (
            df.loc[still_unresolved, finest_col].to_numpy()
        )
        effective_level[still_unresolved] = np.int8(finest_level)

    df["effective_h3_cell"] = effective_cell
    df["h3_effective_level"] = effective_level

    # Clean up temporary columns
    for col in h3_col_map.values():
        df = df.drop(columns=[col], errors="ignore")

    # Summary stats (printed coarsest → finest)
    for lvl in h3_levels:
        n_at_lvl = int((effective_level == lvl).sum())
        print(f"[adaptive_h3] Level {lvl}: {n_at_lvl:,} points")

    return df


def merge_small_slices(
    df: pd.DataFrame,
    min_size: int = 100,
    label_col: str = "ewoc_code",
    h3_level_name: str = "h3_l3_cell",
    group_cols: Optional[Sequence[str]] = None,
    max_iterations: int = 25,
    min_improvement: float = 0.05,
    mark_undersized: bool = True,
    context_col: str = "context_h3_cell",
) -> pd.DataFrame:
    """Merge small slices with neighbouring H3 cells until they exceed *min_size*.

    A "slice" is defined by: ``group_cols + [label_col] + [h3_level_name]``.

    Context preservation (*context_col*)
    ------------------------------------
    Merging is decided **per (group, label, cell)**, so maize in cell X can be
    relocated to cell Y while wheat in cell X stays put.  That is the right
    granularity for finding enough same-class support, but it destroys the
    meaning of the *cell* as a geographic neighbourhood — and the cell is
    exactly what the context metrics (``add_alt_class_centroid_metrics``,
    ``add_knn_label_purity_for_flagged``, ``compute_slice_trust``) group by.
    After merging, "the other classes near me" became an arbitrary,
    label-dependent set, so the alt-class margin and the kNN purity were not
    measuring what their docstrings claimed.

    This function therefore snapshots the **pre-merge** cell into *context_col*
    and leaves it untouched.  Callers should key scoring on *h3_level_name*
    (post-merge, class support) and every context metric on *context_col*
    (pre-merge, genuine neighbourhood).

    Merge policy
    ------------
    * A merge is only accepted when the target actually contributes points
      (``best_count > 0``) **and** the merged slice is closer to *min_size*.
    * Targets are restricted to cells present in the data and to the same or a
      coarser resolution, so a merge never invents an empty cell.
    * Candidate ordering is deterministic (count desc, then cell id) so the
      same input always produces the same merge map.
    * ``merge_steps`` records how many hops each row was moved, so a slice
      assembled from four cells three hops away is visible downstream rather
      than looking identical to an untouched one.
    """
    import h3 as _h3

    df = df.copy()
    group_cols = list(group_cols or [])
    key_cols = [*group_cols, label_col, h3_level_name]

    # Snapshot the pre-merge cell: this is the stable geographic neighbourhood
    # that every context metric must be computed on.
    if context_col not in df.columns:
        df[context_col] = df[h3_level_name].astype(str)
    df["merge_steps"] = np.uint8(0)

    # Pre-compute H3 neighbours for all cells present.
    #
    # Resolution-aware: in *adaptive* H3 mode the ``effective_h3_cell`` column
    # mixes cells from several resolutions (dense regions resolved fine, sparse
    # regions resolved coarse).  ``grid_disk`` only returns same-resolution
    # neighbours, so a small fine-level (e.g. L3) slice sitting next to a region
    # that was resolved coarse (e.g. L2) would find *no* neighbour that exists
    # in the column and could never merge — leaving it permanently undersized.
    # To fix this we also offer, as merge candidates, the cell's parents at any
    # coarser resolution that is actually present in the column.  A small L3
    # slice can then merge into the L2 cell that geographically contains it.
    present_cells = set(str(c) for c in df[h3_level_name].unique().tolist())
    try:
        resolutions_present = sorted({_h3.get_resolution(c) for c in present_cells})
    except Exception:
        resolutions_present = []

    neighbour_map = {}
    for cell in present_cells:
        cands = set(_h3.grid_disk(cell, 1)) - {cell}  # same-resolution ring
        try:
            r = _h3.get_resolution(cell)
        except Exception:
            r = None
        if r is not None:
            for rr in resolutions_present:
                if rr < r:  # only coarser parents
                    try:
                        cands.add(_h3.cell_to_parent(cell, rr))
                    except Exception:
                        pass
        # Restrict to candidates that actually exist as effective cells so we
        # never create a brand-new (empty) target slice.
        neighbour_map[cell] = [c for c in cands if c in present_cells]

    # Iterative bulk merge
    counts = df.groupby(key_cols).size()
    for _ in range(max_iterations):
        small = counts[counts < min_size]
        if small.empty:
            break

        before_total_small = int(small.sum())

        # Build candidate merges: each small key with its best neighbour
        merge_rows: List[Tuple] = []
        for key, _ in small.items():
            if not isinstance(key, tuple):
                key = (key,)
            group_vals = key[:-2]
            label_value = key[-2]
            cell = key[-1]

            neighbours = neighbour_map.get(cell, [])
            if not neighbours:
                continue

            # Deterministic candidate ordering: most same-label support first,
            # ties broken by cell id.  Without the id tie-break the chosen
            # target depended on dict/set iteration order, so two runs over the
            # same data could build different slices and emit different flags.
            scored_neighbours = sorted(
                (
                    (int(counts.get((*group_vals, label_value, nb), 0)), str(nb))
                    for nb in neighbours
                ),
                key=lambda t: (-t[0], t[1]),
            )
            best_count, best_target = (
                scored_neighbours[0] if scored_neighbours else (0, None)
            )

            if best_target is not None and best_count > 0:
                merge_rows.append((*group_vals, label_value, cell, best_target))

        if not merge_rows:
            break

        merge_df = pd.DataFrame(
            merge_rows, columns=[*group_cols, label_col, h3_level_name, "target_cell"]
        )

        df = df.merge(merge_df, on=key_cols, how="left")
        mask = df["target_cell"].notna()
        if mask.any():
            df.loc[mask, h3_level_name] = df.loc[mask, "target_cell"].astype(str)
            df.loc[mask, "merge_steps"] = (
                df.loc[mask, "merge_steps"].to_numpy(dtype=np.int16) + 1
            ).astype(np.uint8)
        df = df.drop(columns=["target_cell"], errors="ignore")

        # Recompute counts and check improvement
        counts = df.groupby(key_cols).size()
        small_after = counts[counts < min_size]
        after_total_small = int(small_after.sum())
        improvement = (
            (before_total_small - after_total_small) / before_total_small
            if before_total_small > 0
            else 0.0
        )
        if improvement < min_improvement:
            break

    if mark_undersized:
        final_counts = df.groupby(key_cols).size()
        undersized_keys = set(final_counts[final_counts < min_size].index)
        df["undersized_slice"] = df.set_index(key_cols).index.isin(undersized_keys)

    df["slice_id"] = df.groupby(key_cols, sort=True).ngroup().astype(np.uint32)
    return df


def compute_slice_centroids(
    df: pd.DataFrame,
    label_col: str = "ewoc_code",
    h3_level_name: str = "h3_l3_cell",
    group_cols: Optional[Sequence[str]] = None,
    centroid_mode: str = "trimmed",
    centroid_trim: float = 0.10,
) -> pd.DataFrame:
    """Compute centroids of embeddings per slice (group_cols + h3 + label).

    By default a contamination-resistant (iterative trimmed-mean) centroid is
    used so that slice outliers do not mask themselves by pulling the centroid
    toward the anomalous mass.  Pass ``centroid_mode="mean"`` for the legacy
    plain-mean behaviour.
    """
    group_cols = list(group_cols or [])
    group_keys = [*group_cols, h3_level_name, label_col]

    def _centroid(emb_list: Iterable[np.ndarray]) -> np.ndarray:
        arr = np.vstack(list(emb_list))
        return robust_centroid(arr, mode=centroid_mode, trim_frac=centroid_trim)

    centroids = (
        df.groupby(group_keys)["embedding"]
        .apply(_centroid)
        .reset_index()
        .rename(columns={"embedding": "centroid"})
    )
    return centroids


# ---------------------------------------------------------------------------
# 6. Scoring (per-slice, hierarchical)
# ---------------------------------------------------------------------------


def compute_scores_for_slice(
    df_slice: pd.DataFrame,
    centroid: Optional[np.ndarray] = None,
    norm_percentiles: Tuple[float, float] = (5.0, 95.0),
    max_full_pairwise_n: Optional[int] = None,
    force_knn: bool = False,
    knn_k: int = 10,
    centroid_mode: str = "trimmed",
    centroid_trim: float = 0.10,
) -> pd.DataFrame:
    """Compute anomaly scores for a single slice×class dataframe.

    This function auto-selects the kNN computation strategy:
      - Full pairwise NxN distance matrix if feasible
      - kNN-only computation (NearestNeighbors) if N is large or *force_knn=True*

    Returns columns:
      cosine_distance, knn_distance, cos_norm, knn_norm, S, rank_percentile,
      cos_rank, knn_rank, S_rank, S_rank_min, cos_z, knn_z, S_z, mean_score
    """
    embeddings = np.vstack(df_slice["embedding"].to_numpy()).astype("float32", copy=False)
    n = embeddings.shape[0]

    if centroid is None:
        # Use a contamination-resistant centroid by default so that the very
        # outliers we are scoring do not pull the reference point toward
        # themselves (masking).  Pass centroid_mode="mean" to restore legacy
        # behaviour.
        centroid = robust_centroid(
            embeddings, mode=centroid_mode, trim_frac=centroid_trim
        )

    # Cosine distance to centroid
    cos_dist = np.array(
        [1.0 - _cosine_similarity(e, centroid) for e in embeddings], dtype=np.float32
    )

    # Choose kNN strategy: sqrt(N) capped at 50, but at least knn_k
    # k = min(int(knn_k), n - 1) if n > 1 else 0
    SQRT_N = int(np.sqrt(n))
    k = min(max(int(knn_k), min(int(SQRT_N), 50)), n - 1) if n > 1 else 0

    use_knn_only = force_knn or (max_full_pairwise_n is not None and n > max_full_pairwise_n)

    neigh_idx: Optional[np.ndarray] = None
    if k <= 0:
        knn_dist = np.zeros(n, dtype=np.float32)
    elif use_knn_only:
        # kNN-only path (memory friendly for large slices)
        nn = NearestNeighbors(
            n_neighbors=k + 1,  # include self, then drop it
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        nn.fit(embeddings)
        distances, neigh = nn.kneighbors(embeddings)
        knn_dist = distances[:, 1:].mean(axis=1).astype(np.float32, copy=False)
        neigh_idx = neigh[:, 1:]
    else:
        # Full pairwise NxN (more memory intensive)
        dist_matrix = _cosine_distance_matrix(embeddings)
        np.fill_diagonal(dist_matrix, np.inf)
        part = np.argpartition(dist_matrix, k, axis=1)[:, :k]
        knn_dist = (
            np.take_along_axis(dist_matrix, part, axis=1)
            .mean(axis=1)
            .astype(np.float32, copy=False)
        )
        neigh_idx = part

    knn_dist = np.nan_to_num(knn_dist, nan=0.0, posinf=0.0, neginf=0.0)

    # --- neighbourhood offset ------------------------------------------
    # A *small* kNN distance is normally evidence that a point is an inlier.
    # That inference breaks when the errors are coherent: a group of samples
    # mislabelled the same way (one dataset digitised against the wrong legend,
    # one region, one season) forms a dense cluster, so each member finds its
    # fellow errors as nearest neighbours and looks perfectly supported.  This
    # is the classic *masking* problem, and it is why the detector's recall
    # collapsed exactly where contamination was worst.
    #
    # ``neighbourhood_offset`` asks a different question: where does the
    # point's own neighbourhood sit relative to the class centroid?  For a
    # genuine inlier it sits on the centroid (offset ~ 0).  For a member of a
    # coherent wrong cluster it sits far away (offset large) even though its
    # kNN distance is tiny.  Downstream this stops a self-consistent error
    # cluster from vetoing the centroid evidence.
    if neigh_idx is not None and neigh_idx.size:
        emb_n = embeddings / (
            np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        )
        nbr_mean = emb_n[neigh_idx].mean(axis=1)
        nbr_mean /= np.linalg.norm(nbr_mean, axis=1, keepdims=True) + 1e-12
        c_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
        neighbourhood_offset = (1.0 - (nbr_mean @ c_norm)).astype(np.float32)
    else:
        neighbourhood_offset = np.zeros(n, dtype=np.float32)

    # Percentile-based normalization
    cos_norm = _normalize_percentile_minmax(cos_dist, norm_percentiles=norm_percentiles)
    knn_norm = _normalize_percentile_minmax(knn_dist, norm_percentiles=norm_percentiles)
    scores = 0.5 * (cos_norm + knn_norm)

    # Rank-based scores
    cos_rank = _rank_pct(cos_dist)
    knn_rank = _rank_pct(knn_dist)
    s_rank = 0.5 * (cos_rank + knn_rank)
    # rank_percentile_rank = pd.Series(s_rank).rank(pct=True, method="max").to_numpy(dtype=np.float32)

    # Robust z-score scores (median/MAD) + sigmoid squashing
    cos_z = _robust_z(cos_dist)
    knn_z = _robust_z(knn_dist)
    s_z = 0.5 * (_sigmoid(cos_z) + _sigmoid(knn_z))

    ranks = pd.Series(s_rank).rank(pct=True, method="max").to_numpy()

    # Build output — drop embedding columns to keep the result lean
    df_scored = df_slice.copy()[[c for c in df_slice.columns if "embedding" not in c]]
    df_scored["cosine_distance"] = cos_dist
    df_scored["knn_distance"] = knn_dist
    df_scored["neighbourhood_offset"] = neighbourhood_offset
    df_scored["cos_norm"] = cos_norm
    df_scored["knn_norm"] = knn_norm
    df_scored["S"] = scores
    df_scored["rank_percentile"] = ranks.astype(np.float32)

    df_scored["cos_rank"] = cos_rank
    df_scored["knn_rank"] = knn_rank
    df_scored["S_rank"] = s_rank
    df_scored["S_rank_min"] = np.minimum(cos_rank, knn_rank).astype(np.float32)
    # df_scored["rank_percentile_rank"] = rank_percentile_rank
    df_scored["cos_z"] = cos_z
    df_scored["knn_z"] = knn_z
    df_scored["S_z"] = s_z

    # Confidence score: average of the three score variants
    df_scored["mean_score"] = (
        (df_scored["S_rank"] + df_scored["S_rank_min"] + df_scored["S_z"]) / 3.0
    ).astype(np.float32)

    return df_scored


def _score_group_simple(
    g: pd.DataFrame,
    norm_percentiles: Tuple[float, float],
    max_full_pairwise_n: Optional[int],
    centroid_mode: str = "trimmed",
    centroid_trim: float = 0.10,
    min_scoring_slice_size: int = MIN_SCORING_SLICE_SIZE,
    centroid: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Score a single group; mark it *unscored* when the slice is too small.

    Slices below *min_scoring_slice_size* previously received ``0.0`` in every
    score column, which is indistinguishable downstream from "we looked and it
    is fine".  They now carry ``scored = False`` so the caller can emit an
    explicit ``unscored`` flag state instead of ``normal`` — the detector's
    blind spot becomes visible rather than being silently recorded as a clean
    bill of health.
    """
    if len(g) < int(min_scoring_slice_size):
        g = g.copy()
        for c in _SCORE_COLS:
            g[c] = np.nan
        g["scored"] = False
        return g

    out = compute_scores_for_slice(
        g,
        centroid=centroid,  # computed inside when None
        norm_percentiles=norm_percentiles,
        max_full_pairwise_n=max_full_pairwise_n,
        force_knn=False,
        knn_k=10,
        centroid_mode=centroid_mode,
        centroid_trim=centroid_trim,
    )
    out["scored"] = True
    return out


def _add_hierarchical_ref_outlier_class(
    df: pd.DataFrame,
    label_cols: Sequence[str],
    group_cols: Sequence[str],
    h3_level_name: str,
    min_slice_size: int,
    out_ref_class_col: str = "ref_outlier_class",
    out_ref_level_col: str = "ref_outlier_level",
    out_ref_group_n_col: str = "ref_group_n",
) -> pd.DataFrame:
    """Decide, per point, which label level is used for scoring.

    - Level 0 if level-0 slice size >= *min_slice_size*
    - Else first higher level with group size >= *min_slice_size*
    - Else coarsest level

    Also computes ``slice_n``, ``ref_group_n``.
    """
    df = df.copy()

    if not label_cols:
        raise ValueError("label_cols must be non-empty")

    # Level-0 slice size
    slice_keys_v0 = [*group_cols, h3_level_name, label_cols[0]]
    df["slice_n"] = (
        df.groupby(slice_keys_v0)["sample_id"]
        .transform("size")
        .astype(np.int32)
    )

    # Single-level mode: always score at level 0
    if len(label_cols) < 2:
        df[out_ref_level_col] = np.int8(0)
        df[out_ref_class_col] = df[label_cols[0]].astype(object)
        df[out_ref_group_n_col] = df["slice_n"].astype(np.int32)
        return df

    # Group sizes for higher levels
    n_cols: dict = {}
    for lc in label_cols[0:]:
        keys = [*group_cols, h3_level_name, lc]
        ncol = f"_n_{lc}"
        df[ncol] = (
            df.groupby(keys)["sample_id"]
            .transform("size")
            .astype(np.int32)
        )
        n_cols[lc] = ncol

    n = len(df)
    ref_level = np.full(n, -1, dtype=np.int8)

    # Level 0 if big enough
    big0 = df["slice_n"].to_numpy() >= int(min_slice_size)
    ref_level[big0] = 0

    # First higher level that meets threshold
    for lvl, lc in enumerate(label_cols[1:], start=1):
        ncol = n_cols[lc]
        ok = (ref_level == -1) & (df[ncol].to_numpy() >= int(min_slice_size))
        ref_level[ok] = np.int8(lvl)

    # Remaining: coarsest
    ref_level[ref_level == -1] = np.int8(len(label_cols) - 1)

    df[out_ref_level_col] = ref_level

    # ref_outlier_class and ref_group_n
    #
    # NOTE: ``copy=True`` is required, not cosmetic.  ``Series.to_numpy()``
    # returns a *view* on the underlying block for these dtypes, so the
    # in-place assignments below used to write straight through into
    # ``df[label_cols[0]]`` and ``df["slice_n"]``.  On pandas < 3 that silently
    # corrupted the level-0 label column (which ``score_slices_hierarchical``
    # then groups by, so the level-0 slices were built from mangled labels);
    # on pandas >= 3 copy-on-write makes the view read-only and the whole
    # hierarchical path raised "assignment destination is read-only".
    ref_class = df[label_cols[0]].astype(object).to_numpy(copy=True)
    ref_n = df["slice_n"].to_numpy(copy=True)

    for lvl, lc in enumerate(label_cols[1:], start=1):
        m = ref_level == lvl
        if not m.any():
            continue
        ref_class[m] = df[lc].astype(object).to_numpy()[m]
        ref_n[m] = df[n_cols[lc]].to_numpy()[m]

    df[out_ref_class_col] = ref_class
    df[out_ref_group_n_col] = ref_n.astype(np.int32)

    return df


def score_slices_hierarchical(
    df: pd.DataFrame,
    label_cols: Sequence[str],
    group_cols: Sequence[str],
    h3_level_name: str,
    min_slice_size: int,
    norm_percentiles: Tuple[float, float],
    max_full_pairwise_n: Optional[int],
    ref_level_col: str = "ref_outlier_level",
    ref_class_col: str = "ref_outlier_class",
    centroid_mode: str = "trimmed",
    centroid_trim: float = 0.10,
    min_scoring_slice_size: int = MIN_SCORING_SLICE_SIZE,
    fallback_shrinkage_k: float = 30.0,
) -> pd.DataFrame:
    """Score points by level-0 slices, falling back to coarser label levels
    for undersized slices.

    Scores are written back ONLY for the original undersized-slice points.

    Rare-class bias in the fallback (*fallback_shrinkage_k*)
    -------------------------------------------------------
    The fallback scores a point against **every** point in the cell sharing its
    *coarser* label — e.g. rye is scored against the pooled "cereals" cloud,
    which in a wheat-dominated cell is essentially the wheat cloud.  Rye is
    then far from that centroid and its nearest neighbours are wheat, so it is
    flagged for being *rare*, not for being *wrong*.  Every minority crop in
    every mixed cell inherited this bias.

    The reference centroid is now shrunk toward the point's own fine class::

        lambda   = n_fine / (n_fine + fallback_shrinkage_k)
        centroid = lambda * fine_centroid + (1 - lambda) * coarse_centroid

    With plenty of same-fine-class support the reference is the fine centroid;
    with almost none it degrades gracefully to the coarse one.  Set
    ``fallback_shrinkage_k=0`` to score against the pure fine centroid, or a
    very large value to restore the legacy pooled-coarse behaviour.

    Fallback-scored rows keep ``ref_outlier_level > 0``, which the caller uses
    to cap their escalation — a flag raised against a borrowed reference should
    never reach ``candidate``.
    """
    from tqdm import tqdm

    if df["sample_id"].duplicated().any():
        raise ValueError("sample_id must be unique for hierarchical scoring updates")

    df = df.copy()
    for c in _SCORE_COLS:
        if c not in df.columns:
            df[c] = np.nan
    if "scored" not in df.columns:
        df["scored"] = False

    # Ensure slice_n exists (size of level-0 slice)
    slice_keys_v0 = [*group_cols, h3_level_name, label_cols[0]]
    if "slice_n" not in df.columns:
        df["slice_n"] = (
            df.groupby(slice_keys_v0)["sample_id"]
            .transform("size")
            .astype(np.int32)
        )

    df_idx = df.set_index("sample_id", drop=False)

    tqdm.pandas()

    # 1) Score rows that use level 0 directly (normal path)
    direct = df_idx[df_idx[ref_level_col] == 0]
    if not direct.empty:
        g0 = direct[
            [*group_cols, h3_level_name, label_cols[0], "sample_id", "embedding"]
        ].reset_index(drop=True)

        # Iterate groups explicitly rather than via groupby.apply: pandas 3
        # no longer passes the grouping columns to the callback, and the old
        # code depended on that (silently, behind a suppressed FutureWarning).
        scored_parts = []
        for _key, g in tqdm(
            list(g0.groupby([*group_cols, h3_level_name, label_cols[0]], sort=True)),
            desc="Scoring level-0 slices",
        ):
            scored_parts.append(
                _score_group_simple(
                    g, norm_percentiles, max_full_pairwise_n,
                    centroid_mode=centroid_mode, centroid_trim=centroid_trim,
                    min_scoring_slice_size=min_scoring_slice_size,
                )
            )
        scored0 = pd.concat(scored_parts, ignore_index=True)
        scored0 = scored0.set_index("sample_id", drop=False)
        df_idx.loc[scored0.index, _SCORE_COLS] = scored0[_SCORE_COLS].to_numpy()
        df_idx.loc[scored0.index, "scored"] = scored0["scored"].to_numpy()

    # 2) Score fallback groups once, then write back only to target rows
    fallback = df_idx[df_idx[ref_level_col] > 0]
    if not fallback.empty:
        # NOTE: label_cols[0] is part of the key on purpose.  Without it, every
        # fine class in the cell that fell back to the same coarse class landed
        # in ONE group, so `target_set` was their union: the shrinkage centroid
        # was a blend of all of them and `n_fine` was inflated.  A cell with 10
        # rye + 10 barley + 500 wheat gave rye a rye/barley blended reference and
        # lambda = 20/(20+k) instead of 10/(10+k) — reintroducing the very
        # rare-class bias the shrinkage exists to remove.  The reference cloud is
        # still the coarse class; only the shrinkage target is now per fine class.
        fb_keys = [ref_level_col, *group_cols, h3_level_name, ref_class_col,
                   label_cols[0]]
        target_map = fallback.groupby(fb_keys)["sample_id"].apply(list)

        for key, target_ids in tqdm(
            target_map.items(), total=len(target_map), desc="Scoring fallback ref groups"
        ):
            ref_level = int(key[0])
            ref_class = key[-2]   # coarse class (key[-1] is now the fine class)
            ref_label_col = label_cols[ref_level]

            # Build reference set mask on the FULL dataframe
            m = df_idx[ref_label_col].astype(object).to_numpy() == ref_class

            offset = 1
            for i, gc in enumerate(group_cols):
                m &= df_idx[gc].astype(object).to_numpy() == key[offset + i]

            h3_val = key[offset + len(group_cols)]
            m &= df_idx[h3_level_name].astype(object).to_numpy() == h3_val

            ref_df = df_idx.loc[m, ["sample_id", "embedding"]].reset_index(drop=True)
            if ref_df.empty:
                continue

            # --- shrinkage reference centroid --------------------------------
            # Score the fallback rows against a centroid that leans on their
            # own fine class as far as its support allows, instead of the
            # pooled coarse cloud (which systematically penalises rare crops).
            target_set = set(target_ids)
            fine_mask = ref_df["sample_id"].isin(target_set).to_numpy()
            coarse_emb = np.vstack(ref_df["embedding"].to_numpy()).astype(np.float32)
            coarse_centroid = robust_centroid(
                coarse_emb, mode=centroid_mode, trim_frac=centroid_trim
            )
            n_fine = int(fine_mask.sum())
            if n_fine >= 2 and fallback_shrinkage_k >= 0:
                fine_centroid = robust_centroid(
                    coarse_emb[fine_mask], mode=centroid_mode, trim_frac=centroid_trim
                )
                lam = n_fine / (n_fine + float(fallback_shrinkage_k)) if (
                    n_fine + fallback_shrinkage_k
                ) > 0 else 1.0
                centroid = lam * fine_centroid + (1.0 - lam) * coarse_centroid
                centroid = (centroid / (np.linalg.norm(centroid) + 1e-12)).astype(
                    np.float32
                )
            else:
                centroid = coarse_centroid

            scored_ref = _score_group_simple(
                ref_df, norm_percentiles, max_full_pairwise_n,
                centroid_mode=centroid_mode, centroid_trim=centroid_trim,
                min_scoring_slice_size=min_scoring_slice_size,
                centroid=centroid,
            )
            scored_ref = scored_ref[scored_ref["sample_id"].isin(target_set)]
            if scored_ref.empty:
                continue

            scored_ref = scored_ref.set_index("sample_id", drop=False)
            df_idx.loc[scored_ref.index, _SCORE_COLS] = scored_ref[_SCORE_COLS].to_numpy()
            df_idx.loc[scored_ref.index, "scored"] = scored_ref["scored"].to_numpy()

    # Rows in slices too small to score legitimately carry NaN and are marked
    # scored=False.  Anything else with a NaN score is a genuine bug.
    scored_mask = df_idx["scored"].fillna(False).to_numpy(dtype=bool)
    bad_mask = scored_mask & df_idx[_SCORE_COLS].isna().any(axis=1).to_numpy()
    if bad_mask.any():
        bad = df_idx.loc[bad_mask, ["sample_id", ref_level_col, ref_class_col]].head(20)
        raise ValueError(
            f"Hierarchical scoring left NaNs in score columns. Example rows:\n{bad}"
        )

    return df_idx.reset_index(drop=True)


# ---------------------------------------------------------------------------
# 7. Context-aware metrics (centroid margins, kNN purity)
# ---------------------------------------------------------------------------


def add_alt_class_centroid_metrics(
    df: pd.DataFrame,
    *,
    label_col: str,
    context_cols: Sequence[str],
    embedding_col: str = "embedding",
) -> pd.DataFrame:
    """For each context group (*context_cols*), compute per-label centroids and
    for each point:

    - ``self_centroid_dist_ctx`` : cosine dist to centroid of its own label
    - ``alt_label_ctx``         : closest other label centroid
    - ``alt_centroid_dist_ctx`` : cosine dist to closest other label centroid
    - ``alt_margin_ctx``        : alt – self  (≤0 suggests confusion)
    - ``context_n_labels``      : number of labels present in context
    """
    df = df.copy()

    out_alt_label = np.full(len(df), None, dtype=object)
    out_self = np.full(len(df), np.nan, dtype=np.float32)
    out_alt = np.full(len(df), np.nan, dtype=np.float32)
    out_margin = np.full(len(df), np.nan, dtype=np.float32)
    out_nlab = np.zeros(len(df), dtype=np.uint16)

    # Positional index for writing back into flat arrays
    pos = np.arange(len(df), dtype=np.int64)
    df = df.copy()
    df["_pos"] = pos

    # Pre-extract embeddings for fast vstack inside groups
    emb_series = df[embedding_col].to_numpy()

    for _, g in df.groupby(list(context_cols), dropna=False, sort=False):
        idx = g["_pos"].to_numpy()
        labels = g[label_col].to_numpy()

        uniq = pd.unique(labels)
        out_nlab[idx] = len(uniq)
        if len(uniq) < 2:
            continue

        X = np.vstack(emb_series[idx]).astype(np.float32, copy=False)
        Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

        # Centroids per label
        centroids = []
        cent_labels = []
        for lab in uniq:
            m = labels == lab
            C = Xn[m].mean(axis=0)
            C = C / (np.linalg.norm(C) + 1e-12)
            centroids.append(C)
            cent_labels.append(lab)

        C = np.vstack(centroids).astype(np.float32, copy=False)  # (L, D)
        sims = Xn @ C.T                                          # (N, L)
        dists = 1.0 - sims                                       # cosine distances

        lab_to_j = {lab: j for j, lab in enumerate(cent_labels)}
        own_j = np.array([lab_to_j[lab] for lab in labels], dtype=np.int32)

        self_dist = dists[np.arange(len(idx)), own_j]

        # Mask own label to find nearest OTHER centroid
        dists_other = dists.copy()
        dists_other[np.arange(len(idx)), own_j] = np.inf
        alt_j = np.argmin(dists_other, axis=1)
        alt_dist = dists[np.arange(len(idx)), alt_j]
        alt_lab = np.array([cent_labels[j] for j in alt_j], dtype=object)

        out_self[idx] = self_dist.astype(np.float32, copy=False)
        out_alt[idx] = alt_dist.astype(np.float32, copy=False)
        out_alt_label[idx] = alt_lab
        out_margin[idx] = (alt_dist - self_dist).astype(np.float32, copy=False)

    df["context_n_labels"] = out_nlab
    df["self_centroid_dist_ctx"] = out_self
    df["alt_label_ctx"] = out_alt_label
    df["alt_centroid_dist_ctx"] = out_alt
    df["alt_margin_ctx"] = out_margin
    df = df.drop(columns=["_pos"], errors="ignore")
    return df


def add_knn_label_purity_for_flagged(
    df_all: pd.DataFrame,
    flagged_df: pd.DataFrame,
    *,
    label_col: str,
    context_cols: Sequence[str],
    embedding_col: str = "embedding",
    purity_knn_k: int = 10,
    cap_sqrt_k: int = 50,
) -> pd.DataFrame:
    """Compute kNN label-purity within each context group, but only for
    rows where ``flagged == True``.

    Uses *df_all* for embeddings and full neighbourhood; writes results back
    into *flagged_df*.
    """
    # Flagged subset keys — limit work to contexts that contain flagged points
    flagged_only = flagged_df.loc[
        flagged_df["flagged"] == True,  # noqa: E712
        ["sample_id", *context_cols, label_col],
    ]
    # Use all rows in flagged_df, regardless of flagged status
    # flagged_only = flagged_df[["sample_id", *context_cols, label_col]]
    if flagged_only.empty:
        flagged_df["knn_same_label_frac_ctx"] = np.nan
        flagged_df["knn_majority_label_ctx"] = None
        flagged_df["knn_majority_frac_ctx"] = np.nan
        return flagged_df

    # Restrict df_all to only relevant contexts
    ctx_keys = flagged_only[context_cols].drop_duplicates()
    df_sub = df_all.merge(ctx_keys, on=list(context_cols), how="inner")

    # Prepare outputs keyed by sample_id
    out_same: dict = {}
    out_maj_lab: dict = {}
    out_maj_frac: dict = {}

    flagged_ids = set(flagged_only["sample_id"].tolist())

    for _, g in df_sub.groupby(list(context_cols), dropna=False, sort=False):
        n = len(g)
        if n < 2:
            continue

        # k similar to scoring logic (sqrt(n) capped, but at least purity_knn_k)
        k = min(max(int(purity_knn_k), min(int(np.sqrt(n)), int(cap_sqrt_k))), n - 1)
        if k <= 0:
            continue

        sids = g["sample_id"].to_numpy()
        labels = g[label_col].to_numpy()

        flagged_mask = np.array([sid in flagged_ids for sid in sids], dtype=bool)
        if not flagged_mask.any():
            continue

        X = np.vstack(g[embedding_col].to_numpy()).astype(np.float32, copy=False)

        nn = NearestNeighbors(
            n_neighbors=k + 1,
            metric="cosine",
            algorithm="brute",
            n_jobs=-1,
        )
        nn.fit(X)

        # Robustly drop self-neighbour if present
        q_idx = np.where(flagged_mask)[0]
        distances, neigh = nn.kneighbors(X[q_idx], return_distance=True)

        rows = []
        for r, qi in enumerate(q_idx):
            nn_ids = neigh[r]
            nn_ids = nn_ids[nn_ids != qi]
            rows.append(nn_ids[:k])
        neigh = np.vstack(rows)

        neigh_labels = labels[neigh]  # (n_flagged, k)

        for row_i, qi in enumerate(q_idx):
            sid = sids[qi]
            own = labels[qi]
            nl = neigh_labels[row_i]

            same_frac = float(np.mean(nl == own))

            vals, counts = np.unique(nl, return_counts=True)
            j = int(np.argmax(counts))
            maj_lab = vals[j]
            maj_frac = float(counts[j] / len(nl))

            out_same[sid] = same_frac
            out_maj_lab[sid] = maj_lab
            out_maj_frac[sid] = maj_frac

    # Write back
    flagged_df = flagged_df.copy()
    flagged_df["knn_same_label_frac_ctx"] = (
        flagged_df["sample_id"].map(out_same).astype("float32")
    )
    flagged_df["knn_majority_label_ctx"] = flagged_df["sample_id"].map(out_maj_lab)
    flagged_df["knn_majority_frac_ctx"] = (
        flagged_df["sample_id"].map(out_maj_frac).astype("float32")
    )
    return flagged_df


# ---------------------------------------------------------------------------
# 8. Confidence computation & fusion
# ---------------------------------------------------------------------------


def add_confidence_from_score(
    df: pd.DataFrame,
    score_col: str = "mean_score",
    out_col: str = "confidence",
    t: float = 0.975,        # knee: confidence starts dropping after this
    alpha: float = 0.3,      # tail sharpness (bigger => harsher near 1)
    conf_min: float = 0.01,  # never go below this
    eps: float = 1e-9,       # numerical stability near 1
    flagged_col: Optional[str] = None,
) -> pd.DataFrame:
    """Accelerating confidence drop as score → 1, with hard floor *conf_min*.

    .. math::

        y = \\text{clip}((x - t) / (1 - t), 0, 1)

        \\text{conf\\_raw} = \\exp(-\\alpha \\cdot y / (1 - y + \\varepsilon))

        \\text{confidence} = \\text{conf\\_min} + (1 - \\text{conf\\_min}) \\cdot \\text{conf\\_raw}

    - ``x <= t``  ⇒  confidence = 1
    - ``x → 1``   ⇒  confidence → conf_min (not 0)

    Flag gating (``flagged_col``)
    -----------------------------
    The continuous *score_col* (``mean_score``) is built from **within-slice
    rank** statistics, so the top-ranked sample of *every* slice receives a
    high score — even a perfectly clean slice.  Feeding that directly into the
    confidence curve manufactures low confidence (and thus down-weighting) for
    the relatively-highest sample of clean slices, and lets the continuous
    confidence disagree with the discrete ``anomaly_flag``.

    When *flagged_col* is provided, confidence is **clamped to 1.0 for any
    sample that was not flagged** by :func:`flag_anomalies`.  The decay then
    only modulates the *severity* of samples that the slice-level threshold
    already identified as anomalous.  This makes ``confidence_nonoutlier`` and
    ``anomaly_flag`` mutually consistent and removes the clean-slice penalty —
    pass ``flagged_col=None`` to restore the legacy ungated behaviour.
    """
    x = pd.to_numeric(df[score_col], errors="coerce").astype("float64")
    x = x.clip(lower=0.0, upper=1.0).to_numpy()

    if not (0.0 < t < 1.0):
        raise ValueError("t must be in (0, 1)")
    if not (alpha > 0.0):
        raise ValueError("alpha must be > 0")
    if not (0.0 < conf_min < 1.0):
        raise ValueError("conf_min must be in (0, 1)")
    if not (eps > 0.0):
        raise ValueError("eps must be > 0")

    y = (x - t) / max(1e-12, (1.0 - t))
    y = np.clip(y, 0.0, 1.0)

    conf_raw = np.exp(-alpha * (y / (1.0 - y + eps)))
    conf = conf_min + (1.0 - conf_min) * conf_raw

    if flagged_col is not None and flagged_col in df.columns:
        not_flagged = ~df[flagged_col].fillna(False).to_numpy(dtype=bool)
        conf[not_flagged] = 1.0

    conf = np.clip(conf, conf_min, 1.0).astype(np.float32)

    df[out_col] = conf
    return df


def add_flagged_robust_confidence(
    df: pd.DataFrame,
    score_col: str = "mean_score",
    flagged_col: str = "flagged",
    out_z_col: str = "z_mad",
    out_conf_col: str = "confidence",
    # mapping params
    z_knee: float = 3.0,
    eps_conf: float = 1e-3,
    z_extreme: float = 10.0,
    clip_exp: float = 50.0,
    default_unflagged_conf: float = 1.0,
) -> pd.DataFrame:
    """For each slice (caller passes one slice at a time), compute robust
    MAD-z from *score_col* and assign confidence:

    - if not flagged → ``default_unflagged_conf``
    - if flagged → ``1 / (1 + exp(k*(z - z_knee)))``

    Also writes ``z_mad`` for debugging/auditing.
    """
    out = df.copy()

    x = pd.to_numeric(out[score_col], errors="coerce").astype("float64")
    med = float(np.nanmedian(x.to_numpy()))
    abs_dev = np.abs(x - med)
    mad = float(np.nanmedian(abs_dev.to_numpy()))
    denom = mad if (np.isfinite(mad) and mad > 0.0) else 1.0

    z = (x - med) / denom
    z = z.clip(lower=0.0)  # only penalize high-side outliers; keep non-outliers at z=0

    out[out_z_col] = z.astype(np.float32)

    # choose k so that confidence(z_extreme) ~= eps_conf
    # conf(z) = 1/(1+exp(k*(z - z_knee)))  -> exp(k*(z_extreme-z_knee)) = 1/eps - 1
    k = float(np.log(1.0 / eps_conf - 1.0) / max(1e-6, (z_extreme - z_knee)))

    z_arg = np.clip(k * (z.to_numpy() - z_knee), -clip_exp, clip_exp)
    conf_flagged = 1.0 / (1.0 + np.exp(z_arg))

    flagged = out[flagged_col].fillna(False).to_numpy(dtype=bool)
    conf = np.full(len(out), float(default_unflagged_conf), dtype="float64")
    conf[flagged] = conf_flagged[flagged]

    out[out_conf_col] = np.clip(conf, 0.0, 1.0).astype(np.float32)
    return out


def apply_confidence_fusion(
    df: pd.DataFrame,
    base_conf_col: str = "confidence",
    out_conf_col: str = "confidence_alt",
    # margin inputs
    margin_col: str = "alt_margin_ctx",
    self_dist_col: str = "self_centroid_dist_ctx",
    alt_dist_col: str = "alt_centroid_dist_ctx",
    # purity input
    purity_col: str = "knn_same_label_frac_ctx",
    # margin penalty params
    margin_m0: float = 0.001,
    margin_a: float = 10.0,
    # purity penalty params
    purity_beta: float = 0.5,
    # behavior
    default_factor_if_nan: float = 1.0,
    clip_exp: float = 50.0,
) -> pd.DataFrame:
    """Fuse auxiliary ambiguity signals into base confidence::

        confidence_alt = confidence × p_margin × p_purity

    ``p_margin``
        Logistic of ``(alt_margin - m0)``; larger margin ⇒ clearer separation.

    ``p_purity``
        ``(knn_same_label_frac_ctx) ** beta``; lower purity ⇒ stronger penalty.

    - If margin / purity is NaN, factor defaults to 1.0 (no penalty).
    - Output is float32 in [0, 1].
    """
    if base_conf_col not in df.columns:
        raise KeyError(f"Missing base confidence column: '{base_conf_col}'")

    conf0 = pd.to_numeric(df[base_conf_col], errors="coerce").astype("float64").to_numpy()
    conf0 = np.clip(conf0, 0.0, 1.0)

    # --- Margin ----------------------------------------------------------
    if margin_col in df.columns:
        margin = pd.to_numeric(df[margin_col], errors="coerce").astype("float64").to_numpy()
    elif (self_dist_col in df.columns) and (alt_dist_col in df.columns):
        self_d = pd.to_numeric(df[self_dist_col], errors="coerce").astype("float64").to_numpy()
        alt_d = pd.to_numeric(df[alt_dist_col], errors="coerce").astype("float64").to_numpy()
        margin = alt_d - self_d
    else:
        margin = np.full(len(df), np.nan, dtype="float64")

    z = margin_a * (margin - margin_m0)
    z = np.clip(z, -clip_exp, clip_exp)
    p_margin = 1.0 / (1.0 + np.exp(-z))
    p_margin = np.where(np.isfinite(p_margin), p_margin, default_factor_if_nan)

    # --- Purity ----------------------------------------------------------
    if purity_col in df.columns:
        pur = pd.to_numeric(df[purity_col], errors="coerce").astype("float64").to_numpy()
        pur = np.clip(pur, 0.0, 1.0)
        p_purity = np.power(pur, purity_beta)
        p_purity = np.where(np.isfinite(p_purity), p_purity, default_factor_if_nan)
    else:
        p_purity = np.full(len(df), default_factor_if_nan, dtype="float64")

    # For unflagged rows, p_margin and p_purity remain 1.0
    if "flagged" in df.columns:
        flagged_mask = df["flagged"].fillna(False).to_numpy(dtype=bool)
        unflagged = ~flagged_mask
        p_margin = np.where(unflagged, 1.0, p_margin)
        p_purity = np.where(unflagged, 1.0, p_purity)

    # Cap minimum values to avoid too much confidence reduction
    p_margin = np.maximum(p_margin, 0.85)
    p_purity = np.maximum(p_purity, 0.85)

    # Final fusion
    conf = conf0 * p_margin * p_purity
    conf = np.clip(conf, 0.0, 1.0)

    out = df.copy()
    out[out_conf_col] = conf.astype(np.float32)
    out["p_margin"] = p_margin.astype(np.float32)
    out["p_purity"] = p_purity.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# 9. Flagging / thresholding
# ---------------------------------------------------------------------------


def flag_anomalies(
    df_scores: pd.DataFrame,
    label_col: str = "ewoc_code",
    threshold_mode: str = "percentile",
    h3_level_name: str = "h3_l3_cell",
    group_cols: Optional[Sequence[str]] = None,
    percentile_q: float = 0.96,
    mad_k: float = 3.0,
    abs_threshold: Optional[float] = None,
    fdr_alpha: float = 0.05,
    min_flagged_per_slice: Optional[int] = None,
    max_flagged_fraction: Optional[float] = None,
    flag_score_col: str = "S",
    abs_z_col: Optional[str] = "abs_z",
    abs_z_k: Optional[float] = 3.0,
    require_absolute: bool = True,
    tie_break_cols: Optional[Sequence[str]] = None,
    scored_mask_col: Optional[str] = "scored",
    local_metric_col: str = "cosine_distance",
    stable_scale_col: str = "null_scale_sigma",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Flag anomalies within each slice group.

    Slice keys: ``group_cols + [h3_level_name] + [label_col]``.

    The absolute gate (*abs_z_col*, *abs_z_k*, *require_absolute*)
    ------------------------------------------------------------
    The within-slice test alone cannot distinguish "the most unusual point in
    a clean slice" from "a genuinely mislabelled point", because
    ``flag_score_col`` is percentile-normalised **per slice** and therefore
    spans the same range everywhere.  Worse, the ``median + k·MAD`` rule
    evaluated on that bounded score is a knife-edge: on synthetic slices it
    flags 0 % at both 2 % and 30 % true contamination (at 30 % the contaminant
    inflates the slice's own median and MAD past the score ceiling) but 9 % at
    10 %.

    With ``require_absolute=True`` a point must **also** exceed *abs_z_k*
    robust sigma against the cross-slice null from
    :mod:`EBA_detector.calibration`.  That null is built from one summary
    statistic per slice, so a contaminated slice cannot calibrate its own
    errors away — which restores detection in the heavily-contaminated regime
    while removing the fixed per-slice quota that produced false positives on
    clean slices.

    Set ``require_absolute=False`` (or ``abs_z_k=None``) for the legacy
    relative-only behaviour, e.g. for ablations.

    Determinism
    -----------
    Because ``S`` is min–max normalised and clipped, roughly 2–4 % of every
    slice sits at exactly ``S == 1.0``.  Truncating that tied block with
    *max_flagged_fraction* used to select arbitrarily among ties, so the
    flagged set depended on row order and was not reproducible between runs or
    between ``rerun`` and ``update`` mode.  Ordering now falls back through
    *tie_break_cols* (default: ``abs_z`` then ``sample_id``).
    """
    group_cols = list(group_cols or [])
    group_keys = [*group_cols, h3_level_name, label_col]
    flag_col = flag_score_col
    if flag_col not in df_scores.columns:
        raise KeyError(f"flag_anomalies: missing score column {flag_col!r}")

    if threshold_mode not in {"percentile", "mad", "stable_mad", "absolute", "fdr"}:
        raise ValueError(
            "threshold_mode must be one of "
            "{'percentile','mad','stable_mad','absolute','fdr'}"
        )
    if threshold_mode == "stable_mad":
        for c in (local_metric_col, stable_scale_col):
            if c not in df_scores.columns:
                raise KeyError(
                    f"flag_anomalies: threshold_mode='stable_mad' needs {c!r}. "
                    "Run EBA_detector.calibration.add_absolute_scores first."
                )
    if threshold_mode == "absolute" and abs_threshold is None:
        raise ValueError("abs_threshold must be set when threshold_mode='absolute'")

    out = df_scores.reset_index(drop=True).copy()
    n_rows = len(out)
    flags = np.zeros(n_rows, dtype=bool)
    thr_out = np.full(n_rows, np.nan, dtype=np.float64)

    if n_rows == 0:
        out["flagged"] = flags
        out["flag_threshold"] = thr_out
        summary = pd.DataFrame(
            columns=[*group_keys, "total_samples", "flagged_samples", "flagged_fraction"]
        )
        return out, summary

    # --- ordering used for tie-breaking and for the cap ---------------------
    if tie_break_cols is None:
        tie_break_cols = [c for c in (abs_z_col, "sample_id") if c and c in out.columns]
    else:
        tie_break_cols = [c for c in tie_break_cols if c in out.columns]

    sort_cols = [flag_col, *tie_break_cols]
    ascending = [False] + [False] * len(tie_break_cols)
    # sample_id is an identifier, not a magnitude — sort it ascending so the
    # tie-break is a stable lexical rule rather than a pseudo-score.
    for i, c in enumerate(sort_cols):
        if c == "sample_id":
            ascending[i] = True
    order_positions = (
        out.sort_values(sort_cols, ascending=ascending, kind="mergesort")
        .index.to_numpy()
        .astype(np.int64)
    )
    # global_rank[i] = where row i sits in the deterministic descending order
    global_rank = np.empty(n_rows, dtype=np.int64)
    global_rank[order_positions] = np.arange(n_rows, dtype=np.int64)

    rank_in_slice = np.zeros(n_rows, dtype=np.int64)

    # --- absolute gate ------------------------------------------------------
    if require_absolute and abs_z_col and abs_z_k is not None:
        if abs_z_col not in out.columns:
            raise KeyError(
                f"flag_anomalies: require_absolute=True but {abs_z_col!r} is missing. "
                "Run EBA_detector.calibration.add_absolute_scores first, or pass "
                "require_absolute=False for legacy relative-only flagging."
            )
        abs_ok = (
            pd.to_numeric(out[abs_z_col], errors="coerce").to_numpy(dtype="float64")
            >= float(abs_z_k)
        )
        abs_ok = np.nan_to_num(abs_ok, nan=False).astype(bool)
    else:
        abs_ok = np.ones(n_rows, dtype=bool)

    # Points the pipeline never actually scored must not be flagged.
    if scored_mask_col and scored_mask_col in out.columns:
        scorable = out[scored_mask_col].fillna(False).to_numpy(dtype=bool)
    else:
        scorable = np.ones(n_rows, dtype=bool)

    values = pd.to_numeric(out[flag_col], errors="coerce").to_numpy(dtype="float64")

    if threshold_mode == "stable_mad":
        # Local reference, stable dispersion: the threshold is the slice's own
        # median distance plus k times the *cross-slice* sigma.  The plain
        # 'mad' mode instead divides by the slice's own MAD, which a
        # contaminated slice inflates along with its median — that is why the
        # legacy gate flagged 0 % at 30 % true contamination while flagging
        # 9 % at 10 %.
        local_values = pd.to_numeric(
            out[local_metric_col], errors="coerce"
        ).to_numpy(dtype="float64")
        stable_scale = pd.to_numeric(
            out[stable_scale_col], errors="coerce"
        ).to_numpy(dtype="float64")
    else:
        local_values = values
        stable_scale = None

    # --- per-slice relative test (vectorised; no groupby.apply) -------------
    #
    # The previous implementation relied on ``groupby(...).apply()`` returning
    # the grouping columns to the callback.  pandas deprecated that in 2.2 and
    # removed it in 3.0 — and the resulting FutureWarning was explicitly
    # suppressed in this function, so the breakage would have surfaced as
    # changed *results*, not as an error.  Index-based assignment has no such
    # dependency.
    indices = out.groupby(group_keys, dropna=False, sort=False).indices

    for _key, pos in indices.items():
        pos = np.asarray(pos)
        v = values[pos]
        finite = np.isfinite(v)
        if not finite.any():
            rank_in_slice[pos] = 0
            continue
        n = len(pos)
        thr = np.inf

        if threshold_mode == "percentile":
            thr = float(np.nanquantile(v, percentile_q))
        elif threshold_mode == "mad":
            med = float(np.nanmedian(v))
            mad = float(np.nanmedian(np.abs(v - med)))
            # Degenerate slice: MAD == 0 means >50% of points share one score,
            # so there is no robust scale to threshold on. Flag nothing.
            thr = med + mad_k * mad if mad > 0 else np.inf
        elif threshold_mode == "stable_mad":
            lv = local_values[pos]
            sl = stable_scale[pos]
            # An all-NaN scale means the class null was degenerate (see
            # calibration._DEGENERATE_SCALE); threshold at +inf, i.e. flag
            # nothing, rather than dividing by a floored pseudo-scale.
            med = float(np.nanmedian(lv)) if np.isfinite(lv).any() else np.nan
            sigma = float(np.nanmedian(sl)) if np.isfinite(sl).any() else np.nan
            thr = (
                med + mad_k * sigma
                if np.isfinite(med) and np.isfinite(sigma) and sigma > 0
                else np.inf
            )
        elif threshold_mode == "absolute":
            thr = float(abs_threshold)
        elif threshold_mode == "fdr":
            r = pd.Series(v).rank(ascending=False, method="max").to_numpy()
            pvals = r / (n + 1.0)
            p_sorted = np.sort(pvals)
            bh = (np.arange(1, n + 1) / n) * fdr_alpha
            passed = p_sorted <= bh
            thr = np.nan  # threshold expressed on p-values, recorded below
            if passed.any():
                p_cut = p_sorted[int(np.max(np.where(passed)))]
                sel = pvals <= p_cut
            else:
                sel = np.zeros(n, dtype=bool)

        if threshold_mode == "stable_mad":
            lv = local_values[pos]
            sel = np.where(np.isfinite(lv), lv >= thr, False)
        elif threshold_mode != "fdr":
            sel = np.where(finite, v >= thr, False)

        thr_out[pos] = thr

        # deterministic descending order inside this slice
        local_order = np.argsort(global_rank[pos], kind="mergesort")
        rank_in_slice[pos[local_order]] = np.arange(n)

        sel = sel & abs_ok[pos] & scorable[pos]
        n_flag = int(sel.sum())

        if max_flagged_fraction is not None:
            max_allowed = max(int(np.floor(max_flagged_fraction * n)), 0)
            if n_flag > max_allowed:
                keep = np.zeros(n, dtype=bool)
                if max_allowed > 0:
                    # keep the top `max_allowed` *flagged* points in the
                    # deterministic order
                    flagged_in_order = [i for i in local_order if sel[i]]
                    keep[flagged_in_order[:max_allowed]] = True
                sel = keep
                n_flag = int(sel.sum())

        if min_flagged_per_slice is not None and min_flagged_per_slice > 0:
            # Forcing a minimum manufactures flags in clean slices; only do it
            # among points that pass the absolute gate, so a clean slice can
            # still legitimately yield zero.
            if n_flag < min_flagged_per_slice:
                eligible = [
                    i for i in local_order if abs_ok[pos[i]] and scorable[pos[i]]
                ]
                sel = np.zeros(n, dtype=bool)
                sel[eligible[: min(min_flagged_per_slice, len(eligible))]] = True

        flags[pos] = sel

    out["flagged"] = flags
    out["flag_threshold"] = thr_out.astype(np.float32)
    out["slice_rank"] = rank_in_slice.astype(np.int32)

    summary = (
        out.groupby(group_keys, dropna=False)
        .agg(total_samples=(flag_col, "size"), flagged_samples=("flagged", "sum"))
        .reset_index()
    )
    summary["flagged_fraction"] = summary["flagged_samples"] / summary["total_samples"]
    return out, summary


# ---------------------------------------------------------------------------
# 10. Incremental update helpers (impact zone, unscored sample detection)
# ---------------------------------------------------------------------------


#: Flag values that mean "the pipeline reached a final decision about this row"
#: even though some numeric columns are legitimately null.  Without this, rows
#: whose ``ewoc_code`` is absent from the legend (NaN label) or whose slice was
#: too small to score were re-detected as "unscored" on *every* incremental
#: update, so the impact zone grew each run and the update mode never
#: converged.  ``normal`` etc. are included so the check is a simple non-null
#: test on the flag column.
TERMINAL_FLAG_VALUES: set = {
    "normal",
    "flagged",
    "suspect",
    "candidate",
    "unscored",
    "unscorable",
    "unmapped",
    "skipped",
}


def find_unscored_samples(
    long_parquet_dir: Union[str, Path],
    anomaly_cols: Optional[List[str]] = None,
    parquet_glob: str = "**/*.parquet",
    read_cols: Optional[List[str]] = None,
    flag_col: Optional[str] = None,
) -> pd.DataFrame:
    """Scan partitioned long-format parquets and return rows with missing anomaly scores.

    A row is "unscored" if the anomaly columns do not exist in the file at all
    (newly added dataset), or if the pipeline never reached a decision for it.

    "Reached a decision" is determined from *flag_col* when available: a row
    carrying any of :data:`TERMINAL_FLAG_VALUES` is considered handled, even if
    other anomaly columns are null.  Falling back to "any column is NaN" (the
    previous rule) made unmappable and too-small-to-score rows permanently
    unscored, so each ``--mode update`` run rediscovered them and re-expanded
    the impact zone without ever making progress.

    Only the lightweight identifier columns are returned — never band data.

    Parameters
    ----------
    long_parquet_dir
        Root directory of the hive-partitioned long-format parquet dataset
        (the ``_with_anomalies`` version, or the raw version if columns were
        pre-added with NaN).
    anomaly_cols
        The 6 anomaly column names.  Defaults to :data:`ANOMALY_COLUMNS`.
    parquet_glob
        Glob pattern to find parquet files under *long_parquet_dir*.
    read_cols
        Extra columns to include in the output beyond the identifiers +
        anomaly columns.  Useful for e.g. ``['ewoc_code']``.

    Returns
    -------
    pd.DataFrame
        Columns: ``ref_id, sample_id`` plus any *read_cols*.
        One row per **unique** ``(ref_id, sample_id)`` that is unscored.
    """
    import pyarrow.parquet as pq

    long_parquet_dir = Path(long_parquet_dir)
    if anomaly_cols is None:
        anomaly_cols = list(ANOMALY_COLUMNS)

    id_cols = ["ref_id", "sample_id"]
    if flag_col is None:
        flag_candidates = [c for c in anomaly_cols if c.endswith("_anomaly_flag")]
        flag_col = flag_candidates[0] if flag_candidates else None
    want_cols = list(dict.fromkeys(id_cols + (read_cols or []) + anomaly_cols))

    parquet_files = sorted(long_parquet_dir.glob(parquet_glob))
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found under {long_parquet_dir} with pattern {parquet_glob}"
        )

    unscored_parts: List[pd.DataFrame] = []
    for pf in parquet_files:
        schema = pq.read_schema(str(pf))
        available = set(schema.names)

        # If anomaly columns are completely missing → entire file is unscored
        has_anomaly_cols = all(c in available for c in anomaly_cols)

        # Read only the columns we need (intersection with what's available)
        cols_to_read = [c for c in want_cols if c in available]
        df = pd.read_parquet(pf, columns=cols_to_read)
        if df.empty:
            continue

        if not has_anomaly_cols:
            # All rows are unscored
            unscored = df
        elif flag_col is not None and flag_col in df.columns:
            # Decision-based: a terminal flag value means "handled", whatever
            # the other columns contain.
            decided = df[flag_col].astype("object").isin(TERMINAL_FLAG_VALUES)
            unscored = df[~decided]
        else:
            # Rows where ANY anomaly column is NaN (legacy fallback)
            mask = df[anomaly_cols].isna().any(axis=1)
            unscored = df[mask]

        if unscored.empty:
            continue

        # Keep only identifier columns (+ read_cols), deduplicate per sample_id
        keep = [c for c in id_cols + (read_cols or []) if c in unscored.columns]
        unscored_parts.append(unscored[keep].drop_duplicates(subset="sample_id"))

    if not unscored_parts:
        return pd.DataFrame(columns=id_cols + (read_cols or []))

    result = pd.concat(unscored_parts, ignore_index=True).drop_duplicates(
        subset="sample_id"
    )
    return result


def compute_impact_zone(
    unscored_h3_cells: Sequence[str],
    h3_levels: Sequence[int],
    neighbour_rings: int = 1,
) -> set:
    """Compute the set of H3 cells (at all adaptive levels) that are affected.

    Starting from the ``h3_l3_cell`` values of unscored points, derives
    parent/child cells at every requested level and expands each by
    *neighbour_rings* using ``h3.grid_disk``.

    Parameters
    ----------
    unscored_h3_cells
        The ``h3_l3_cell`` values of unscored points.
    h3_levels
        The adaptive H3 levels used by the pipeline (e.g. ``[2, 3]``).
    neighbour_rings
        How many rings of neighbours to include around each affected cell.
        1 ring is usually sufficient to cover ``merge_small_slices`` spillover.

    Returns
    -------
    set
        Union of all affected H3 cell indices across all levels.
    """
    import h3 as _h3

    impact: set = set()
    unique_cells = set(str(c) for c in unscored_h3_cells if c)

    for lvl in h3_levels:
        level_cells: set = set()
        for cell in unique_cells:
            if lvl == 3:
                derived = cell
            elif lvl < 3:
                derived = _h3.cell_to_parent(cell, lvl)
            else:
                # Finer than L3: we can't derive a single child without lat/lon,
                # but the parent at L3 is the cell itself, so include it.
                # The actual finer cells will be handled when we filter by
                # checking if cell_to_parent(point_h3, lvl_coarse) is in impact.
                derived = cell
            level_cells.add(derived)
            # Expand by neighbour rings
            for ring_cell in _h3.grid_disk(derived, neighbour_rings):
                level_cells.add(ring_cell)
        impact.update(level_cells)

    return impact


def load_affected_embeddings_from_cache(
    embeddings_db_path: str,
    impact_cells: set,
    h3_levels: Sequence[int],
    restrict_model_hash: Optional[str] = None,
    group_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Load embeddings from DuckDB for all points in the impact zone.

    A point is "affected" if its ``h3_l3_cell`` (or its parent at any
    coarser level in *h3_levels*) falls within *impact_cells*.

    This function is memory-efficient: it first queries only the lightweight
    ``h3_l3_cell`` column to determine which L3 cells fall inside the impact
    zone, then fetches only those rows (with embeddings) from DuckDB.  This
    avoids loading the entire ~7M-row embeddings table into memory.

    Parameters
    ----------
    embeddings_db_path
        Path to the DuckDB embeddings cache.
    impact_cells
        Set of H3 cell indices (at various levels) from :func:`compute_impact_zone`.
    h3_levels
        The adaptive H3 levels (e.g. ``[2, 3]``).
    restrict_model_hash
        If set, only load embeddings for this model hash.
    group_cols
        Additional columns to load (e.g. ``['ref_id']``).

    Returns
    -------
    (df, embed_cols)
        DataFrame with all embeddings in the impact zone, plus the list of
        ``embedding_0..embedding_127`` column names.
    """
    import duckdb
    import h3 as _h3

    group_cols = list(group_cols or [])

    con = duckdb.connect(embeddings_db_path, read_only=True)
    try:
        cols_df = con.execute("PRAGMA table_info('embeddings_cache')").fetchdf()
        embed_cols = [c for c in cols_df.name.tolist() if c.startswith("embedding_")]

        available = set(cols_df.name.tolist())
        base_cols = [
            "sample_id", "ewoc_code", "model_hash", "ref_id",
            "h3_l3_cell", "lat", "lon",
        ]
        # Select only columns the cache actually has, so a group_col or time_col
        # that was never cached surfaces as a message instead of a SQL error on
        # a nonexistent column.
        _wanted = list(dict.fromkeys([*base_cols, *group_cols]))
        _missing = [c for c in _wanted if c not in available]
        if _missing:
            print(
                f"[anomaly] NOTE: {_missing} not present in embeddings_cache; "
                "not loaded."
            )
        select_cols = [c for c in _wanted if c in available]

        # ------------------------------------------------------------------
        # Phase 1: Lightweight query — fetch only h3_l3_cell to determine
        # which L3 cells are in the impact zone.  This avoids loading the
        # full embeddings table (~7M rows × 128 floats) into memory.
        # ------------------------------------------------------------------
        if restrict_model_hash:
            l3_cells_df = con.execute(
                "SELECT DISTINCT h3_l3_cell FROM embeddings_cache WHERE model_hash = ?",
                [restrict_model_hash],
            ).fetchdf()
        else:
            l3_cells_df = con.execute(
                "SELECT DISTINCT h3_l3_cell FROM embeddings_cache"
            ).fetchdf()

        if l3_cells_df.empty:
            return pd.DataFrame(), embed_cols

        # Check which L3 cells (or their parents at coarser levels) fall in impact_cells
        all_l3 = l3_cells_df["h3_l3_cell"].tolist()
        matching_l3: set = set()
        for cell in all_l3:
            if not cell:
                continue
            for lvl in h3_levels:
                if lvl == 3:
                    derived = cell
                elif lvl < 3:
                    try:
                        derived = _h3.cell_to_parent(cell, lvl)
                    except Exception:
                        continue
                else:
                    # Finer than L3: the L3 cell itself is the best we can check
                    derived = cell
                if derived in impact_cells:
                    matching_l3.add(cell)
                    break  # no need to check other levels for this cell

        if not matching_l3:
            return pd.DataFrame(), embed_cols

        # ------------------------------------------------------------------
        # Phase 2: Fetch only the matching rows (with embeddings) from DuckDB.
        # Register the matching L3 cells as a temporary table for the IN filter.
        # ------------------------------------------------------------------
        filter_df = pd.DataFrame({"h3_l3_cell": list(matching_l3)})
        con.register("impact_l3_cells", filter_df)

        query = (
            f"SELECT {', '.join('e.' + c for c in select_cols + embed_cols)} "
            f"FROM embeddings_cache e "
            f"INNER JOIN impact_l3_cells f ON e.h3_l3_cell = f.h3_l3_cell"
        )
        if restrict_model_hash:
            query += " AND e.model_hash = ?"
            df = con.execute(query, [restrict_model_hash]).fetchdf()
        else:
            df = con.execute(query).fetchdf()
    finally:
        con.close()

    return df, embed_cols


def _write_parquet_preserve_geo(
    df: pd.DataFrame,
    out_path: Path,
    original_schema_metadata: Optional[dict],
) -> None:
    """Write *df* to *out_path* as parquet, preserving any geoparquet metadata.

    When *original_schema_metadata* contains a ``b'geo'`` key the file is
    written via PyArrow so that the ``geo`` metadata block is re-attached to
    the output schema.  Otherwise falls back to ``df.to_parquet()``.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    if original_schema_metadata and b"geo" in original_schema_metadata:
        # Convert to Arrow, attach original geo + pandas metadata
        tbl = pa.Table.from_pandas(df, preserve_index=False)
        # Merge: keep existing pandas metadata from Arrow conversion, overwrite
        # with original geo metadata so downstream GIS tools can still read it.
        existing_meta = dict(tbl.schema.metadata or {})
        existing_meta[b"geo"] = original_schema_metadata[b"geo"]
        tbl = tbl.replace_schema_metadata(existing_meta)
        pq.write_table(
            tbl,
            str(out_path),
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
    else:
        df.to_parquet(out_path, index=False)


def merge_scores_to_long_parquets(
    scored_df: pd.DataFrame,
    long_parquet_dir: Union[str, Path],
    output_parquet_dir: Union[str, Path],
    anomaly_cols: Optional[List[str]] = None,
    parquet_glob: str = "**/*.parquet",
    only_affected_ref_ids: Optional[set] = None,
) -> int:
    """Write anomaly scores back to long-format parquets.

    For each parquet file under *long_parquet_dir*, left-joins the anomaly
    columns from *scored_df* on ``(ref_id, sample_id)`` and writes to
    *output_parquet_dir* (which may be the same directory for in-place update).

    Works with both nested ``.parquet`` datasets (hive-partitioned) and flat
    ``.geoparquet`` directories.  When the input and output directories are the
    same (in-place update) geoparquet files are updated atomically via a temp
    file and any ``geo`` metadata in the original file is preserved so the
    output remains a valid GeoParquet.

    Parameters
    ----------
    scored_df
        Must contain ``ref_id, sample_id`` plus all *anomaly_cols*.
    long_parquet_dir
        Input long-format parquet root.
    output_parquet_dir
        Output root.  If same as *long_parquet_dir*, files are updated in place.
    anomaly_cols
        Column names to write.  Defaults to :data:`ANOMALY_COLUMNS`.
    parquet_glob
        Glob pattern for finding parquet files.  Use ``"*.geoparquet"`` for a
        flat geoparquet directory; use ``"**/*.parquet"`` for nested datasets.
    only_affected_ref_ids
        If provided, only rewrite parquets whose ``ref_id`` (derived from the
        hive partition name or from the data) is in this set.  All other files
        are either left untouched (if input==output) or copied as-is.

    Returns
    -------
    int
        Number of parquet files written/updated.
    """
    import gc
    import shutil
    import tempfile

    import pyarrow.parquet as pq

    long_parquet_dir = Path(long_parquet_dir)
    output_parquet_dir = Path(output_parquet_dir)
    if anomaly_cols is None:
        anomaly_cols = list(ANOMALY_COLUMNS)

    in_place = long_parquet_dir.resolve() == output_parquet_dir.resolve()

    # Build a lookup: only the columns we need from scored_df
    merge_cols = ["ref_id", "sample_id"] + anomaly_cols
    available_merge = [c for c in merge_cols if c in scored_df.columns]
    # De-duplicate on the FULL join key.  Collapsing on sample_id alone while
    # joining on (ref_id, sample_id) silently dropped the scores of every row
    # whose sample_id repeated under a different ref_id — those rows then came
    # back as NaN and were rediscovered as "unscored" forever.
    dedup_subset = [c for c in ("ref_id", "sample_id") if c in scored_df.columns]
    scores_lookup = scored_df[available_merge].drop_duplicates(subset=dedup_subset)
    if "sample_id" in scores_lookup.columns:
        n_dup_ids = int(scores_lookup["sample_id"].duplicated().sum())
        if n_dup_ids:
            print(
                f"[anomaly] NOTE: {n_dup_ids:,} sample_id values occur under more "
                "than one ref_id; joining on the composite key."
            )

    parquet_files = sorted(long_parquet_dir.glob(parquet_glob))
    n_written = 0

    for pf in parquet_files:
        # Derive ref_id from hive partition path if possible
        # e.g. .../ref_id=2020_AUT_xyz/2020_AUT_xyz.parquet
        # For flat geoparquet: stem IS the ref_id (e.g. 2019_BGR_Eurocrops_POLY_110)
        file_ref_id = None
        for part in pf.parts:
            if part.startswith("ref_id="):
                file_ref_id = part.split("=", 1)[1]
                break
        if file_ref_id is None:
            file_ref_id = pf.stem

        # Skip files not in the affected set
        if only_affected_ref_ids is not None:
            if file_ref_id not in only_affected_ref_ids:
                # If not in-place, copy to output only if the output doesn't
                # already exist (don't overwrite a previously scored file)
                if not in_place:
                    out_path = output_parquet_dir / pf.relative_to(long_parquet_dir)
                    if not out_path.exists():
                        out_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(pf, out_path)
                continue

        # Read the original geo metadata before loading data (cheap)
        orig_schema_meta = dict(pq.read_schema(str(pf)).metadata or {})

        df_long = pd.read_parquet(pf)
        if df_long.empty:
            continue

        # Drop existing anomaly columns to avoid _x/_y suffixes
        cols_to_drop = [c for c in anomaly_cols if c in df_long.columns]
        if cols_to_drop:
            df_long.drop(columns=cols_to_drop, inplace=True)

        # Left-join: broadcasts per-(ref_id, sample_id)
        df_long = df_long.merge(
            scores_lookup,
            on=["ref_id", "sample_id"],
            how="left",
        )

        # Sort for better compression
        if "timestamp" in df_long.columns:
            df_long["timestamp"] = pd.to_datetime(df_long["timestamp"])
            df_long.sort_values(["sample_id", "timestamp"], inplace=True)
        df_long.reset_index(drop=True, inplace=True)

        out_path = output_parquet_dir / pf.relative_to(long_parquet_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if in_place:
            # Write to a temp file first, then atomically replace the original
            tmp_fd, tmp_path = tempfile.mkstemp(
                suffix=pf.suffix, dir=pf.parent, prefix=f"_tmp_{pf.name}_"
            )
            try:
                import os as _os
                _os.close(tmp_fd)
                _write_parquet_preserve_geo(df_long, Path(tmp_path), orig_schema_meta)
                _os.replace(tmp_path, str(pf))
            except Exception:
                try:
                    _os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        else:
            _write_parquet_preserve_geo(df_long, out_path, orig_schema_meta)

        n_written += 1

        del df_long
        gc.collect()

    return n_written


