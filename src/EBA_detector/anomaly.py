"""Anomaly detection pipeline operating purely on cached Presto embeddings.

This version assumes a DuckDB cache already exists with columns:
``sample_id, model_hash, ref_id, ewoc_code, h3_l3_cell, embedding_0..embedding_127``.
Embeddings are never recomputed; the pipeline always loads them from the cache.
Optional label domain switching between ``ewoc_code`` and mapped ``finetune_class``.

Grouping:
- Slices are defined by: group_cols (optional) + [h3 cell] + [label col]
- group_cols defaults to [] (i.e., global per (h3, label) slices)

Adaptive H3 resolution:
- When ``h3_level`` is a list (e.g. ``[1, 2, 3]``), levels are tried
  coarsest → finest.  A slice is resolved at the coarsest level where its
  size is within [min_slice_size, max_slice_size].  Dense regions (Europe)
  with oversized coarse-level slices are pushed to finer cells; sparse
  regions (Africa) are resolved at a coarser level.

Module layout
~~~~~~~~~~~~~
- **anomaly_utils.py** — pure computation helpers (scoring, metrics, mapping,
  flagging, adaptive H3 assignment).  Stateless building blocks.
- **anomaly.py** *(this file)* — pipeline orchestration: data loading,
  incremental mode, class mapping, scoring dispatch, anomaly categorization,
  and output writing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import duckdb
import geopandas as gpd
import h3
import numpy as np
import pandas as pd

# All computation helpers live in anomaly_utils — import them here so that
# any downstream code doing ``from EBA_detector.anomaly import <func>``
# continues to work.
from EBA_detector.anomaly_utils import (
    MIN_SCORING_SLICE_SIZE,
    _SCORE_COLS,
    ANOMALY_COLUMNS,
    _as_label_levels,
    _require_label_columns,
    _load_mapping_df,
    _add_hierarchical_ref_outlier_class,
    assign_adaptive_h3_level,
    merge_small_slices,
    compute_slice_centroids,
    compute_scores_for_slice,
    _score_group_simple,
    score_slices_hierarchical,
    add_alt_class_centroid_metrics,
    add_knn_label_purity_for_flagged,
    add_confidence_from_score,
    add_flagged_robust_confidence,
    apply_confidence_fusion,
    flag_anomalies,
    find_unscored_samples,
    compute_impact_zone,
    load_affected_embeddings_from_cache,
    merge_scores_to_long_parquets,
)

# Robustness extensions: slice-trust gating (handles the un-finetuned-encoder
# circularity) and parcel-aware scoring.
from EBA_detector.robust_extensions import (
    compute_slice_trust,
    apply_trust_to_confidence,
    downgrade_flags_low_trust,
)

# Absolute-scale calibration: turns within-slice relative scores into
# cross-slice comparable evidence.  See calibration.py for why this is the
# central fix for both the clean-slice false positives and the
# heavily-contaminated-slice false negatives.
from EBA_detector.calibration import (
    add_absolute_scores,
    compute_null_reference,
    suggest_abs_z_threshold,
)
from EBA_detector.quality import (
    assert_h3_matches_coordinates,
    assert_single_model_hash,
    validate_embeddings,
)


# ===================================================================
# Pipeline
# ===================================================================


def _worldcereal_map_classes(df: pd.DataFrame, class_mappings_name: str) -> pd.DataFrame:
    """Lazy proxy for ``worldcereal.utils.refdata.map_classes``.

    Imported on demand rather than at module scope: ``map_classes`` is only
    reached when ``map_to_finetune=True``, but a module-level import made the
    whole orchestration layer — including the scoring, flagging and calibration
    it coordinates — impossible to import (and therefore to unit-test) without a
    full worldcereal install.  The README advertises tests that need "no DuckDB,
    no network"; this is what makes that true of ``anomaly.py`` as well.
    """
    from worldcereal.utils.refdata import map_classes

    return map_classes(df, class_mappings_name)


def get_class_mappings(*args, **kwargs):
    """Lazy re-export of ``worldcereal.utils.refdata.get_class_mappings``."""
    from worldcereal.utils.refdata import get_class_mappings as _impl

    return _impl(*args, **kwargs)



def _load_embeddings(
    con: duckdb.DuckDBPyConnection,
    group_cols: list[str],
    restrict_model_hash: Optional[str],
    extra_cols: Optional[Sequence[str]] = None,
) -> Tuple[pd.DataFrame, list[str]]:
    """Load cached embeddings from DuckDB.

    Returns ``(df, embed_cols)`` where *embed_cols* are the raw
    ``embedding_0 … embedding_N`` column names.

    *extra_cols* (e.g. a temporal column) are selected **only if the cache
    actually has them**; a missing one is reported rather than turned into a
    SQL error on a nonexistent column.
    """
    cols_df = con.execute("PRAGMA table_info('embeddings_cache')").fetchdf()
    available = set(cols_df.name.tolist())
    embed_cols = [c for c in cols_df.name.tolist() if c.startswith("embedding_")]

    base_cols = [
        "sample_id",
        "ewoc_code",
        "model_hash",
        "ref_id",
        "h3_l3_cell",
        "lat",
        "lon",
        # "country",
    ]
    wanted = [*base_cols, *group_cols, *(list(extra_cols) if extra_cols else [])]
    missing = [c for c in dict.fromkeys(wanted) if c not in available]
    if missing:
        print(
            f"[anomaly] NOTE: requested column(s) {missing} are not in the "
            "embeddings cache and will not be loaded. If one of these is your "
            "time_col or a group_col, rebuild the cache with that column — "
            "otherwise the corresponding control is silently inactive."
        )
    select_cols = [c for c in dict.fromkeys(wanted) if c in available]

    # Parameterised rather than interpolated: model_hash is external input.
    query = f"SELECT {', '.join(select_cols + embed_cols)} FROM embeddings_cache"
    if restrict_model_hash:
        query += " WHERE model_hash = ?"
        df = con.execute(query, [restrict_model_hash]).fetchdf()
    else:
        df = con.execute(query).fetchdf()
    return df, embed_cols


def _handle_incremental_mode(
    df: pd.DataFrame,
    output_samples_path: Optional[str],
    con: Optional[duckdb.DuckDBPyConnection],
) -> Tuple[pd.DataFrame, Optional[gpd.GeoDataFrame], set]:
    """Filter out already-processed sample_ids when resuming.

    Returns ``(df_filtered, existing_df_full_or_None, existing_ids_set)``.
    """
    if not output_samples_path:
        print(
            "[anomaly] WARNING: skip_existing_samples=True but "
            "output_samples_path not set. Processing all samples."
        )
        return df, None, set()

    out_path = Path(output_samples_path)
    if not out_path.exists():
        print(
            f"[anomaly] skip_existing_samples=True but output file doesn't exist yet: "
            f"{output_samples_path}"
        )
        print(f"[anomaly] Processing all {len(df):,} samples from scratch...")
        return df, None, set()

    print(f"[anomaly] Loading existing results from {output_samples_path}...")
    existing_df_full = gpd.read_parquet(output_samples_path)
    if "sample_id" not in existing_df_full.columns:
        if con is not None:
            con.close()
        raise ValueError(
            f"Existing output_samples_path has no 'sample_id' column: {output_samples_path}"
        )
    existing_ids = set(existing_df_full["sample_id"].astype(str).unique())

    before_count = len(df)
    df_sample_ids = df["sample_id"].astype(str)
    df = df[~df_sample_ids.isin(existing_ids)].copy()
    after_count = len(df)

    print(f"[anomaly] Found {len(existing_ids):,} existing samples")
    print(f"[anomaly] Filtering: {before_count:,} -> {after_count:,} rows to process")

    return df, existing_df_full, existing_ids


def _apply_class_mapping(
    df: pd.DataFrame,
    *,
    map_to_finetune: bool,
    mapping_file: Optional[Union[str, dict]],
    label_cols: list[str],
    class_mappings_name: str,
    con: Optional[duckdb.DuckDBPyConnection] = None,
) -> pd.DataFrame:
    """Map ewoc_code → label column(s) using the chosen strategy."""

    if map_to_finetune:
        print(f"[anomaly] Mapping classes using '{class_mappings_name}'...")
        return _worldcereal_map_classes(df, class_mappings_name)

    if mapping_file is None:
        return df

    print(f"[anomaly] Mapping classes using mapping_file")

    map_df = _load_mapping_df(
        mapping_file,
        label_cols=label_cols,
        class_mappings_name=class_mappings_name,
    )

    if "ewoc_code" not in map_df.columns:
        if con is not None:
            con.close()
        raise ValueError("mapping_file must contain an 'ewoc_code' column")

    missing_map_cols = [c for c in label_cols if c not in map_df.columns]
    if missing_map_cols:
        if con is not None:
            con.close()
        raise ValueError(f"mapping_file missing required label column(s): {missing_map_cols}")

    # Normalize ewoc_code for joining (remove dashes)
    keep_cols = ["ewoc_code", *label_cols]
    map_df = map_df[keep_cols].copy()
    map_df["ewoc_code_clean"] = (
        map_df["ewoc_code"].astype(str).str.replace("-", "", regex=False)
    )

    df["ewoc_code_clean"] = df["ewoc_code"].astype(str).str.replace("-", "", regex=False)

    # Drop pre-existing label columns to avoid collisions
    df = df.drop(columns=[c for c in label_cols if c in df.columns], errors="ignore")

    df = df.merge(
        map_df[["ewoc_code_clean", *label_cols]],
        on="ewoc_code_clean",
        how="left",
    )
    df = df.drop(columns=["ewoc_code_clean"], errors="ignore")
    return df


def _assign_anomaly_categories(
    flagged_df: pd.DataFrame,
    *,
    abs_z_suspect: float = 4.0,
    abs_z_candidate: float = 5.5,
    purity_suspect_max: float = 0.60,
    purity_candidate_max: float = 0.35,
    purity_veto: float = 0.80,
    margin_suspect_max: float = 0.0,
    margin_candidate_max: float = 0.0,
    rank_suspect_min: float = 0.98,
    rank_candidate_min: float = 0.995,
    suspect_votes: int = 2,
    candidate_votes: int = 3,
    require_absolute: bool = True,
) -> pd.DataFrame:
    """Assign ``S_anomaly`` and ``combined_anomaly`` escalation categories.

    What changed and why
    --------------------
    The previous ``combined_anomaly`` claimed a 2-of-3 / 3-of-3 "consensus"
    over ``S_rank``, ``S_rank_min`` and ``S_z``.  Those are not three
    independent views: ``S_rank`` is the *mean* and ``S_rank_min`` the *min* of
    the same two rank vectors, so the pair moves together and can outvote the
    only signal carrying an absolute scale.  Because rank percentiles are
    uniform by construction, "2 of 3 above 0.98" reduced to "the top ~2 % of
    this slice" — a fixed quota that fires in clean slices just as reliably as
    in dirty ones.

    The vote is now built from **genuinely different measurements**:

    ``absolute``
        ``abs_z`` — distance in robust sigma against the cross-slice null
        (:mod:`EBA_detector.calibration`).  The only vote that knows whether
        this slice is unusual *compared to other slices*.
    ``purity``
        ``knn_same_label_frac_ctx`` — what fraction of the point's nearest
        neighbours in its geographic context carry the same label.  Independent
        of any centroid.
    ``margin``
        ``alt_margin_ctx`` — distance to the nearest *other* class centroid
        minus distance to its own.  Negative means the embedding prefers a
        different label, which is the closest thing to direct evidence of
        mislabelling that this method produces.
    ``rank``
        The within-slice rank, retained as one vote among several rather than
        as the whole decision.

    Purity veto
    -----------
    A point whose neighbours overwhelmingly share its label (``>= purity_veto``)
    is capped at ``flagged`` and can never reach ``suspect`` / ``candidate``,
    however far it sits from the centroid.  Being an unusual *example* of a
    class — a late-planted field, an odd cultivar, a different soil — is not
    evidence of a wrong label when the neighbourhood agrees. This single rule
    removes a large share of the basemap false positives.

    Support caps
    ------------
    Rows scored against a borrowed reference (``ref_outlier_level > 0``, the
    coarse-label fallback) or sitting in an undersized slice are capped one
    level down: the evidence is real but weaker, and it should not drive
    deletion from a training set.
    """
    n = len(flagged_df)
    is_flagged = flagged_df["flagged"].fillna(False).to_numpy(dtype=bool)

    def _num(col: str, default: float) -> np.ndarray:
        if col not in flagged_df.columns:
            return np.full(n, default, dtype="float64")
        return (
            pd.to_numeric(flagged_df[col], errors="coerce")
            .fillna(default)
            .to_numpy(dtype="float64")
        )

    # --- S_anomaly: legacy rank-only view, retained for comparison ----------
    # Kept so ablations can quantify exactly how much the absolute gate and the
    # purity veto changed the outcome; it is NOT the shipped decision.
    S_anomaly = "S_anomaly"
    flagged_df[S_anomaly] = "normal"
    flagged_df.loc[is_flagged, S_anomaly] = "flagged"
    rank_pct = _num("rank_percentile", 0.0)
    s_val = _num("S", 0.0)
    flagged_df.loc[
        (rank_pct >= 0.98) & (s_val >= 0.95) & is_flagged, S_anomaly
    ] = "suspect"
    flagged_df.loc[
        (rank_pct >= 0.99) & (s_val >= 0.99) & is_flagged, S_anomaly
    ] = "candidate"

    # --- combined_anomaly: independent-signal consensus ---------------------
    combined = "combined_anomaly"
    flagged_df[combined] = "normal"
    flagged_df.loc[is_flagged, combined] = "flagged"

    abs_z = _num("abs_z", -np.inf)
    purity = _num("knn_same_label_frac_ctx", np.nan)
    margin = _num("alt_margin_ctx", np.nan)
    s_rank = _num("S_rank", 0.0)

    # A missing auxiliary signal must not count as evidence of anomaly.
    purity_known = np.isfinite(purity)
    margin_known = np.isfinite(margin)

    votes_suspect = (
        (abs_z >= abs_z_suspect).astype(int)
        + (purity_known & (purity <= purity_suspect_max)).astype(int)
        + (margin_known & (margin <= margin_suspect_max)).astype(int)
        + (s_rank >= rank_suspect_min).astype(int)
    )
    votes_candidate = (
        (abs_z >= abs_z_candidate).astype(int)
        + (purity_known & (purity <= purity_candidate_max)).astype(int)
        + (margin_known & (margin <= margin_candidate_max)).astype(int)
        + (s_rank >= rank_candidate_min).astype(int)
    )

    # The absolute vote is mandatory for escalation: without a cross-slice
    # scale there is no way to tell "worst in a clean slice" from "wrong".
    # With require_absolute=False (the legacy / ablation mode requested by
    # run_pipeline) the mandatory gate is lifted here as well — previously the
    # flag stage honoured the switch but escalation kept demanding abs_z, so
    # "--no-absolute-gate" was not the clean relative-only ablation it claimed
    # to be.  The abs_z *vote* still counts when the score is present.
    if require_absolute:
        abs_ok_suspect = abs_z >= abs_z_suspect
        abs_ok_candidate = abs_z >= abs_z_candidate
    else:
        abs_ok_suspect = np.ones(n, dtype=bool)
        abs_ok_candidate = np.ones(n, dtype=bool)

    flagged_df.loc[
        is_flagged & abs_ok_suspect & (votes_suspect >= suspect_votes), combined
    ] = "suspect"
    flagged_df.loc[
        is_flagged & abs_ok_candidate & (votes_candidate >= candidate_votes), combined
    ] = "candidate"

    flagged_df["escalation_votes"] = votes_suspect.astype(np.int8)

    # --- purity veto --------------------------------------------------------
    # Only applied where purity is *informative*.  In a context containing a
    # single label every point trivially has purity 1.0, which would veto every
    # escalation in the region — turning the safeguard into a blanket mute.
    # The veto means "the neighbours disagree with the flag", which requires at
    # least one alternative label to be present to disagree with.
    if "knn_same_label_frac_ctx" in flagged_df.columns:
        if "context_n_labels" in flagged_df.columns:
            informative = (
                pd.to_numeric(flagged_df["context_n_labels"], errors="coerce")
                .fillna(0)
                .to_numpy()
                >= 2
            )
        else:
            informative = np.ones(n, dtype=bool)
        vetoed = purity_known & informative & (purity >= purity_veto)
        flagged_df.loc[
            vetoed & flagged_df[combined].isin(["suspect", "candidate"]), combined
        ] = "flagged"
        flagged_df["purity_veto"] = vetoed

        # No corroboration available -> no strong claim.
        #
        # `suspect` / `candidate` assert that a sample is probably *mislabelled*.
        # That claim needs an alternative label for it to have been confused
        # with.  Where the context holds a single label, the purity and margin
        # votes are unavailable and the consensus silently degenerates back to
        # the two correlated within-slice signals — reintroducing the fixed
        # per-slice quota this rewrite exists to remove.  Cap such rows at
        # `flagged`: still surfaced for review, but not asserted as a label
        # error on evidence that cannot distinguish "unusual" from "wrong".
        no_corroboration = ~informative
        flagged_df.loc[
            no_corroboration & flagged_df[combined].isin(["suspect", "candidate"]),
            combined,
        ] = "flagged"
        flagged_df["corroborated"] = informative

    # --- weak-support caps --------------------------------------------------
    weak = np.zeros(n, dtype=bool)
    if "undersized_slice" in flagged_df.columns:
        weak |= flagged_df["undersized_slice"].fillna(False).to_numpy(dtype=bool)
    if "ref_outlier_level" in flagged_df.columns:
        weak |= (
            pd.to_numeric(flagged_df["ref_outlier_level"], errors="coerce")
            .fillna(0)
            .to_numpy()
            > 0
        )
    if "merge_steps" in flagged_df.columns:
        weak |= (
            pd.to_numeric(flagged_df["merge_steps"], errors="coerce")
            .fillna(0)
            .to_numpy()
            >= 2
        )

    if weak.any():
        # Compute the demotion from a SNAPSHOT.  Chaining two .loc assignments
        # re-reads the column after the first one, so a `candidate` became
        # `suspect` and was then immediately demoted again to `flagged` — a
        # two-level drop, not the one level documented.  That mattered much more
        # after `weak` widened from `undersized_slice` alone to also cover every
        # hierarchical-fallback row and every twice-merged slice, which between
        # them made `candidate` effectively unreachable for a large share of the
        # population.
        _demote = {"candidate": "suspect", "suspect": "flagged"}
        for col in (S_anomaly, combined):
            snapshot = flagged_df[col].to_numpy(copy=True)
            demoted = np.array(
                [_demote.get(v, v) if w else v for v, w in zip(snapshot, weak)],
                dtype=object,
            )
            flagged_df[col] = demoted
    flagged_df["weak_support"] = weak

    # --- terminal states for rows the detector could not judge --------------
    # These used to be reported as "normal", i.e. indistinguishable from "we
    # looked and it is fine".  They are now explicit so downstream training can
    # decide what to do with them, and so the incremental update pathway stops
    # rediscovering them as unscored on every run.
    if "scored" in flagged_df.columns:
        unscored = ~flagged_df["scored"].fillna(False).to_numpy(dtype=bool)
        flagged_df.loc[unscored, [S_anomaly, combined]] = "unscored"

    return flagged_df


def _merge_with_existing(
    flagged_gdf: gpd.GeoDataFrame,
    existing_df_full: Optional[gpd.GeoDataFrame],
) -> gpd.GeoDataFrame:
    """Append newly-scored rows to previously-saved results (incremental mode)."""
    if existing_df_full is None:
        return flagged_gdf

    print(
        f"[anomaly] Merging {len(flagged_gdf):,} new results with "
        f"{len(existing_df_full):,} existing results..."
    )

    # Align schemas (union of columns) to avoid missing-column issues
    all_cols = list(
        dict.fromkeys(
            [*existing_df_full.columns.tolist(), *flagged_gdf.columns.tolist()]
        )
    )
    existing_aligned = existing_df_full.reindex(columns=all_cols)
    new_aligned = flagged_gdf.reindex(columns=all_cols)

    combined = pd.concat([existing_aligned, new_aligned], axis=0, ignore_index=True)

    # Safety: if any overlaps happen, keep the last occurrence (new wins)
    if "sample_id" in combined.columns:
        combined["sample_id"] = combined["sample_id"].astype(str)
        combined = combined.drop_duplicates(
            subset=["sample_id"], keep="last"
        ).reset_index(drop=True)

    print(f"[anomaly] Total combined: {len(combined):,} samples")
    return combined


def _write_outputs(
    flagged_gdf: gpd.GeoDataFrame,
    summary_df: Optional[pd.DataFrame],
    slice_keys: list[str],
    output_samples_path: Optional[str],
    output_summary_path: Optional[str],
) -> None:
    """Persist results to disk (parquet + Excel)."""
    S_anomaly = "S_anomaly"
    combined_anomaly = "anomaly_flag"  # renamed from combined_anomaly before this call

    if output_samples_path:
        print(f"[anomaly] Writing flagged samples -> {output_samples_path}")
        # flagged_gdf = flagged_gdf.drop(
        #     columns=["embedding", "base_embedding"], errors="ignore"
        # )
        flagged_gdf = flagged_gdf.drop(columns=["base_embedding"], errors="ignore")
        flagged_gdf.to_parquet(output_samples_path, index=False)

    if output_summary_path and summary_df is not None:
        print(f"[anomaly] Writing summary -> {output_summary_path}")
        summary_df.to_parquet(output_summary_path, index=False)
        summary_df.to_excel(
            Path(output_summary_path).with_suffix(".xlsx"),
            index=False,
        )

        # Cross-tabulation: long form
        cross_long = (
            flagged_gdf.groupby([*slice_keys, S_anomaly, combined_anomaly], dropna=False)
            .size()
            .reset_index(name="n")
        )
        cross_long.to_parquet(
            Path(output_summary_path).with_name(
                Path(output_summary_path).stem + "_anomalies_cross_long.parquet"
            ),
            index=False,
        )
        cross_long.to_excel(
            Path(output_summary_path).with_name(
                Path(output_summary_path).stem + "_anomalies_cross_long.xlsx"
            ),
            index=False,
        )

        # Cross-tabulation: wide matrix with flattened column names
        cross_wide = cross_long.pivot_table(
            index=slice_keys,
            columns=[S_anomaly, combined_anomaly],
            values="n",
            fill_value=0,
            aggfunc="sum",
        )
        # Flatten MultiIndex columns -> e.g. "S=candidate__C=suspect"
        cross_wide.columns = [
            f"S={s}__C={c}" for (s, c) in cross_wide.columns.to_list()
        ]
        cross_wide = cross_wide.reset_index()

        cross_wide.to_parquet(
            Path(output_summary_path).with_name(
                Path(output_summary_path).stem + "_anomalies_cross_wide.parquet"
            ),
            index=False,
        )
        cross_wide.to_excel(
            Path(output_summary_path).with_name(
                Path(output_summary_path).stem + "_anomalies_cross_wide.xlsx"
            ),
            index=True,
        )


# ===================================================================
# Main entry point
# ===================================================================


def run_pipeline(
    embeddings_db_path: str,
    restrict_model_hash: Optional[str] = None,
    label_domain: Union[str, Sequence[str]] = "ewoc_code",
    map_to_finetune: bool = False,
    class_mappings_name: str = "LANDCOVER10",
    mapping_file: Optional[Union[str, dict]] = None,
    h3_level: Union[int, Sequence[int]] = 3,
    group_cols: Optional[Sequence[str]] = None,
    min_slice_size: int = 100,
    max_slice_size: Optional[int] = None,
    merge_small_slice: bool = True,
    max_merge_iterations=10,
    threshold_mode: str = "stable_mad",
    percentile_q: float = 0.96,
    mad_k: float = 3.3,
    abs_threshold: Optional[float] = None,
    fdr_alpha: float = 0.05,
    min_flagged_per_slice: Optional[int] = None,
    max_flagged_fraction: Optional[float] = None,
    max_full_pairwise_n: Optional[int] = 0,
    norm_percentiles: Tuple[float, float] = (2.0, 98.0),
    centroid_mode: str = "trimmed",
    centroid_trim: float = 0.45,
    gate_confidence_by_flag: bool = True,
    apply_slice_trust: bool = False,
    slice_trust_min: float = 0.05,
    # --- absolute-scale calibration -------------------------------------
    require_absolute: bool = True,
    abs_z_k: float = 3.3,
    abs_z_suspect: float = 4.0,
    abs_z_candidate: float = 5.5,
    null_extra_keys: Optional[Sequence[str]] = ("h3_null_region",),
    null_region_level: Optional[int] = 1,
    null_shrink_k: float = 5.0,
    abs_combine: str = "min",
    null_scale_estimator: str = "left_tail",
    # --- support / quality ----------------------------------------------
    min_scoring_slice_size: int = MIN_SCORING_SLICE_SIZE,
    quality_gate: bool = True,
    strict_quality: bool = False,
    # --- temporal control -------------------------------------------------
    time_col: Optional[str] = None,
    # --- context for auxiliary signals ------------------------------------
    context_group_cols: Optional[Sequence[str]] = None,
    # --- auxiliary-signal fusion -----------------------------------------
    apply_confidence_fusion_to_output: bool = True,
    purity_veto: float = 0.80,
    output_samples_path: Optional[str] = None,
    output_summary_path: Optional[str] = None,
    skip_existing_samples: bool = False,
    skip_classes: Optional[Sequence[str]] = None,
    debug: bool = False,
    embeddings_df: Optional[Tuple[pd.DataFrame, list]] = None,
    write_outputs: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run anomaly detection using only cached embeddings.

    Grouping
    --------
    slice = group_cols + [h3 cell at chosen level] + [label_col]

    Parameters
    ----------
    h3_level
        H3 resolution(s) for spatial grouping.

        - **Single int** (e.g. ``3``): use a fixed H3 level for all points
          (original behaviour).
        - **List of ints** (e.g. ``[1, 2, 3]``): *adaptive* mode.  Levels
          are tried **coarsest → finest** (ascending by H3 number).
          A slice is resolved at the coarsest level where its size is both
          ≥ *min_slice_size* and ≤ *max_slice_size*.  Slices that are too
          large at a coarse level are pushed to finer levels where the
          geographic cell is smaller.  Slices that are too small are also
          pushed finer; any still-unresolved points after the finest level
          are assigned there unconditionally and handled later by
          ``merge_small_slices``.
    max_slice_size
        (Adaptive mode only.)  Upper cap on slice size per level.  If a
        slice at the current (coarse) H3 level exceeds this, those points
        are pushed to the next finer level.  At the finest level the cap is
        not enforced — all remaining points are resolved unconditionally.
        Ignored when *h3_level* is a single int.
    norm_percentiles
        Percentiles used for per-slice min-max normalization of
        cosine_distance and knn_distance.  Default ``(5, 95)``
        preserves existing behavior.
    skip_existing_samples
        If *True* and *output_samples_path* exists, loads existing results,
        skips already-processed sample_id rows, computes only missing ones,
        then appends old + new and writes back.  This does **not** recompute
        outlier scores for existing sample_ids.
    skip_classes
        Optional list of label values (in the *label_domain* column) that
        are excluded from all scoring, flagging, and confidence steps.
        Their rows are held aside and re-joined to the output at the end
        with all score / outlier columns set to NaN.  Pass e.g.
        ``skip_classes=["built-up", "ignore"]``.
    embeddings_df
        If provided, a tuple of ``(df, embed_cols)`` — pre-loaded embeddings
        DataFrame and the list of embedding column names.  When set, the
        pipeline skips the DuckDB load entirely and uses this data instead.
        This is used by the incremental update pathway to pass only the
        impact-zone subset of embeddings.
    write_outputs
        If *False*, skip writing output parquet / Excel files even when
        *output_samples_path* / *output_summary_path* are set.  The scored
        DataFrames are still returned.  Default *True* preserves existing
        behaviour.
    centroid_mode
        How the per-slice reference centroid is computed: ``"trimmed"``
        (default; iterative trimmed-mean that resists outlier *masking*),
        ``"median"`` (per-dimension median), or ``"mean"`` (legacy plain mean —
        use only for ablations).  The trimmed centroid prevents a contaminated
        slice from hiding its own outliers by dragging the reference toward the
        anomalous mass.
    centroid_trim
        Fraction of farthest points dropped when recomputing the trimmed
        centroid.  Should be >= the largest outlier fraction you expect in a
        slice (default 0.10, matching the typical ``max_flagged_fraction``).
    gate_confidence_by_flag
        When *True* (default), ``confidence_nonoutlier`` is clamped to 1.0 for
        samples that were not flagged, keeping the continuous confidence
        consistent with the discrete ``anomaly_flag`` and avoiding a spurious
        penalty on the highest-ranked point of clean slices.  Set *False* for
        the legacy ungated behaviour.
    require_absolute, abs_z_k, abs_z_suspect, abs_z_candidate
        Enable and tune the **absolute** gate.  Every within-slice score in
        this pipeline is percentile-normalised per slice, so on its own it
        cannot tell "the most unusual point of a clean slice" from "a
        mislabelled point": the top ~2 % of every slice reaches the escalation
        thresholds by construction.  With *require_absolute* a point must also
        exceed *abs_z_k* robust sigma against a null pooled across many slices
        of the same class, built from one summary statistic per slice so a
        contaminated slice cannot calibrate its own errors away.  Set
        ``require_absolute=False`` to reproduce the legacy relative-only
        behaviour for ablations.
    null_extra_keys, null_region_level
        How the cross-slice null is localised.

        The null answers "how far from its own slice centroid does a typical
        sample of this class sit?".  Pooling that question **globally** is
        wrong, because the answer legitimately differs region to region: wheat
        in a uniform monoculture disperses far less around its local centroid
        than wheat in a fragmented smallholder landscape.  A global null is set
        by whichever landscape contributes the most slices, and every region
        that is legitimately more variable then looks anomalous *as a whole*.

        Measured on clean synthetic data whose regions differ only in their
        legitimate spread, with a global per-class null::

            uniform    (sigma 0.15)   0.00 % flagged
            uniform    (sigma 0.18)   0.00 %
            mixed      (sigma 0.25)   0.29 %
            mixed      (sigma 0.28)   1.17 %
            fragmented (sigma 0.40)   3.50 %
            fragmented (sigma 0.45)   4.12 %

        — a pure regional false-positive gradient, landing hardest in exactly
        the landscapes that are hardest to verify on a basemap.  Conditioning
        the null on the region flattens it to 0.29-0.67 % everywhere and cuts
        the overall rate from 1.51 % to 0.49 %.

        *null_region_level* is the H3 resolution of that region key: the slice
        cell's parent at this level is written to ``h3_null_region`` and used as
        a null conditioner.  Level 1 (~610,000 km2, ~880 km across) holds ~50 L3
        cells, enough slices to estimate a null for a common class while staying
        far more homogeneous than a global pool.  Use 2 for tighter locality
        where density supports it, or ``None`` to disable the region key.

        *null_shrink_k* controls how much locality is actually spent.  The
        dominant effect is localisation itself — measured at 3, 5, 12 and 25
        slices per region, a local null gave roughly a third the clean
        false-positive rate of a pooled one (0.7-0.8 % vs 2.0-2.4 %) at every
        support level.  Shrinkage is *insurance*, not the source of the gain: a
        null estimated from a couple of slices is noisy, and on real geography,
        where regions straddle hexagon boundaries and some groups fall back,
        an unshrunk local null did misbehave.  Each local null is therefore
        shrunk toward the global one by
        ``w = n_slices / (n_slices + null_shrink_k)``, so a region with 30
        slices sits at w = 0.86 and one with 3 sits at w = 0.38.  There is no
        threshold to trip over and sparse regions degrade smoothly.

        Be clear that this is a **trade, not a free win**.  Averaged over five
        seeds on regions of differing legitimate spread::

            10 % contamination   recall 0.890 -> 0.790, precision 0.901 -> 0.969
            20 % contamination   recall 0.806 -> 0.710, precision 0.977 -> 0.996
            clean data           regional FP spread 3.85 % -> 0.78 %

        A good share of what the pooled null was "finding" was the regional
        bias rather than real errors, which is why precision rises as recall
        falls.  The direction of the recall effect is scenario-dependent — it
        goes the other way where tight regions dominate the collection.  Where
        regions genuinely do *not* differ, conditioning costs ~2 % of recall and
        still lowers the false-positive rate.  Add your own agro-ecological-zone or year column to
        *null_extra_keys* if you have one — it is a better region proxy than a
        hexagon.
    null_scale_estimator
        How the per-slice dispersion feeding the cross-slice null is measured.
        ``"left_tail"`` (default) uses ``median - q25``, i.e. only the clean
        left half of the distance distribution.  Label errors push distances
        right, so the classic MAD is inflated by them — and when contamination
        is present in *every* slice of a class the null itself becomes
        inflated and the gate stops firing.  Measured recall at 40 % slice
        contamination: 0.009 with ``"mad"``, 0.42 with ``"left_tail"``, at a
        matched clean false-positive rate.  Use ``"mad"`` only for ablations.
    abs_combine
        How the per-metric absolute z-scores are combined: ``"min"`` (default,
        conservative — a point must look anomalous on *both* the centroid
        distance and the kNN distance), ``"max"`` (higher recall), or
        ``"mean"``.
    min_scoring_slice_size
        Slices below this are not scored.  Their rows now receive the explicit
        ``unscored`` flag state instead of ``normal``, so "we did not look" is
        distinguishable from "we looked and it is fine".  Previously this was a
        hard-coded module constant of 50 that silently disagreed with
        *min_slice_size*.
    quality_gate, strict_quality
        Validate the embeddings before scoring and quarantine degenerate ones
        as ``unscorable``.  A zero-norm embedding gets ``cosine_distance = 1.0``
        — the maximum possible — so failed inference or an all-cloud time
        series was previously *guaranteed* to be flagged.  *strict_quality*
        additionally hard-errors on mixed ``model_hash`` and on H3 cells that
        disagree with the coordinates.
    time_col
        Optional temporal key (year / season).  Reference data spans many
        years and the embeddings are season-specific, so a 2018 sample in a
        cell dominated by 2021 samples is distant for phenological rather than
        label reasons.  When given, this column joins the slice and context
        keys so points are only ever compared within the same period, and a
        ``time_minority_frac`` diagnostic is emitted.
    apply_confidence_fusion_to_output, purity_veto
        Fold the kNN label-purity and alt-class-margin signals into the shipped
        ``confidence_nonoutlier`` (previously they were computed at real cost,
        written to ``confidence_alt``, and then discarded), and cap escalation
        for points whose neighbourhood agrees with their label.
    """
    group_cols = list(group_cols or [])
    # Auxiliary-signal context: geographic by default (see section 7).
    context_group_cols = list(context_group_cols or [])
    label_cols = _as_label_levels(label_domain)
    label_col = label_cols[0]  # keep existing logic anchored to level-0

    # Normalize h3_level into a list; determine if adaptive mode is active
    if isinstance(h3_level, (list, tuple)):
        h3_levels = [int(x) for x in h3_level]
        adaptive_h3 = len(h3_levels) > 1
    else:
        h3_levels = [int(h3_level)]
        adaptive_h3 = False

    # Only enforce when mapping_file is not provided
    if mapping_file is None:
        if isinstance(label_domain, (list, tuple)):
            raise ValueError(
                "Hierarchical label_domain requires mapping_file "
                "(labels provided by Excel or JSON, or an in-memory CLASS_MAPPINGS dict)."
            )
        if label_domain not in {"ewoc_code", "finetune_class", "balancing_class"}:
            raise ValueError("label_domain must be 'ewoc_code' or 'finetune_class'")

    # ------------------------------------------------------------------
    # 1. Load embeddings from DuckDB (or use pre-supplied data)
    # ------------------------------------------------------------------
    con: Optional[duckdb.DuckDBPyConnection] = None
    if embeddings_df is not None:
        df, embed_cols = embeddings_df
        print(f"[anomaly] Using pre-supplied embeddings: {len(df):,} rows")
    else:
        print("[anomaly] Connecting DuckDB and loading cached embeddings...")
        con = duckdb.connect(embeddings_db_path)
        df, embed_cols = _load_embeddings(
            con, group_cols, restrict_model_hash,
            extra_cols=[time_col] if time_col else None,
        )

    print(f"[anomaly] Loaded {len(df):,} rows from embeddings_cache")
    if df.empty:
        if con is not None:
            con.close()
        raise ValueError(
            "No rows loaded from embeddings_cache. Check model_hash or DB path."
        )

    # ------------------------------------------------------------------
    # 1b. Input sanity: encoder consistency and H3/coordinate agreement
    # ------------------------------------------------------------------
    # Distances between vectors from two different encoders are meaningless,
    # and the whole spatial-slicing premise rests on h3_l3_cell being right.
    # Both were previously assumed rather than checked.
    if quality_gate:
        assert_single_model_hash(
            df, restrict_model_hash=restrict_model_hash, strict=strict_quality
        )
        try:
            mismatch = assert_h3_matches_coordinates(df, strict=strict_quality)
            if mismatch:
                print(
                    f"[anomaly] H3/coordinate mismatch on {mismatch:.2%} of sampled rows"
                )
        except ValueError:
            if con is not None:
                con.close()
            raise

    # ------------------------------------------------------------------
    # 2. Incremental mode — skip already-processed sample_ids
    # ------------------------------------------------------------------
    existing_df_full: Optional[gpd.GeoDataFrame] = None

    if skip_existing_samples:
        df, existing_df_full, existing_ids = _handle_incremental_mode(
            df, output_samples_path, con
        )
        if df.empty:
            print("[anomaly] All samples already processed. Returning existing results...")
            if con is not None:
                con.close()
            return existing_df_full, None

    # ------------------------------------------------------------------
    # 3. Validation & column setup
    # ------------------------------------------------------------------
    missing_group_cols = [c for c in group_cols if c not in df.columns]
    if missing_group_cols:
        if con is not None:
            con.close()
        raise ValueError(
            f"Requested group_cols not found in loaded data: {missing_group_cols}"
        )

    # For adaptive mode, we use the finest level as the reference for
    # debug filtering; the actual adaptive assignment happens after
    # class mapping + embedding preparation (section 5b).
    # For fixed mode, we use the single level as before.
    _finest_h3_level = max(h3_levels)  # finest = highest resolution number
    h3_level_name = "effective_h3_cell" if adaptive_h3 else f"h3_l{h3_levels[0]}_cell"

    if not adaptive_h3:
        _fixed_level = h3_levels[0]
        if _fixed_level != 3:
            df[h3_level_name] = df["h3_l3_cell"].apply(
                lambda h: h3.cell_to_parent(h, _fixed_level)
            )
        else:
            df[h3_level_name] = df["h3_l3_cell"]

    if df["ewoc_code"].dtype != np.int64:
        df["ewoc_code"] = pd.to_numeric(df["ewoc_code"], errors="coerce").astype("Int64")

    if debug:
        print(
            "[DEBUG] Running in debug mode: restricting to small sample of data, "
            "only loading 10 H3 cells..."
        )
        # Use finest level for debug cell sampling
        if adaptive_h3:
            _debug_col = f"_h3_l{_finest_h3_level}_dbg"
            df[_debug_col] = df["h3_l3_cell"].apply(
                lambda h: h3.cell_to_parent(h, _finest_h3_level)
            ) if _finest_h3_level != 3 else df["h3_l3_cell"]
            sample_cells = df[_debug_col].unique()[:10].tolist()
            df = df[df[_debug_col].isin(sample_cells)]
            df = df.drop(columns=[_debug_col], errors="ignore")
        else:
            sample_cells = df[h3_level_name].unique()[:10].tolist()
            df = df[df[h3_level_name].isin(sample_cells)]

    # ------------------------------------------------------------------
    # 4. Class mapping
    # ------------------------------------------------------------------
    df = _apply_class_mapping(
        df,
        map_to_finetune=map_to_finetune,
        mapping_file=mapping_file,
        label_cols=label_cols,
        class_mappings_name=class_mappings_name,
        con=con,
    )

    # ------------------------------------------------------------------
    # 4b. Split out skip_classes rows — they bypass all scoring
    # ------------------------------------------------------------------
    skip_classes = list(skip_classes or [])
    df_skipped: pd.DataFrame = pd.DataFrame()
    if skip_classes:
        skip_mask = df[label_col].astype(str).isin([str(c) for c in skip_classes])
        df_skipped = df[skip_mask].copy()
        df = df[~skip_mask].copy()
        print(
            f"[anomaly] skip_classes {skip_classes}: held aside "
            f"{len(df_skipped):,} rows, processing {len(df):,} rows."
        )

    # ------------------------------------------------------------------
    # 5. Prepare embedding vectors & drop NaN labels
    # ------------------------------------------------------------------
    print("[anomaly] Preparing embeddings array...")
    embed_array = df[embed_cols].to_numpy(dtype=np.float32)
    df["embedding"] = [row for row in embed_array]
    # Drop raw embedding_0..embedding_127 columns early (we keep only df["embedding"])
    df = df.drop(columns=embed_cols, errors="ignore")

    _require_label_columns(df, label_cols)

    # ------------------------------------------------------------------
    # 5a. Embedding quality gate
    # ------------------------------------------------------------------
    # Degenerate vectors are not "very anomalous samples", they are missing
    # data.  _cosine_similarity returns 0.0 for a zero-norm vector, which makes
    # cosine_distance 1.0 — the maximum attainable — so a failed inference was
    # previously guaranteed to top its slice and be flagged.  Quarantine them.
    df_unscorable: pd.DataFrame = pd.DataFrame()
    if quality_gate:
        df, df_unscorable, qreport = validate_embeddings(df, embedding_col="embedding")
        if qreport.n_rejected:
            print(f"[anomaly] Embedding quality gate: {qreport}")
            print(qreport.to_frame().to_string(index=False))
        else:
            print(f"[anomaly] Embedding quality gate: all {qreport.n_total:,} rows OK")

    # ------------------------------------------------------------------
    # 5a-ii. Rows whose ewoc_code is absent from the legend
    # ------------------------------------------------------------------
    # These used to be silently dropped by dropna(), which meant they never
    # received anomaly columns — so every subsequent `--mode update` run
    # rediscovered them as "unscored" and re-expanded the impact zone, and the
    # incremental pathway never converged.  Hold them aside explicitly and give
    # them the terminal `unmapped` state at the end.
    count_before_drop = len(df)
    unmapped_mask = df[label_cols].isna().any(axis=1)
    df_unmapped = df[unmapped_mask].copy()
    df = df[~unmapped_mask].copy()
    if len(df_unmapped):
        codes = (
            df_unmapped["ewoc_code"].astype(str).value_counts().head(15)
            if "ewoc_code" in df_unmapped.columns
            else pd.Series(dtype=int)
        )
        print(
            f"[anomaly] {len(df_unmapped):,} of {count_before_drop:,} rows have no "
            f"mapping to {label_cols} — held aside as 'unmapped' (NOT silently "
            "dropped). This is a legend-coverage gap, not a data quirk."
        )
        if len(codes):
            print(f"[anomaly] Most frequent unmapped ewoc_codes:\n{codes.to_string()}")
    print(f"[anomaly] count_after_drop: {len(df):,}")

    label_col = label_cols[0]

    # ------------------------------------------------------------------
    # 5a-iii. Temporal key
    # ------------------------------------------------------------------
    # Embeddings are season-specific and the reference collection spans many
    # years, so comparing a 2018 sample against a 2021 neighbourhood measures
    # phenology, not label error.  When a time column is available it joins
    # the slice key so points are only compared within the same period.
    time_keys: list[str] = []
    if time_col:
        if time_col in df.columns:
            df[time_col] = df[time_col].astype(str)
            time_keys = [time_col]
            print(
                f"[anomaly] Temporal control ON: '{time_col}' joins the slice and "
                f"context keys ({df[time_col].nunique():,} distinct periods)"
            )
        else:
            print(
                f"[anomaly] WARNING: time_col='{time_col}' is not present in the "
                "loaded embeddings, so TEMPORAL CONTROL IS INACTIVE for this run. "
                "The standard embeddings_cache schema carries no temporal column "
                "(sample_id, model_hash, ref_id, ewoc_code, h3_l3_cell, lat, lon, "
                "embedding_*), so it must be added when the cache is built — or "
                "the embeddings passed in via embeddings_df=. Scoring will "
                "compare across periods, which inflates distances for samples "
                "from minority years."
            )

    slice_keys = [*group_cols, h3_level_name, label_col, *time_keys]

    # ------------------------------------------------------------------
    # 5b. Adaptive H3 level assignment (if h3_level is a list)
    # ------------------------------------------------------------------
    if adaptive_h3:
        print(
            f"[anomaly] Adaptive H3 mode: levels {h3_levels} "
            f"(finest→coarsest), min_slice_size={min_slice_size}"
        )
        if max_slice_size is not None:
            print(f"[anomaly] Max slice size cap: {max_slice_size:,}")
        df = assign_adaptive_h3_level(
            df,
            h3_levels=h3_levels,
            label_col=label_col,
            group_cols=[*group_cols, *time_keys],
            min_slice_size=min_slice_size,
            max_slice_size=max_slice_size,
        )
        # h3_level_name is already "effective_h3_cell" for adaptive mode
        # Update slice_keys to use the effective cell
        slice_keys = [*group_cols, h3_level_name, label_col, *time_keys]

    # ------------------------------------------------------------------
    # 6. Merge small slices
    # ------------------------------------------------------------------
    if merge_small_slice:
        _n_slices_before_merge = df.groupby(slice_keys).ngroups
        print(
            f"[anomaly] Merging small slices (min_size={min_slice_size})... "
            f"[{_n_slices_before_merge:,} slices before merge]"
        )
        df = merge_small_slices(
            df,
            min_size=min_slice_size,
            label_col=label_col,
            h3_level_name=h3_level_name,
            group_cols=[*group_cols, *time_keys],
            max_iterations=max_merge_iterations,
        )
        _n_slices_after_merge = df.groupby(slice_keys).ngroups
        print(f"[anomaly] After merge: {_n_slices_after_merge:,} slices")
    else:
        print("[anomaly] Skipping merge_small_slices for coarse H3 level")
        df["context_h3_cell"] = df[h3_level_name].astype(str)
        df["merge_steps"] = np.uint8(0)

    # ------------------------------------------------------------------
    # 7. Hierarchical ref-class assignment + context centroid metrics
    # ------------------------------------------------------------------
    df = _add_hierarchical_ref_outlier_class(
        df,
        label_cols=label_cols,
        group_cols=[*group_cols, *time_keys],
        h3_level_name=h3_level_name,
        min_slice_size=min_slice_size,
        out_ref_class_col="ref_outlier_class",
        out_ref_level_col="ref_outlier_level",
        out_ref_group_n_col="ref_group_n",
    )

    # adding context centroid metrics
    #
    # IMPORTANT: context metrics are computed on ``context_h3_cell`` — the
    # *pre-merge* cell — not on the post-merge scoring cell.  Merging is decided
    # per (cell, label), so after it the post-merge cell is no longer a
    # geographic neighbourhood: maize may have moved to a different cell than
    # wheat, and "the other classes near me" became an arbitrary label-dependent
    # set.  The alt-class margin and kNN purity were therefore not measuring
    # what their docstrings claimed.
    print("[anomaly] Computing context centroid metrics...")
    context_cell = "context_h3_cell" if "context_h3_cell" in df.columns else h3_level_name
    # The context deliberately does NOT include group_cols.
    #
    # A *slice* is per-dataset (group_cols=["ref_id"]) so that one dataset's
    # labelling convention cannot contaminate another's reference cloud.  But
    # the *context* answers "what else is on the ground around this point?",
    # and that question is geographic: restricting it to the same ref_id makes
    # the auxiliary evidence collapse for single-crop datasets — every point
    # trivially has kNN purity 1.0 and there is no alternative class centroid to
    # measure a margin against.  With the group_cols default now ["ref_id"],
    # keeping group_cols in the context would have silently disabled the purity
    # and margin votes across a large part of the collection.
    context_cols = [*context_group_cols, context_cell, *time_keys]
    df = add_alt_class_centroid_metrics(
        df,
        label_col=label_col,
        context_cols=context_cols,
        embedding_col="embedding",
    )

    # Temporal-minority diagnostic: how rare is this sample's period inside its
    # own context?  A high value means its distance is likely driven by
    # phenology rather than by a wrong label.
    if time_keys:
        tk = time_keys[0]
        # The denominator MUST be the numerator's grouping minus the time key
        # (i.e. context_group_cols + context cell), NOT group_cols.  Using
        # group_cols=["ref_id"] here while the numerator used the geographic
        # context made the two counts refer to different populations, so the
        # "fraction" could go negative whenever several datasets shared a cell.
        spatial_context_cols = [*context_group_cols, context_cell]
        ctx_size = df.groupby(spatial_context_cols, dropna=False)[tk].transform("size")
        same_period = df.groupby(context_cols, dropna=False)[tk].transform("size")
        df["time_minority_frac"] = (
            1.0 - (same_period / ctx_size.replace(0, np.nan))
        ).astype(np.float32)

    # ------------------------------------------------------------------
    # 8. Scoring
    # ------------------------------------------------------------------
    print("[anomaly] Scoring slices...")

    if len(label_cols) > 1:
        # Hierarchical scoring path
        print(f"[anomaly] Hierarchical label_domain enabled: {label_cols}")
        scored_df = score_slices_hierarchical(
            df,
            label_cols=label_cols,
            group_cols=[*group_cols, *time_keys],
            h3_level_name=h3_level_name,
            min_slice_size=min_slice_size,
            norm_percentiles=norm_percentiles,
            max_full_pairwise_n=max_full_pairwise_n,
            ref_level_col="ref_outlier_level",
            ref_class_col="ref_outlier_class",
            centroid_mode=centroid_mode,
            centroid_trim=centroid_trim,
            min_scoring_slice_size=min_scoring_slice_size,
        )
    else:
        # Single-level scoring path
        print("[anomaly] Computing per-slice centroids...")
        centroids = compute_slice_centroids(
            df,
            label_col=label_col,
            h3_level_name=h3_level_name,
            group_cols=[*group_cols, *time_keys],
            centroid_mode=centroid_mode,
            centroid_trim=centroid_trim,
        )

        df_with_centroid = df.merge(
            centroids,
            on=slice_keys,
            how="left",
        )

        def _score_group(g: pd.DataFrame) -> pd.DataFrame:
            g = g.copy()
            g["slice_n"] = len(g)
            if len(g) < int(min_scoring_slice_size):
                # Too small to score.  NaN (not 0.0) plus scored=False, so the
                # row ends up as an explicit `unscored` flag rather than being
                # reported as `normal` — "we did not look" must not read as
                # "we looked and it is fine".
                g = g[[c for c in g.columns if "embedding" not in c]]
                for c in _SCORE_COLS:
                    g[c] = np.nan
                g["scored"] = False
                return g

            out = compute_scores_for_slice(
                g,
                centroid=g["centroid"].iloc[0],
                norm_percentiles=norm_percentiles,
                max_full_pairwise_n=max_full_pairwise_n,
                force_knn=False,
                knn_k=10,
                centroid_mode=centroid_mode,
                centroid_trim=centroid_trim,
            )
            out["scored"] = True
            return out

        from tqdm import tqdm as tqdm_cls

        # sort=True keeps group iteration order deterministic across runs.
        groups = list(df_with_centroid.groupby(slice_keys, sort=True))
        results = []
        with tqdm_cls(groups, desc="Scoring slices", unit="slice") as pbar:
            for key, group in pbar:
                label_val = key[-1] if isinstance(key, tuple) else key
                n_pts = len(group)
                pbar.set_postfix_str(f"{n_pts:,} pts | {label_val}", refresh=False)
                results.append(_score_group(group))

        scored_df = pd.concat(results, ignore_index=True)

    # Drop embedding columns to save memory
    scored_df = scored_df.drop(columns=embed_cols, errors="ignore")
    # scored_df = scored_df.drop(columns=["embedding", "base_embedding"], errors="ignore")
    scored_df = scored_df.drop(columns=["base_embedding"], errors="ignore")

    if "scored" not in scored_df.columns:
        scored_df["scored"] = True
    n_unscored = int((~scored_df["scored"].fillna(False).astype(bool)).sum())
    if n_unscored:
        print(
            f"[anomaly] {n_unscored:,} rows ({n_unscored / max(len(scored_df), 1):.1%}) "
            f"sit in slices below min_scoring_slice_size={min_scoring_slice_size} and "
            "cannot be scored — they will be reported as 'unscored', not 'normal'."
        )

    # ------------------------------------------------------------------
    # 8b. Absolute-scale calibration
    # ------------------------------------------------------------------
    # Convert the raw distances into z-scores against a null pooled ACROSS
    # slices of the same class.  This is what makes a flag mean the same thing
    # everywhere: without it, every slice is min-max normalised onto [0, 1] and
    # therefore yields the same proportion of "suspects" whether it is clean or
    # 30 % mislabelled.
    # Region key for the null.  Derived from the slice cell AFTER merging, so a
    # merged slice is attributed to the region it actually sits in.
    if null_region_level is not None and h3_level_name in scored_df.columns:
        def _parent_at(cell):
            try:
                c = str(cell)
                if h3.get_resolution(c) <= int(null_region_level):
                    return c
                return h3.cell_to_parent(c, int(null_region_level))
            except Exception:
                return None

        _uniq = pd.unique(scored_df[h3_level_name].astype(str))
        _map = {c: _parent_at(c) for c in _uniq}
        scored_df["h3_null_region"] = scored_df[h3_level_name].astype(str).map(_map)
        _n_bad = int(scored_df["h3_null_region"].isna().sum())
        if _n_bad:
            print(
                f"[anomaly] {_n_bad:,} rows have no derivable L{null_region_level} "
                "region key; their null falls back to the global one."
            )

    null_keys = [label_col, *(list(null_extra_keys) if null_extra_keys else [])]
    null_keys = [k for k in null_keys if k in scored_df.columns]

    n_scorable = int(scored_df["scored"].fillna(False).astype(bool).sum())
    if n_scorable == 0:
        # Every slice was below min_scoring_slice_size. That is a legitimate
        # (if uninformative) outcome — typically a very sparse region — so it
        # must not crash the run. Emit NaN evidence; nothing can be flagged.
        print(
            "[anomaly] No slice is large enough to score — skipping calibration. "
            "All rows will be reported as 'unscored'."
        )
        scored_df["abs_z"] = np.nan
        scored_df["cos_abs_z"] = np.nan
        scored_df["neighbour_abs_z"] = np.nan
        scored_df["null_scale_sigma"] = np.nan
    else:
        print(f"[anomaly] Calibrating absolute scale (null keys: {null_keys}) ...")
        null_ref = compute_null_reference(
            scored_df,
            null_keys=null_keys,
            slice_key_cols=slice_keys,
            min_slice_n=max(int(min_scoring_slice_size), 30),
            scored_mask_col="scored",
            scale_estimator=null_scale_estimator,
            shrink_k=null_shrink_k,
        )
        scored_df = add_absolute_scores(
            scored_df, null_ref, null_keys=null_keys, combine=abs_combine
        )
        # How localised was the null in practice?  A high fallback rate means
        # the region level is too fine for this collection's density.
        if len(null_keys) > 1:
            _own = null_ref[null_ref.get("__is_global__", False) != True]  # noqa: E712
            _have = (
                set(map(tuple, _own[null_keys].astype(str).to_numpy()))
                if len(_own) else set()
            )
            _rk = list(map(tuple, scored_df[null_keys].astype(str).to_numpy()))
            _fb = sum(1 for k in _rk if k not in _have)
            print(
                f"[anomaly] Null localised on {null_keys}: {len(_own):,} local "
                f"nulls; {_fb / max(len(_rk), 1):.1%} of rows fell back to the "
                "global null (coarsen null_region_level if this is high)."
            )
        _z_equiv = suggest_abs_z_threshold(
            scored_df, target_flag_fraction=0.02, scored_mask_col="scored"
        )
        # Report exceedance over the SCORED population only; unscored rows
        # carry NaN abs_z and counting them deflated the printed percentage.
        _sc = scored_df["scored"].fillna(False).to_numpy(dtype=bool)
        _z_sc = scored_df.loc[_sc, "abs_z"].to_numpy(dtype="float64")
        # NaN abs_z (degenerate null) counts as "does not exceed".
        with np.errstate(invalid="ignore"):
            _over = float(np.mean(_z_sc >= abs_z_k)) if _z_sc.size else float("nan")
        print(
            f"[anomaly] abs_z gate = {abs_z_k} sigma "
            f"(a flat 2% budget would correspond to z = {_z_equiv:.2f}); "
            f"{_over:.2%} of scored rows exceed the gate before the within-slice test"
        )

    # ------------------------------------------------------------------
    # 9. Flagging
    # ------------------------------------------------------------------
    print(f"[anomaly] Flagging anomalies (mode={threshold_mode})...")
    flagged_df, summary_df = flag_anomalies(
        scored_df,
        label_col=label_col,
        h3_level_name=h3_level_name,
        group_cols=[*group_cols, *time_keys],
        threshold_mode=threshold_mode,
        percentile_q=percentile_q,
        mad_k=mad_k,
        abs_threshold=abs_threshold,
        fdr_alpha=fdr_alpha,
        min_flagged_per_slice=min_flagged_per_slice,
        max_flagged_fraction=max_flagged_fraction,
        abs_z_col="abs_z",
        abs_z_k=abs_z_k,
        require_absolute=require_absolute,
        scored_mask_col="scored",
    )
    _n_flagged = int(flagged_df["flagged"].sum())
    print(
        f"[anomaly] Flagged {_n_flagged:,} / {len(flagged_df):,} "
        f"({_n_flagged / max(len(flagged_df), 1):.2%})"
        + ("" if require_absolute else "  [absolute gate DISABLED — relative only]")
    )

    # ------------------------------------------------------------------
    # 10. Confidence scoring
    # ------------------------------------------------------------------
    print("[anomaly] Computing robust confidence for flagged points...")

    flagged_df = add_confidence_from_score(
        flagged_df,
        score_col="mean_score",
        out_col="confidence",
        # Gate by the slice-level flag so that the continuous
        # confidence_nonoutlier is consistent with the discrete anomaly_flag:
        # samples the slice threshold did NOT flag keep confidence 1.0 instead
        # of being penalised purely for being the highest-ranked point of an
        # otherwise-clean slice.  Set gate_confidence_by_flag=False to restore
        # the legacy (ungated, rank-driven) behaviour.
        flagged_col="flagged" if gate_confidence_by_flag else None,
    )

    # Never down-weight a row we could not score.
    unscored_mask = ~flagged_df["scored"].fillna(False).to_numpy(dtype=bool)
    flagged_df.loc[unscored_mask, "confidence"] = 1.0

    # ------------------------------------------------------------------
    # 11. kNN label purity + confidence fusion
    # ------------------------------------------------------------------
    # Purity is computed on the PRE-merge context cell (see section 7).
    print("[anomaly] Computing kNN label purity for flagged points...")
    flagged_df = add_knn_label_purity_for_flagged(
        df_all=df,              # this df still has embeddings
        flagged_df=flagged_df,  # from flag_anomalies
        label_col=label_col,
        context_cols=context_cols,
        embedding_col="embedding",
        purity_knn_k=10,
        cap_sqrt_k=50,
    )

    print("[anomaly] Applying confidence fusion...")
    flagged_df = apply_confidence_fusion(flagged_df)  # produces confidence_alt

    # Fold the auxiliary signals into the SHIPPED confidence.
    #
    # Previously `confidence_alt` was computed here — at the cost of a kNN per
    # context group — and then never read again: run_pipeline renamed only
    # `confidence` to `confidence_nonoutlier`, and the CLI kept just that
    # column.  The margin and purity evidence therefore had exactly zero
    # influence on the output.  Wire it through.
    if apply_confidence_fusion_to_output and "confidence_alt" in flagged_df.columns:
        flagged_df["confidence_base"] = flagged_df["confidence"].astype(np.float32)
        flagged_df["confidence"] = flagged_df["confidence_alt"].astype(np.float32)

    # ------------------------------------------------------------------
    # 12. Anomaly categorization
    # ------------------------------------------------------------------
    flagged_df = _assign_anomaly_categories(
        flagged_df,
        abs_z_suspect=abs_z_suspect,
        abs_z_candidate=abs_z_candidate,
        purity_veto=purity_veto,
        require_absolute=require_absolute,
    )

    # ------------------------------------------------------------------
    # 12b. Slice-trust gating
    # ------------------------------------------------------------------
    # The detector measures distance to a within-slice reference, which is only
    # meaningful where the (here un-finetuned) embedding geometry actually
    # separates classes.  Estimate a per-context trust score and use it to (a)
    # attenuate confidence penalties and (b) downgrade anomaly categories in
    # slices whose geometry we cannot trust — so untrustworthy slices inject
    # fewer / softer flags instead of silent noise.
    if apply_slice_trust:
        print("[anomaly] Computing slice-trust (embedding separability) gate...")
        trust_df = compute_slice_trust(
            df[["sample_id", label_col, *context_cols, "embedding"]],
            label_col=label_col,
            context_cols=context_cols,
            embedding_col="embedding",
            out_col="slice_trust",
        )
        flagged_df = flagged_df.merge(
            trust_df[["sample_id", "slice_trust"]], on="sample_id", how="left"
        )
        flagged_df["slice_trust"] = flagged_df["slice_trust"].fillna(0.5)
        flagged_df = apply_trust_to_confidence(
            flagged_df,
            conf_col="confidence",
            trust_col="slice_trust",
            flagged_col="flagged",
            min_trust=slice_trust_min,
        )
        flagged_df = downgrade_flags_low_trust(
            flagged_df,
            flag_col="combined_anomaly",
            trust_col="slice_trust",
            suspect_min_trust=slice_trust_min,
            candidate_min_trust=min(2.0 * slice_trust_min, 0.6),
        )

    # ------------------------------------------------------------------
    # 13. Final cleanup & output
    # ------------------------------------------------------------------
    # Re-attach the rows that were held aside before scoring, each with an
    # explicit terminal state.  Previously the quarantined populations either
    # did not exist (quality gate) or were silently dropped (unmapped codes),
    # which is what made the incremental update pathway rediscover them forever.
    def _reattach(extra: pd.DataFrame, flag_value: str) -> None:
        nonlocal flagged_df
        if extra is None or extra.empty:
            return
        extra = extra.drop(columns=embed_cols, errors="ignore")
        extra = extra.drop(columns=["embedding", "base_embedding"], errors="ignore")
        for col in flagged_df.columns:
            if col not in extra.columns:
                extra[col] = np.nan
        # Preserve WHY a row was rejected.  Without this the output says
        # "unscorable" with no way to tell a failed encoder run from a duplicate
        # id — the triage the quality gate exists to enable.
        if "quality_reason" in extra.columns and "quality_reason" not in flagged_df.columns:
            flagged_df["quality_reason"] = pd.NA
        extra["S_anomaly"] = flag_value
        extra["combined_anomaly"] = flag_value
        extra["flagged"] = False
        extra["scored"] = False
        # Never down-weight a sample the detector could not judge.
        extra["confidence"] = 1.0
        extra = extra.reindex(columns=flagged_df.columns)
        flagged_df = pd.concat([flagged_df, extra], axis=0, ignore_index=True)
        print(f"[anomaly] Re-attached {len(extra):,} rows as '{flag_value}'.")

    _reattach(df_unscorable, "unscorable")
    _reattach(df_unmapped, "unmapped")

    # Re-attach skipped-class rows with NaN for all score/outlier columns
    if not df_skipped.empty:
        # Drop raw embedding columns from skipped rows (not needed in output)
        df_skipped = df_skipped.drop(columns=embed_cols, errors="ignore")
        df_skipped = df_skipped.drop(columns=["embedding", "base_embedding"], errors="ignore")
        score_outlier_cols = [
            *_SCORE_COLS,
            "flagged", "flag_threshold", "slice_n", "undersized_slice",
            "ref_outlier_class", "ref_outlier_level", "ref_group_n",
            "self_centroid_dist_ctx", "alt_centroid_dist_ctx",
            "knn_same_label_frac_ctx", "knn_majority_frac_ctx",
            "p_margin", "p_purity",
            "confidence", "confidence_alt",
            "S_anomaly", "combined_anomaly",
        ]
        for col in score_outlier_cols:
            if col not in df_skipped.columns:
                df_skipped[col] = np.nan
        # Ensure schema alignment before concat to avoid FutureWarning on all-NA columns
        for col in flagged_df.columns:
            if col not in df_skipped.columns:
                df_skipped[col] = np.nan
        # Terminal state, not NaN.  `find_unscored_samples` decides "already
        # handled" from the flag column, and NaN is never terminal — so these
        # rows were rediscovered as unscored on every `--mode update`.  With
        # skip_classes=["ignore"] covering roughly half of CROPTYPE24, that is a
        # large recurring cost, and it only avoided non-convergence because
        # `_discover_domain_impact` happened to filter them by ewoc_code as well.
        df_skipped["S_anomaly"] = "skipped"
        df_skipped["combined_anomaly"] = "skipped"
        df_skipped["flagged"] = False
        df_skipped["scored"] = False
        df_skipped["confidence"] = 1.0
        df_skipped = df_skipped.reindex(columns=flagged_df.columns)
        flagged_df = pd.concat([flagged_df, df_skipped], axis=0, ignore_index=True)
        print(
            f"[anomaly] Re-attached {len(df_skipped):,} skipped-class rows as 'skipped'."
        )

    # Convert float64 → float32 to reduce output size
    for c in flagged_df.select_dtypes(include=["float64"]).columns:
        flagged_df[c] = flagged_df[c].astype(np.float32)

    flagged_df["geometry"] = gpd.points_from_xy(flagged_df["lon"], flagged_df["lat"])
    flagged_gdf = gpd.GeoDataFrame(flagged_df, geometry="geometry", crs="EPSG:4326")

    # Drop extra columns to reduce size
    # ["cosine_distance", "knn_distance", "cos_norm", "knn_norm",
    #              "cos_rank", "knn_rank", "S_rank", "S_rank_min",
    #              "cos_z", "knn_z", "S_z", "mean_score"]
    # Columns dropped from the review output.
    #
    # The evidence behind a flag is deliberately RETAINED now: abs_z, the raw
    # distances, the kNN purity, the alt-class margin and the vote count all
    # survive.  Previously every one of these was dropped here, so a reviewer
    # opening a flagged point on a basemap had no way to see *why* it was
    # flagged and no way to tell a strong flag from a marginal one — which is
    # precisely the audit that was needed to catch the false positives.
    drop_cols = [
        "centroid",
        "cos_norm",
        "knn_norm",
        "cos_rank",
        "knn_rank",
        "p_margin",
        "p_purity",
    ]
    embed_raw = [c for c in flagged_gdf.columns if c.startswith("embedding_")]
    drop_cols += embed_raw
    flagged_gdf = flagged_gdf.drop(columns=drop_cols, errors="ignore")

    # Incremental merge (if resuming from existing output)
    if skip_existing_samples and existing_df_full is not None:
        flagged_gdf = _merge_with_existing(flagged_gdf, existing_df_full)
    flagged_gdf.rename(columns={'confidence': 'confidence_nonoutlier', 'combined_anomaly' : 'anomaly_flag'}, inplace=True)
    # Write to disk (optional — can be skipped in incremental/update mode)
    if write_outputs:
        _write_outputs(
            flagged_gdf, summary_df, slice_keys,
            output_samples_path, output_summary_path,
        )

    # ------------------------------------------------------------------
    # 14. Dataset-level rollup
    # ------------------------------------------------------------------
    # Point-wise flags dilute systematic error: a whole ref_id digitised
    # against the wrong legend shows up as a slightly raised flag rate across
    # thousands of points rather than as one actionable finding — and once the
    # errors approach half of a class in a region, point-wise geometry cannot
    # resolve them at all.  robust_extensions.aggregate_parcel_scores existed
    # for exactly this and was never called from the pipeline.  Emit the rollup
    # so a systematically-off dataset is visible at the level it occurs.
    if "ref_id" in flagged_gdf.columns:
        try:
            _roll = (
                flagged_gdf.assign(
                    _is_flagged=flagged_gdf["anomaly_flag"].isin(
                        ["flagged", "suspect", "candidate"]
                    )
                )
                .groupby("ref_id")
                .agg(
                    n=("sample_id", "size"),
                    flag_rate=("_is_flagged", "mean"),
                    median_abs_z=("abs_z", "median"),
                )
                .reset_index()
                .sort_values("flag_rate", ascending=False)
            )
            _median_rate = float(_roll["flag_rate"].median())
            _suspicious = _roll[_roll["flag_rate"] > max(3.0 * _median_rate, 0.05)]
            print(
                f"[anomaly] Dataset rollup: median ref_id flag rate "
                f"{_median_rate:.2%}; {len(_suspicious)} ref_id(s) above 3x that."
            )
            if len(_suspicious):
                print(_suspicious.head(10).to_string(index=False))
            if output_summary_path and write_outputs:
                _roll.to_parquet(
                    Path(output_summary_path).with_name(
                        Path(output_summary_path).stem + "_by_ref_id.parquet"
                    ),
                    index=False,
                )
        except Exception as exc:  # diagnostics must never break a run
            print(f"[anomaly] Dataset rollup skipped: {type(exc).__name__}: {exc}")

    if con is not None:
        con.close()
    return flagged_gdf, summary_df


# ===================================================================
# CLI
# ===================================================================
# class_mappings_name answers "how to map", while label_domain answers "what to slice on after mapping"

if __name__ == "__main__":
    out_folder = Path(
        "/path/to/TestFolder/Outliers/"
        "h3l2_mad_3_maxrank_groupRefId_sqrtk_norm2_98_new"
    )
    out_folder.mkdir(parents=True, exist_ok=True)
    run_pipeline(
        embeddings_db_path=(
            "/projects/worldcereal/data/cached_embeddings/"
            "embeddings_cache_LANDCOVER10_geo.duckdb"
        ),
        restrict_model_hash=None,
        label_domain="finetune_class",
        map_to_finetune=True,
        class_mappings_name="LANDCOVER10",
        # Adaptive H3: try level 3 first (finest), fall back to 2 then 1
        # for sparse regions.  Use a single int (e.g. h3_level=2) for
        # fixed-level mode (original behaviour).
        h3_level=[3, 2, 1],
        group_cols=["ref_id"],
        min_slice_size=100,
        max_slice_size=1000,  # cap to prevent runaway merging in dense areas
        merge_small_slice=True,
        threshold_mode="mad",
        percentile_q=0.96,
        mad_k=4.0,
        abs_threshold=None,
        fdr_alpha=0.05,
        min_flagged_per_slice=None,
        max_flagged_fraction=None,
        max_full_pairwise_n=0,  # disable full pairwise matrix calculation
        norm_percentiles=(2.0, 98.0),
        output_samples_path=str(
            out_folder
            / "outliers_h3l2_mad_3_maxrank_groupRefId_ranked_sqrtk_norm2_98_new.parquet"
        ),
        output_summary_path=str(
            out_folder
            / "outliers_h3l2_mad_3_maxrank_groupRefId_summary_sqrtk_norm2_98_new.parquet"
        ),
        debug=False,
    )

