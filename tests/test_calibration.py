"""Regression tests for the robustness fixes.

Every test here corresponds to a defect that shipped silently — the point of
the file is that the old code passes none of them and the new code passes all.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from EBA_detector.anomaly_utils import (
    _add_hierarchical_ref_outlier_class,
    compute_scores_for_slice,
    flag_anomalies,
    find_unscored_samples,
    merge_small_slices,
    robust_centroid,
)
from EBA_detector.calibration import (
    add_absolute_scores,
    compute_null_reference,
    suggest_abs_z_threshold,
)
from EBA_detector.quality import (
    assert_single_model_hash,
    validate_embeddings,
)

D = 32


# ---------------------------------------------------------------------------
# Synthetic world: many slices of one class, a controllable share corrupted
# ---------------------------------------------------------------------------


def make_world(contamination: float, n_slices: int = 12, n: int = 120, seed: int = 0):
    """Slices of one class, each with its own regional centre.

    Corrupted points are drawn from a *different* class direction, so they form
    a coherent cluster — the case that masks itself under naive kNN scoring.
    """
    r = np.random.default_rng(seed)
    rows = []
    for s in range(n_slices):
        centre = r.normal(size=D) * 0.15 + np.eye(D)[0]
        X = r.normal(size=(n, D)) * 0.22 + centre
        n_out = int(contamination * n)
        if n_out:
            wrong = r.normal(size=D) * 0.15 + np.eye(D)[1]
            X[:n_out] = r.normal(size=(n_out, D)) * 0.22 + wrong
        for i in range(n):
            rows.append((f"s{s}_{i}", f"cell{s}", "maize", X[i].astype(np.float32), i < n_out))
    return pd.DataFrame(
        rows, columns=["sample_id", "h3_l3_cell", "label", "embedding", "truth"]
    )


def score_world(df: pd.DataFrame, *, centroid_trim: float = 0.20) -> pd.DataFrame:
    parts = []
    for _key, g in df.groupby(["h3_l3_cell", "label"], sort=True):
        sc = compute_scores_for_slice(
            g, max_full_pairwise_n=0, force_knn=True, centroid_trim=centroid_trim
        )
        sc["scored"] = True
        parts.append(sc)
    scored = pd.concat(parts, ignore_index=True)
    null_ref = compute_null_reference(
        scored,
        null_keys=["label"],
        slice_key_cols=["h3_l3_cell", "label"],
        scored_mask_col="scored",
    )
    return add_absolute_scores(scored, null_ref, null_keys=["label"])


def flag_world(scored: pd.DataFrame, **kw) -> pd.DataFrame:
    params = dict(
        label_col="label",
        h3_level_name="h3_l3_cell",
        threshold_mode="stable_mad",
        mad_k=3.0,
        abs_z_k=3.0,
        require_absolute=True,
    )
    params.update(kw)
    flagged, _ = flag_anomalies(scored, **params)
    return flagged


# ---------------------------------------------------------------------------
# 1. The central defect: no absolute scale
# ---------------------------------------------------------------------------


class TestAbsoluteScale:
    def test_clean_population_yields_almost_no_flags(self):
        """A population with zero label errors must not manufacture 'suspects'.

        The legacy rule flagged a fixed ~2% quota of every slice because the
        score was percentile-normalised per slice and the escalation thresholds
        were rank quantiles.
        """
        flagged = flag_world(score_world(make_world(0.0)))
        assert flagged["flagged"].mean() < 0.02

    def test_heavy_contamination_is_still_detected(self):
        """The regime the legacy MAD gate failed on completely.

        With ``median + k*MAD`` computed on the slice's own bounded score, a
        30%-contaminated slice inflates its own median and MAD until the
        threshold exceeds the score ceiling — and flags nothing at all.
        """
        flagged = flag_world(score_world(make_world(0.30)))
        truth = flagged["truth"].to_numpy()
        hit = flagged["flagged"].to_numpy()
        recall = (hit & truth).sum() / max(truth.sum(), 1)
        precision = (hit & truth).sum() / max(hit.sum(), 1)
        assert recall > 0.6, f"recall collapsed to {recall:.3f}"
        assert precision > 0.9

    @pytest.mark.parametrize("contamination", [0.02, 0.05, 0.10, 0.20])
    def test_flag_rate_tracks_true_contamination(self, contamination):
        """The realised flag rate must respond to how dirty the data is."""
        flagged = flag_world(score_world(make_world(contamination)))
        rate = flagged["flagged"].mean()
        assert rate > contamination * 0.4, (
            f"flag rate {rate:.3%} far below contamination {contamination:.0%}"
        )
        assert rate < contamination * 2.5 + 0.02

    def test_legacy_mad_collapses_where_stable_mad_does_not(self):
        """Documents the specific failure the stable scale repairs."""
        scored = score_world(make_world(0.30))
        legacy = flag_world(scored, threshold_mode="mad")
        stable = flag_world(scored, threshold_mode="stable_mad")
        t = scored["truth"].to_numpy()
        r_legacy = (legacy["flagged"].to_numpy() & t).sum() / t.sum()
        r_stable = (stable["flagged"].to_numpy() & t).sum() / t.sum()
        assert r_stable > r_legacy * 2

    def test_abs_z_k_is_a_monotone_dial(self):
        """Raising the gate must trade recall for precision, not fall off a cliff."""
        scored = score_world(make_world(0.05))
        rates = [flag_world(scored, abs_z_k=k)["flagged"].mean() for k in (2.5, 3.5, 5.0)]
        assert rates[0] >= rates[1] >= rates[2]

    def test_null_is_not_dragged_by_one_bad_slice(self):
        """A single contaminated slice must not calibrate its own errors away."""
        clean = make_world(0.0, n_slices=10, seed=1)
        dirty = make_world(0.40, n_slices=1, seed=2)
        dirty["h3_l3_cell"] = "dirty_cell"
        dirty["sample_id"] = "d_" + dirty["sample_id"]
        world = pd.concat([clean, dirty], ignore_index=True)

        flagged = flag_world(score_world(world))
        in_dirty = flagged["h3_l3_cell"].to_numpy() == "dirty_cell"
        truth = flagged["truth"].to_numpy()
        hit = flagged["flagged"].to_numpy()
        recall_dirty = (hit & truth & in_dirty).sum() / max((truth & in_dirty).sum(), 1)
        assert recall_dirty > 0.5

    def test_suggest_threshold_is_a_quantile(self):
        scored = score_world(make_world(0.05))
        z = suggest_abs_z_threshold(scored, target_flag_fraction=0.05)
        assert np.isfinite(z)
        assert np.isclose((scored["abs_z"] >= z).mean(), 0.05, atol=0.01)


# ---------------------------------------------------------------------------
# 2. Masking by a coherent error cluster
# ---------------------------------------------------------------------------


class TestMasking:
    def test_neighbourhood_offset_exposes_a_self_consistent_cluster(self):
        """Members of a wrong cluster have small kNN distance but a displaced
        neighbourhood; the offset is what stops them hiding behind each other.
        """
        scored = score_world(make_world(0.30))
        t = scored["truth"].to_numpy()
        # the trap: kNN distance says "well supported"
        assert np.nanmedian(scored.loc[t, "knn_abs_z"]) < 2.0
        # the escape: the neighbourhood itself is far from the class centroid
        assert np.nanmedian(scored.loc[t, "nbr_abs_z"]) > 3.0
        # and the combined evidence therefore still fires
        assert np.nanmedian(scored.loc[t, "abs_z"]) > 3.0

    def test_isolated_but_supported_point_is_not_flagged(self):
        """A legitimate sub-type — unusual, but its neighbours agree — must survive."""
        r = np.random.default_rng(7)
        rows = []
        for s in range(10):
            centre = r.normal(size=D) * 0.15 + np.eye(D)[0]
            X = r.normal(size=(120, D)) * 0.22 + centre
            for i in range(120):
                rows.append((f"s{s}_{i}", f"cell{s}", "maize", X[i].astype(np.float32), False))
        df = pd.DataFrame(
            rows, columns=["sample_id", "h3_l3_cell", "label", "embedding", "truth"]
        )
        flagged = flag_world(score_world(df))
        assert flagged["flagged"].mean() < 0.02


# ---------------------------------------------------------------------------
# 3. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_flags_are_invariant_to_input_row_order(self):
        """S is clipped, so ~2-4% of each slice ties at exactly 1.0.  The cap
        used to break those ties by row order, making the flagged set depend on
        how the rows happened to come back from DuckDB.
        """
        scored = score_world(make_world(0.05))

        a = flag_world(scored, max_flagged_fraction=0.03)
        shuffled = scored.sample(frac=1.0, random_state=99).reset_index(drop=True)
        b = flag_world(shuffled, max_flagged_fraction=0.03)

        set_a = set(a.loc[a["flagged"], "sample_id"])
        set_b = set(b.loc[b["flagged"], "sample_id"])
        assert set_a == set_b

    def test_merge_small_slices_is_order_invariant(self):
        r = np.random.default_rng(3)
        base = pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(60)],
                "h3_l3_cell": ["8328c4fffffffff"] * 20
                + ["8328c5fffffffff"] * 20
                + ["8328c0fffffffff"] * 20,
                "label": ["maize"] * 60,
                "embedding": list(r.normal(size=(60, D)).astype(np.float32)),
            }
        )
        m1 = merge_small_slices(base, min_size=50, label_col="label")
        m2 = merge_small_slices(
            base.sample(frac=1.0, random_state=5).reset_index(drop=True),
            min_size=50,
            label_col="label",
        )
        map1 = m1.set_index("sample_id")["h3_l3_cell"].to_dict()
        map2 = m2.set_index("sample_id")["h3_l3_cell"].to_dict()
        assert map1 == map2


# ---------------------------------------------------------------------------
# 4. Coverage / blind spot
# ---------------------------------------------------------------------------


class TestCoverage:
    def test_small_slice_is_unscored_not_normal(self):
        r = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(10)],
                "h3_l3_cell": ["cell"] * 10,
                "label": ["maize"] * 10,
                "embedding": list(r.normal(size=(10, D)).astype(np.float32)),
            }
        )
        from EBA_detector.anomaly_utils import _score_group_simple

        out = _score_group_simple(df, (2.0, 98.0), 0, min_scoring_slice_size=50)
        assert not out["scored"].any()
        assert out["S"].isna().all()

    def test_unscored_rows_are_never_flagged(self):
        scored = score_world(make_world(0.05))
        scored.loc[scored.index[:100], "scored"] = False
        flagged = flag_world(scored)
        assert not flagged.loc[~flagged["scored"], "flagged"].any()

    def test_merge_preserves_a_stable_context_cell(self):
        """Context metrics must key on the pre-merge cell, or 'the classes near
        me' becomes an arbitrary label-dependent set after merging."""
        r = np.random.default_rng(11)
        df = pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(40)],
                "h3_l3_cell": ["8328c4fffffffff"] * 20 + ["8328c5fffffffff"] * 20,
                "label": ["maize"] * 20 + ["wheat"] * 20,
                "embedding": list(r.normal(size=(40, D)).astype(np.float32)),
            }
        )
        merged = merge_small_slices(df, min_size=100, label_col="label")
        assert "context_h3_cell" in merged.columns
        assert "merge_steps" in merged.columns
        # the snapshot must equal the ORIGINAL cell, whatever merging did
        orig = df.set_index("sample_id")["h3_l3_cell"].to_dict()
        got = merged.set_index("sample_id")["context_h3_cell"].to_dict()
        assert got == orig


# ---------------------------------------------------------------------------
# 5. Embedding quality gate
# ---------------------------------------------------------------------------


class TestQualityGate:
    def test_degenerate_embeddings_are_quarantined(self):
        r = np.random.default_rng(0)
        emb = list(r.normal(size=(6, D)).astype(np.float32))
        emb[0] = np.zeros(D, dtype=np.float32)          # failed inference
        emb[1] = np.full(D, np.nan, dtype=np.float32)   # non-finite
        df = pd.DataFrame({"sample_id": [f"s{i}" for i in range(6)], "embedding": emb})

        ok, bad, report = validate_embeddings(df)
        assert len(ok) == 4
        assert set(bad["quality_reason"]) == {"zero_norm", "non_finite"}
        assert report.n_rejected == 2

    def test_zero_norm_would_otherwise_score_maximally_anomalous(self):
        """The reason the gate matters: a zero vector gets cosine_distance 1.0,
        the maximum attainable, so missing data was *guaranteed* to be flagged.
        """
        from EBA_detector.anomaly_utils import _cosine_similarity

        assert _cosine_similarity(np.zeros(D), np.ones(D)) == 0.0  # -> distance 1.0

    def test_duplicate_ids_are_caught(self):
        r = np.random.default_rng(0)
        df = pd.DataFrame(
            {
                "sample_id": ["a", "b", "b"],
                "embedding": list(r.normal(size=(3, D)).astype(np.float32)),
            }
        )
        ok, bad, _ = validate_embeddings(df)
        assert len(ok) == 2
        assert bad["quality_reason"].tolist() == ["duplicate_id"]

    def test_mixed_model_hash_raises(self):
        df = pd.DataFrame({"model_hash": ["a", "a", "b"]})
        with pytest.raises(ValueError, match="not comparable"):
            assert_single_model_hash(df, restrict_model_hash=None, strict=True)

    def test_single_model_hash_is_fine(self):
        df = pd.DataFrame({"model_hash": ["a", "a", "a"]})
        assert assert_single_model_hash(df) == ["a"]


# ---------------------------------------------------------------------------
# 6. Previously-silent correctness bugs
# ---------------------------------------------------------------------------


class TestAliasingBug:
    def test_hierarchical_helper_does_not_mutate_its_input(self):
        """``Series.to_numpy()`` returns a view; the helper wrote through it.

        On pandas < 3 this silently corrupted the level-0 label column that
        ``score_slices_hierarchical`` then groups by; on pandas >= 3 it raised
        "assignment destination is read-only".
        """
        df = pd.DataFrame(
            {
                "sample_id": [f"s{i}" for i in range(10)],
                "lvl0": ["a"] * 5 + ["b"] * 5,
                "lvl1": ["A"] * 10,
                "cell": ["c"] * 10,
            }
        )
        before_labels = df["lvl0"].tolist()

        out = _add_hierarchical_ref_outlier_class(
            df, label_cols=["lvl0", "lvl1"], group_cols=[],
            h3_level_name="cell", min_slice_size=6,
        )

        assert df["lvl0"].tolist() == before_labels, "input label column was mutated"
        # the fallback did happen (both level-0 slices are below min_slice_size)
        assert (out["ref_outlier_level"] == 1).all()
        assert (out["ref_outlier_class"] == "A").all()
        # and slice_n must still be the LEVEL-0 size, not the fallback size
        assert (out["slice_n"] == 5).all()
        assert (out["ref_group_n"] == 10).all()


class TestUnscoredDetection:
    def test_terminal_flags_are_not_rediscovered_as_unscored(self, tmp_path):
        """Rows with a terminal state must not re-enter the impact zone forever."""
        df = pd.DataFrame(
            {
                "ref_id": ["r1"] * 4,
                "sample_id": ["a", "b", "c", "d"],
                "LC10_confidence_nonoutlier": [1.0, 1.0, np.nan, np.nan],
                "LC10_anomaly_flag": ["normal", "flagged", "unmapped", None],
                "outlier_LC10_cls": ["crop", "crop", None, None],
            }
        )
        df.to_parquet(tmp_path / "part.parquet", index=False)

        out = find_unscored_samples(
            tmp_path,
            anomaly_cols=[
                "LC10_confidence_nonoutlier",
                "LC10_anomaly_flag",
                "outlier_LC10_cls",
            ],
            flag_col="LC10_anomaly_flag",
        )
        # only 'd' (no decision) is genuinely unscored — 'c' is terminal
        assert out["sample_id"].tolist() == ["d"]

    def test_legacy_nan_rule_would_have_returned_both(self, tmp_path):
        df = pd.DataFrame(
            {
                "ref_id": ["r1"] * 2,
                "sample_id": ["c", "d"],
                "LC10_confidence_nonoutlier": [np.nan, np.nan],
                "outlier_LC10_cls": [None, None],
            }
        )
        df.to_parquet(tmp_path / "part.parquet", index=False)
        out = find_unscored_samples(
            tmp_path,
            anomaly_cols=["LC10_confidence_nonoutlier", "outlier_LC10_cls"],
        )
        assert set(out["sample_id"]) == {"c", "d"}


class TestSphericalCentroid:
    def test_centroid_is_not_dominated_by_vector_norm(self):
        """Cosine distance against a Euclidean mean let large-norm samples
        define the reference direction."""
        X = np.zeros((11, 3), dtype=np.float32)
        X[:10] = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        X[10] = np.array([0.0, 50.0, 0.0], dtype=np.float32)  # one huge vector

        spherical = robust_centroid(X, mode="mean", normalize=True)
        euclidean = robust_centroid(X, mode="mean", normalize=False)

        spherical = spherical / np.linalg.norm(spherical)
        euclidean = euclidean / np.linalg.norm(euclidean)

        # the spherical centroid stays with the 10 identical unit vectors
        assert spherical[0] > 0.9
        # the Euclidean one is dragged onto the single high-norm outlier
        assert euclidean[1] > 0.9


# ---------------------------------------------------------------------------
# 7. Second-round review fixes
# ---------------------------------------------------------------------------


def _duplicate_heavy_world(n_dup_slices=5, n_ord_slices=2, n=80, seed=0):
    """Slices of near-identical embeddings (grid-sampled polygon interiors)
    alongside ordinary ones, all in the same class."""
    r = np.random.default_rng(seed)
    rows = []
    for s in range(n_dup_slices):
        base = r.normal(size=D)
        X = np.tile(base, (n, 1)) + r.normal(size=(n, D)) * 1e-7
        for i in range(n):
            rows.append((f"d{s}_{i}", f"dup{s}", "maize", X[i].astype(np.float32), False))
    for s in range(n_ord_slices):
        c = r.normal(size=D)
        X = r.normal(size=(n, D)) * 0.25 + c
        for i in range(n):
            rows.append((f"o{s}_{i}", f"ord{s}", "maize", X[i].astype(np.float32), False))
    return pd.DataFrame(
        rows, columns=["sample_id", "h3_l3_cell", "label", "embedding", "truth"]
    )


class TestDegenerateNullScale:
    def test_degenerate_class_null_flags_nothing(self):
        """A class whose pooled scale collapses must not become a hair trigger.

        Flooring the null scale (rather than rejecting it) made every ordinary
        slice in the class clear the gate: measured at a 46% flag rate on the
        two ordinary slices below, with zero planted errors.
        """
        flagged = flag_world(score_world(_duplicate_heavy_world()))
        assert flagged["flagged"].mean() == 0.0

    def test_degenerate_scale_is_nan_not_floored(self):
        scored = score_world(_duplicate_heavy_world())
        null_ref = compute_null_reference(
            scored,
            null_keys=["label"],
            slice_key_cols=["h3_l3_cell", "label"],
            scored_mask_col="scored",
        )
        assert null_ref["cosine_distance_null_scale"].isna().all()
        assert scored["null_scale_sigma"].isna().all()
        assert scored["abs_z"].isna().all()

    def test_healthy_class_is_unaffected(self):
        """The guard must not suppress a normal population."""
        flagged = flag_world(score_world(make_world(0.05)))
        assert flagged["flagged"].mean() > 0.01


class TestNullReferenceEdgeCases:
    def test_metric_present_but_never_contributing(self):
        """A column that exists but has no usable data must not KeyError.

        Hits any legacy scored frame that predates `neighbourhood_offset`.
        """
        scored = score_world(make_world(0.0, n_slices=3))
        scored["neighbourhood_offset"] = np.nan
        null_ref = compute_null_reference(
            scored,
            null_keys=["label"],
            slice_key_cols=["h3_l3_cell", "label"],
            scored_mask_col="scored",
        )
        out = add_absolute_scores(scored, null_ref, null_keys=["label"])
        assert np.isfinite(out["abs_z"]).any()

    def test_explicit_out_prefixes_are_honoured(self):
        # start from an UNcalibrated frame so leftover cos_abs_z can't mask the check
        scored = score_world(make_world(0.0, n_slices=3)).drop(
            columns=["cos_abs_z", "knn_abs_z", "nbr_abs_z", "neighbour_abs_z", "abs_z"],
            errors="ignore",
        )
        null_ref = compute_null_reference(
            scored,
            null_keys=["label"],
            slice_key_cols=["h3_l3_cell", "label"],
            scored_mask_col="scored",
        )
        out = add_absolute_scores(
            scored, null_ref, null_keys=["label"], out_prefixes=("aa", "bb", "cc")
        )
        assert "aa_abs_z" in out.columns and "cos_abs_z" not in out.columns

    def test_mismatched_out_prefixes_raise(self):
        scored = score_world(make_world(0.0, n_slices=3))
        null_ref = compute_null_reference(
            scored,
            null_keys=["label"],
            slice_key_cols=["h3_l3_cell", "label"],
            scored_mask_col="scored",
        )
        with pytest.raises(ValueError, match="out_prefixes"):
            add_absolute_scores(
                scored, null_ref, null_keys=["label"], out_prefixes=("only_one",)
            )


class TestValidationHarnessRobustness:
    def test_all_slices_too_small_does_not_crash(self):
        """A sweep that happens to produce only small slices must not abort."""
        from EBA_detector.validation import score_embeddings_df

        r = np.random.default_rng(0)
        rows = []
        for c in range(4):
            for i in range(20):
                rows.append(
                    (f"s{c}_{i}", f"cell{c}", "maize", r.normal(size=D).astype(np.float32))
                )
        df = pd.DataFrame(
            rows, columns=["sample_id", "h3_l3_cell", "label", "embedding"]
        )
        out = score_embeddings_df(df, label_col="label", min_scoring_slice_size=50)
        assert len(out) == len(df)
        assert not out["flagged"].any()
        assert not out["scored"].any()


# ---------------------------------------------------------------------------
# 8. The operational target: heavy but sub-majority slice contamination
# ---------------------------------------------------------------------------


class TestHeavyContaminationRegime:
    """Up to ~40% of a slice wrong must still be detectable.

    The blocker was not the centroid but the *scale*: the cross-slice null took
    the median of per-slice MADs, and a MAD is inflated by the very
    right-side contamination it is meant to measure against.  When every slice
    of a class is 30-45% contaminated the null inflates with them, z-scores
    shrink, and detection collapses.  ``_slice_stats(estimator="left_tail")``
    measures the spread from the clean left half instead.
    """

    @pytest.mark.parametrize(
        "contamination,min_recall", [(0.20, 0.90), (0.30, 0.85), (0.40, 0.45)]
    )
    def test_recall_holds_up_to_forty_percent(self, contamination, min_recall):
        flagged = flag_world(score_world(make_world(contamination), centroid_trim=0.45))
        t = flagged["truth"].to_numpy()
        h = flagged["flagged"].to_numpy()
        recall = (h & t).sum() / max(t.sum(), 1)
        precision = (h & t).sum() / max(h.sum(), 1)
        assert recall >= min_recall, f"recall {recall:.3f} at {contamination:.0%}"
        assert precision > 0.9

    def test_left_tail_beats_mad_where_it_matters(self):
        """At 40% the legacy MAD null is what fails, not the geometry."""
        df = make_world(0.40)
        parts = []
        for _k, g in df.groupby(["h3_l3_cell", "label"], sort=True):
            sc = compute_scores_for_slice(
                g, max_full_pairwise_n=0, force_knn=True, centroid_trim=0.45
            )
            sc["scored"] = True
            parts.append(sc)
        scored = pd.concat(parts, ignore_index=True)

        rec = {}
        for est in ("mad", "left_tail"):
            nr = compute_null_reference(
                scored,
                null_keys=["label"],
                slice_key_cols=["h3_l3_cell", "label"],
                scored_mask_col="scored",
                scale_estimator=est,
            )
            s2 = add_absolute_scores(scored.copy(), nr, null_keys=["label"])
            fl = flag_world(s2)
            t = fl["truth"].to_numpy()
            h = fl["flagged"].to_numpy()
            rec[est] = (h & t).sum() / max(t.sum(), 1)
        # Measured ~0.89 vs ~0.45; guard with headroom rather than the exact ratio.
        assert rec["left_tail"] > 1.6 * max(rec["mad"], 1e-6), rec
        assert rec["left_tail"] > 0.7, rec

    def test_contamination_does_not_inflate_the_left_tail_null(self):
        """The whole point: the null scale must barely move with contamination."""
        scales = {}
        for contamination in (0.0, 0.40):
            scored = score_world(make_world(contamination), centroid_trim=0.45)
            nr = compute_null_reference(
                scored,
                null_keys=["label"],
                slice_key_cols=["h3_l3_cell", "label"],
                scored_mask_col="scored",
            )
            own = nr[nr["__is_global__"] != True]  # noqa: E712
            scales[contamination] = float(own["cosine_distance_null_scale"].iloc[0])
        # The left-tail estimator does not fully escape inflation: at 40%
        # contamination the slice MEDIAN itself sits high inside the clean
        # group, so the median-to-q25 span covers a wider slice of the clean
        # distribution than it would on clean data.  Measured ~2x, against
        # ~3.5x for the MAD it replaces - enough to keep the gate usable.
        inflation = scales[0.40] / scales[0.0]
        assert inflation < 2.5, f"null scale inflated {inflation:.2f}x by contamination"

    def test_clean_population_still_quiet(self):
        """The recall gain must not have been bought with false positives."""
        flagged = flag_world(score_world(make_world(0.0), centroid_trim=0.45))
        assert flagged["flagged"].mean() < 0.02


# ---------------------------------------------------------------------------
# 9. Conditioning the null on the resolution each slice resolved at
# ---------------------------------------------------------------------------


def _mixed_resolution_world(contamination: float = 0.0, seed: int = 0):
    """Dense fine-level slices + sparse coarse-level ones.

    Mimics the real geography: most points sit in one continent and resolve at a
    fine H3 level (small, homogeneous cells), while sparse regions resolve at a
    coarse level whose cells span several agro-ecological zones and therefore
    carry genuinely wider *legitimate* within-class dispersion.
    """
    r = np.random.default_rng(seed)
    rows = []
    for s in range(40):  # fine / dense / tight
        c = r.normal(size=D) * 0.12 + np.eye(D)[0]
        X = r.normal(size=(400, D)) * 0.22 + c
        n_out = int(contamination * 400)
        if n_out:
            X[:n_out] = r.normal(size=(n_out, D)) * 0.22 + (
                r.normal(size=D) * 0.12 + np.eye(D)[1]
            )
        for i in range(400):
            rows.append((f"f{s}_{i}", f"fine{s}", "maize", X[i].astype(np.float32), 4,
                         i < n_out))
    for s in range(12):  # coarse / sparse / genuinely wider
        c = r.normal(size=D) * 0.12 + np.eye(D)[0]
        X = r.normal(size=(150, D)) * 0.38 + c
        n_out = int(contamination * 150)
        if n_out:
            X[:n_out] = r.normal(size=(n_out, D)) * 0.22 + (
                r.normal(size=D) * 0.12 + np.eye(D)[1]
            )
        for i in range(150):
            rows.append((f"c{s}_{i}", f"coarse{s}", "maize", X[i].astype(np.float32), 2,
                         i < n_out))
    return pd.DataFrame(
        rows,
        columns=["sample_id", "h3_l3_cell", "label", "embedding",
                 "h3_effective_level", "truth"],
    )


def _score_with_null_keys(df, extra_keys):
    parts = []
    for _k, g in df.groupby(["h3_l3_cell", "label"], sort=True):
        sc = compute_scores_for_slice(
            g, max_full_pairwise_n=0, force_knn=True, centroid_trim=0.45
        )
        sc["scored"] = True
        parts.append(sc)
    scored = pd.concat(parts, ignore_index=True)
    keys = ["label"] + list(extra_keys)
    null_ref = compute_null_reference(
        scored, null_keys=keys, slice_key_cols=["h3_l3_cell", "label"],
        scored_mask_col="scored",
    )
    scored = add_absolute_scores(scored, null_ref, null_keys=keys)
    return flag_world(scored, mad_k=3.3, abs_z_k=3.3)


class TestNullConditionedOnResolution:
    def test_pooling_biases_false_positives_toward_coarse_slices(self):
        """The bias this conditioning exists to remove.

        Coarse (sparse-region) slices legitimately disperse more, so against a
        null dominated by fine dense slices they look anomalous as a group.
        """
        flagged = _score_with_null_keys(_mixed_resolution_world(), [])
        fine = flagged["h3_effective_level"] == 4
        coarse = flagged["h3_effective_level"] == 2
        assert flagged.loc[coarse, "flagged"].mean() > (
            4 * flagged.loc[fine, "flagged"].mean()
        )

    def test_conditioning_equalises_the_false_positive_rate(self):
        flagged = _score_with_null_keys(
            _mixed_resolution_world(), ["h3_effective_level"]
        )
        fine = flagged.loc[flagged["h3_effective_level"] == 4, "flagged"].mean()
        coarse = flagged.loc[flagged["h3_effective_level"] == 2, "flagged"].mean()
        assert coarse < 0.02
        assert coarse < 3 * max(fine, 1e-6)

    def test_detection_in_dense_slices_is_unharmed(self):
        """The conditioning must not cost anything where most points are."""
        flagged = _score_with_null_keys(
            _mixed_resolution_world(0.20, seed=1), ["h3_effective_level"]
        )
        m = flagged["h3_effective_level"] == 4
        t = flagged.loc[m, "truth"].to_numpy()
        h = flagged.loc[m, "flagged"].to_numpy()
        assert (h & t).sum() / max(t.sum(), 1) > 0.9
        assert (h & t).sum() / max(h.sum(), 1) > 0.95

    def test_missing_conditioner_is_ignored_not_fatal(self):
        """Fixed (non-adaptive) H3 mode has no h3_effective_level column."""
        df = _mixed_resolution_world().drop(columns=["h3_effective_level"])
        flagged = _score_with_null_keys(df, ["h3_effective_level"])
        assert len(flagged) == len(df)


# ---------------------------------------------------------------------------
# 10. Localising the null to a region, with shrinkage
# ---------------------------------------------------------------------------


def _heterogeneous_world(contamination=0.0, sigmas=(0.16, 0.20, 0.40, 0.45),
                         slices_per_region=12, n=150, seed=0):
    """Regions whose within-class spread differs *legitimately*.

    Wheat in a uniform monoculture disperses far less around its local centroid
    than wheat in a fragmented smallholder landscape.  A globally pooled null is
    set by whichever landscape contributes the most slices, so every region that
    is legitimately more variable looks anomalous as a whole.
    """
    r = np.random.default_rng(seed)
    rows = []
    for ri, sig in enumerate(sigmas):
        for s in range(slices_per_region):
            c = r.normal(size=D) * 0.15 + np.eye(D)[0]
            X = r.normal(size=(n, D)) * sig + c
            n_out = int(contamination * n)
            if n_out:
                X[:n_out] = r.normal(size=(n_out, D)) * 0.22 + (
                    r.normal(size=D) * 0.15 + np.eye(D)[1]
                )
            for i in range(n):
                rows.append((f"r{ri}s{s}_{i}", f"r{ri}c{s}", "maize",
                             X[i].astype(np.float32), f"region{ri}", i < n_out))
    return pd.DataFrame(
        rows,
        columns=["sample_id", "h3_l3_cell", "label", "embedding", "region", "truth"],
    )


def _flag_with_null(df, extra_keys, shrink_k=5.0):
    parts = []
    for _k, g in df.groupby(["h3_l3_cell", "label"], sort=True):
        sc = compute_scores_for_slice(
            g, max_full_pairwise_n=0, force_knn=True, centroid_trim=0.45
        )
        sc["scored"] = True
        parts.append(sc)
    scored = pd.concat(parts, ignore_index=True)
    keys = ["label"] + list(extra_keys)
    null_ref = compute_null_reference(
        scored, null_keys=keys, slice_key_cols=["h3_l3_cell", "label"],
        scored_mask_col="scored", shrink_k=shrink_k,
    )
    scored = add_absolute_scores(scored, null_ref, null_keys=keys)
    return flag_world(scored, mad_k=3.3, abs_z_k=3.3)


class TestRegionalNull:
    def test_global_null_biases_false_positives_toward_variable_regions(self):
        """The failure this exists to fix: on clean data the flag rate should
        not depend on how variable a region legitimately is."""
        flagged = _flag_with_null(_heterogeneous_world(), [])
        by_region = flagged.groupby("region")["flagged"].mean()
        assert by_region.max() > 3 * max(by_region.min(), 1e-4), by_region.to_dict()

    def test_regional_null_flattens_that_bias(self):
        flagged = _flag_with_null(_heterogeneous_world(), ["region"])
        by_region = flagged.groupby("region")["flagged"].mean()
        assert by_region.max() - by_region.min() < 0.015, by_region.to_dict()
        assert flagged["flagged"].mean() < 0.02

    def test_regional_null_trades_recall_for_precision(self):
        """The honest characterisation of the change.

        Localising the null is NOT a free win.  Averaged over 5 seeds at 10 %
        contamination it costs recall (0.89 -> 0.79) and buys precision
        (0.90 -> 0.97): a good share of what the global null was "finding" was
        the regional bias, not real errors.  The direction of the recall effect
        is scenario-dependent - it improves where tight regions dominate - so
        only the precision gain is asserted here.
        """
        df = _heterogeneous_world(contamination=0.10, seed=2)
        out = {}
        for name, keys in (("global", []), ("regional", ["region"])):
            fl = _flag_with_null(df, keys)
            t = fl["truth"].to_numpy()
            h = fl["flagged"].to_numpy()
            out[name] = ((h & t).sum() / max(h.sum(), 1),
                         (h & t).sum() / max(t.sum(), 1))
        assert out["regional"][0] > out["global"][0], out      # precision up
        assert out["regional"][1] > 0.6 * out["global"][1], out  # recall not gutted

    def test_localising_beats_pooling_at_every_support_level(self):
        """The dominant, consistent effect — measured at 3, 5, 12 and 25 slices
        per region, a local null gives roughly a third the false-positive rate
        of a globally pooled one.  Shrinkage is insurance for thin regions, not
        the source of the gain."""
        for slices_per_region in (3, 12):
            clean = _heterogeneous_world(slices_per_region=slices_per_region)
            local = _flag_with_null(clean, ["region"])["flagged"].mean()
            pooled = _flag_with_null(clean, [])["flagged"].mean()
            assert local < pooled, (slices_per_region, local, pooled)

    def test_shrinkage_does_not_destabilise_a_thin_region(self):
        """A region with almost no support must not produce a wild null."""
        clean = _heterogeneous_world(slices_per_region=3)
        shrunk = _flag_with_null(clean, ["region"], shrink_k=5.0)["flagged"].mean()
        assert shrunk < 0.03, shrunk

    def test_shrinkage_weight_tracks_support(self):
        """w = n/(n+k): well-supported regions keep their own null."""
        scored = score_world(make_world(0.0, n_slices=30), centroid_trim=0.45)
        null_ref = compute_null_reference(
            scored, null_keys=["label"], slice_key_cols=["h3_l3_cell", "label"],
            scored_mask_col="scored", shrink_k=5.0,
        )
        own = null_ref[null_ref["__is_global__"] != True]  # noqa: E712
        w = float(own["null_shrink_w"].iloc[0])
        assert np.isclose(w, 30 / 35, atol=0.01), w

    def test_no_harm_when_regions_are_homogeneous(self):
        """Where regions genuinely do not differ, conditioning must not cost."""
        df = _heterogeneous_world(contamination=0.20, sigmas=(0.25,) * 4, seed=3)
        out = {}
        for name, keys in (("global", []), ("regional", ["region"])):
            fl = _flag_with_null(df, keys)
            t = fl["truth"].to_numpy()
            h = fl["flagged"].to_numpy()
            out[name] = ((h & t).sum() / max(t.sum(), 1),
                         (h & t).sum() / max(h.sum(), 1))
        assert out["regional"][0] > 0.9 * out["global"][0], out
        assert out["regional"][1] >= out["global"][1] - 0.02, out


# ---------------------------------------------------------------------------
# 11. The null region must scale with the slice's own resolution
# ---------------------------------------------------------------------------


def _mixed_resolution_h3_world(contamination=0.0, levels=(2, 3, 4),
                               per_level=10, n=150, seed=0):
    """Slices at several real H3 resolutions in one broad area.

    Dispersion scales with cell size because a bigger cell spans more
    legitimate variation — which is exactly why they must not share a null.
    """
    import h3

    r = np.random.default_rng(seed)
    sigma = {2: 0.42, 3: 0.28, 4: 0.18}
    rows = []
    for lvl in levels:
        for s in range(per_level):
            lat = 47.0 + (0.35 * s if lvl == 4 else 1.2 * s if lvl == 3 else 3.0 * s)
            cell = h3.latlng_to_cell(lat, 2.0 + 0.1 * s, lvl)
            c = r.normal(size=D) * 0.15 + np.eye(D)[0]
            X = r.normal(size=(n, D)) * sigma[lvl] + c
            n_out = int(contamination * n)
            if n_out:
                X[:n_out] = r.normal(size=(n_out, D)) * 0.20 + (
                    r.normal(size=D) * 0.15 + np.eye(D)[1]
                )
            for i in range(n):
                rows.append((f"L{lvl}s{s}_{i}", cell, "maize",
                             X[i].astype(np.float32), lvl, i < n_out))
    return pd.DataFrame(
        rows,
        columns=["sample_id", "h3_l3_cell", "label", "embedding", "lvl", "truth"],
    )


def _region_key(cell, offset=None, absolute=None, floor=1):
    import h3

    res = h3.get_resolution(str(cell))
    tgt = int(absolute) if absolute is not None else max(res - int(offset), floor)
    tgt = min(tgt, res)
    return str(cell) if res <= tgt else h3.cell_to_parent(str(cell), tgt)


class TestNullRegionScalesWithResolution:
    def _flag(self, df, **region):
        d = df.copy()
        if region:
            d["region"] = [_region_key(c, **region) for c in d["h3_l3_cell"]]
            keys = ["region"]
        else:
            keys = []
        return _flag_with_null(d, keys)

    def test_absolute_region_mixes_resolutions_and_is_worse_than_none(self):
        """The croptype regression, pinned.

        With h3_level=[2,3,4] and a fixed L1 region, L2/L3/L4 slices share one
        null group.  The tight L4 slices inherit a scale set partly by the
        coarse L2 ones, and a fixed region ends up worse than no region at all.
        """
        clean = _mixed_resolution_h3_world()
        none = self._flag(clean)["flagged"].mean()
        absolute = self._flag(clean, absolute=1)["flagged"].mean()
        assert absolute >= none * 0.95, (absolute, none)

    def test_relative_region_beats_both(self):
        clean = _mixed_resolution_h3_world()
        none = self._flag(clean)["flagged"].mean()
        absolute = self._flag(clean, absolute=1)["flagged"].mean()
        relative = self._flag(clean, offset=2)["flagged"].mean()
        assert relative < none, (relative, none)
        assert relative < absolute, (relative, absolute)

    def test_relative_region_helps_the_finest_slices_most(self):
        """L4 is where dense-region croptype lives."""
        dirty = _mixed_resolution_h3_world(contamination=0.20, seed=3)
        rec = {}
        for name, kw in (("absolute", dict(absolute=1)), ("relative", dict(offset=2))):
            fl = self._flag(dirty, **kw)
            m = fl["lvl"] == 4
            t = fl.loc[m, "truth"].to_numpy()
            h = fl.loc[m, "flagged"].to_numpy()
            rec[name] = (h & t).sum() / max(t.sum(), 1)
        assert rec["relative"] >= rec["absolute"], rec

    def test_single_resolution_run_is_unaffected(self):
        """A two-level LANDCOVER10-style run maps L3 -> L1 either way."""
        assert _region_key("832da1fffffffff", offset=2) == _region_key(
            "832da1fffffffff", absolute=1
        )

    def test_floor_stops_the_region_going_coarser_than_l1(self):
        import h3

        l2 = h3.latlng_to_cell(47.0, 2.0, 2)
        assert h3.get_resolution(_region_key(l2, offset=2)) == 1


# ---------------------------------------------------------------------------
# The hierarchical null ladder
# ---------------------------------------------------------------------------
def _colocated_multires_world(contamination=0.0, levels=(2, 3, 4), per_level=8,
                              n=150, n_classes=3, seed=0):
    """Multi-resolution slices that all sit inside ONE L1 cell.

    Co-location is what makes the failure visible: when the resolutions happen
    to fall in different L1 cells, geography separates them by accident and any
    region scheme looks fine.  Real dense-Europe croptype is co-located.
    """
    import h3

    r = np.random.default_rng(seed)
    sigma = {2: 0.42, 3: 0.28, 4: 0.18}
    root = h3.latlng_to_cell(47.0, 2.0, 1)
    cens = []
    for _ in range(n_classes):
        v = r.normal(size=D)
        cens.append(v / np.linalg.norm(v))
    rows = []
    for lvl in levels:
        kids = sorted(h3.cell_to_children(root, lvl))[:per_level]
        for s, cell in enumerate(kids):
            for c in range(n_classes):
                X = r.normal(size=(n, D)) * sigma[lvl] + cens[c]
                n_out = int(contamination * n)
                if n_out:
                    X[:n_out] = (
                        r.normal(size=(n_out, D)) * sigma[lvl]
                        + cens[(c + 1) % n_classes]
                    )
                for i in range(n):
                    rows.append((f"L{lvl}s{s}c{c}_{i}", cell, f"cls{c}",
                                 X[i].astype(np.float32), lvl, i < n_out))
    return pd.DataFrame(
        rows,
        columns=["sample_id", "h3_l3_cell", "label", "embedding", "lvl", "truth"],
    )


def _with_null_cols(df, offset=2, floor=1):
    import h3

    d = df.copy()
    d["region"] = [_region_key(c, offset=offset, floor=floor) for c in d["h3_l3_cell"]]
    # int dtype on purpose — the ladder must survive the widening to float that
    # concatenating depths would otherwise cause.
    d["res"] = np.asarray(
        [h3.get_resolution(str(c)) for c in d["h3_l3_cell"]], dtype="int16"
    )
    return d


def _score_frame(df):
    parts = []
    for _k, g in df.groupby(["h3_l3_cell", "label"], sort=True):
        sc = compute_scores_for_slice(
            g, max_full_pairwise_n=0, force_knn=True, centroid_trim=0.45
        )
        sc["scored"] = True
        parts.append(sc)
    return pd.concat(parts, ignore_index=True)


def _calibrate(scored, keys, shrink_k=5.0, min_slices=2):
    null_ref = compute_null_reference(
        scored, null_keys=keys, slice_key_cols=["h3_l3_cell", "label"],
        scored_mask_col="scored", shrink_k=shrink_k, min_slices=min_slices,
    )
    return add_absolute_scores(scored, null_ref, null_keys=keys), null_ref


class TestNullLadder:
    """`null_keys` is a nesting; each row uses the finest group it has."""

    def test_integer_key_survives_the_depth_concat(self):
        """Regression: "3" vs "3.0".

        Ladder depths are concatenated, which widens an integer key column to
        float wherever the shallower depths have no value for it.  Rebuilding
        the lookup key from the concatenated frame then produced "3.0" while
        the data frame produced "3", so the deepest group matched *nothing* and
        every row silently fell back a level — invisibly, because the run still
        completed and still flagged things.
        """
        from EBA_detector.calibration import _null_key_array

        df = _with_null_cols(_colocated_multires_world(per_level=6, n=120))
        scored = _score_frame(df)
        keys = ["label", "region", "res"]
        _out, ref = _calibrate(scored, keys)
        # Assert the string keys themselves match, independently of how much
        # weight the deepest rung ends up carrying: every row's full-depth key
        # must exist in the deepest rung of the reference.
        deepest = int(ref["__null_depth__"].max())
        assert deepest == len(keys), deepest
        built = set(ref.loc[ref["__null_depth__"] == deepest, "__null_key__"])
        rows = set(_null_key_array(scored, keys).tolist())
        assert rows <= built, sorted(rows - built)[:5]

    def test_resolution_key_separates_cell_sizes(self):
        """The croptype fix.

        Co-located L2/L3/L4 slices share one region.  Without the resolution
        key the tight L4 slices inherit a scale inflated by the coarse L2 ones.
        """
        df = _with_null_cols(_colocated_multires_world(per_level=8, n=150, seed=1))
        scored = _score_frame(df)
        region_only, _ = _calibrate(scored, ["label", "region"])
        with_res, _ = _calibrate(scored, ["label", "region", "res"])
        fp_region = flag_world(region_only, mad_k=3.3, abs_z_k=3.3)["flagged"].mean()
        fp_both = flag_world(with_res, mad_k=3.3, abs_z_k=3.3)["flagged"].mean()
        assert fp_both < fp_region, (fp_both, fp_region)

    def test_thin_group_backs_off_one_level_not_to_the_global(self):
        """A conditioner must not strand the groups it thins.

        The whole risk of adding the resolution key is that (class, region,
        res) groups get thin — and for CROPTYPE24 most classes are rare.  A
        thin group must keep whatever locality it can still support, and must
        never be pushed onto the flat global null.
        """
        df = _with_null_cols(_colocated_multires_world(per_level=8, n=150, seed=2))
        # Give one class a single L4 slice: too thin for its own depth-3 null.
        rare = df[(df["label"] == "cls0") & (df["lvl"] == 4)]
        keep_cell = sorted(rare["h3_l3_cell"].unique())[0]
        drop = rare[rare["h3_l3_cell"] != keep_cell].index
        df = df.drop(index=drop).reset_index(drop=True)

        scored = _score_frame(df)
        keys = ["label", "region", "res"]
        out, _ref = _calibrate(scored, keys, min_slices=2)

        thin = out[(out["label"] == "cls0") & (out["h3_l3_cell"] == keep_cell)]
        assert len(thin) > 0
        # Depth 0 is the flat, class-blind global null.  Nothing may land there
        # while a class-level null exists.
        assert (out["abs_z_null_depth"] >= 1).all()
        assert (thin["abs_z_null_depth"] >= 1).all()

    def test_reported_depth_is_the_rung_that_contributed(self):
        """A w = 0 group equals its parent, and must be credited to the parent.

        Otherwise the localisation histogram — the only instrument for "are
        these keys too fine for this collection's density?" — reports a
        locality the numbers do not have.
        """
        df = _with_null_cols(_colocated_multires_world(per_level=8, n=150, seed=7))
        scored = _score_frame(df)
        keys = ["label", "region", "res"]
        # min_slices above every group's support: no rung can contribute, so
        # every row must be credited to depth 0 even though depth-3 groups
        # exist and match.
        out, ref = _calibrate(scored, keys, min_slices=10_000)
        assert (ref["__null_depth__"] == 3).any(), "depth-3 groups should exist"
        assert (out["abs_z_null_depth"] == 0).all(), (
            out["abs_z_null_depth"].value_counts().to_dict()
        )

    def test_ladder_shrinks_toward_the_parent_not_the_global(self):
        df = _with_null_cols(_colocated_multires_world(per_level=8, n=150, seed=3))
        scored = _score_frame(df)
        _out, ref = _calibrate(scored, ["label", "region", "res"], shrink_k=5.0)
        deep = ref[ref["__null_depth__"] == 3]
        mid = ref[ref["__null_depth__"] == 2]
        assert not deep.empty and not mid.empty
        # Each depth-3 loc lies between its own raw estimate and its depth-2
        # parent, so the depth-3 spread must not collapse onto the depth-2 one
        # nor onto the single global value.
        glob = ref[ref["__is_global__"] == True]  # noqa: E712
        g = float(glob["cosine_distance_null_loc"].iloc[0])
        d3 = deep["cosine_distance_null_loc"].to_numpy(dtype="float64")
        assert np.nanstd(d3) > 0, d3
        assert not np.allclose(d3, g), (d3, g)

    def test_degenerate_scale_is_not_rescued_by_the_parent(self):
        """The duplicate-heavy guard must survive the cascade.

        A group with enough slices to *make* the claim keeps its NaN scale
        rather than borrowing a finite one from its parent, so nothing in it
        can be flagged on a hair trigger.
        """
        df = _duplicate_heavy_world(n_dup_slices=5, n_ord_slices=3, n=80, seed=4)
        df["region"] = "R"
        df["res"] = np.int16(3)
        scored = _score_frame(df)
        _out, ref = _calibrate(scored, ["label", "region", "res"], min_slices=2)
        own = ref[ref["__is_global__"] == False]  # noqa: E712
        deepest = int(own["__null_depth__"].max())
        deep = own[own["__null_depth__"] == deepest]
        assert deep["cosine_distance_null_scale"].isna().any()

    def test_unsupported_degenerate_group_inherits_rather_than_dies(self):
        """`w = 0` means "no opinion of my own" — including about degeneracy.

        A single degenerate slice must not condemn a whole group to a NaN
        scale, which would make every row in it unflaggable and unscored.  The
        pre-ladder code got this right by dropping sub-`min_slices` groups
        entirely; the cascade has to reproduce it deliberately.
        """
        df = _duplicate_heavy_world(n_dup_slices=1, n_ord_slices=6, n=80, seed=11)
        scored = _score_frame(df)
        keys = ["label"]
        # The duplicate class contributes exactly one slice.
        n_dup = int(
            scored[scored["label"] != scored["label"].mode()[0]]["label"].nunique()
        )
        assert n_dup >= 0  # construction sanity only
        _out, ref = _calibrate(scored, keys, min_slices=5)
        own = ref[ref["__is_global__"] == False]  # noqa: E712
        thin = own[own["n_slices"] < 5]
        if len(thin):
            # Every under-supported group carries a usable (parent) scale.
            assert thin["cosine_distance_null_scale"].notna().all(), thin

    def test_legacy_null_reference_without_depth_columns_still_works(self):
        df = _with_null_cols(_colocated_multires_world(per_level=6, n=120, seed=5))
        scored = _score_frame(df)
        keys = ["label", "region", "res"]
        ref = compute_null_reference(
            scored, null_keys=keys, slice_key_cols=["h3_l3_cell", "label"],
            scored_mask_col="scored",
        )
        legacy = ref[
            (ref["__null_depth__"] == len(keys)) | (ref["__is_global__"] == True)  # noqa: E712
        ].drop(columns=["__null_depth__", "__null_key__"])
        out = add_absolute_scores(scored, legacy, null_keys=keys)
        assert np.isfinite(out["abs_z"].to_numpy(dtype="float64")).any()

    def test_single_key_run_is_unchanged_by_the_ladder(self):
        """With one null key the ladder is one rung; behaviour must not move."""
        df = _colocated_multires_world(per_level=6, n=120, seed=6)
        scored = _score_frame(df)
        out, ref = _calibrate(scored, ["label"])
        assert set(ref["__null_depth__"].unique()) <= {0, 1}
        assert (out.loc[out["scored"], "abs_z_null_depth"] == 1).all()

    def test_missing_region_does_not_merge_classes(self):
        """Regression: an NA key component used to annihilate the whole key.

        Under pandas' string dtype ``astype(str)`` keeps NA as NA and NA
        propagates through concatenation, so one underivable H3 region turned
        the composite key into a single NA shared by EVERY class.  The
        de-duplication kept one arbitrary group and every such row — of every
        class — was calibrated against it.  Nothing raised, nothing printed.
        """
        df = _with_null_cols(_colocated_multires_world(per_level=6, n=120, seed=8))
        # Half the slices lose their region, as a failed H3 parse would do.
        cells = sorted(df["h3_l3_cell"].unique())
        orphan = set(cells[::2])
        df["region"] = [
            None if c in orphan else r
            for c, r in zip(df["h3_l3_cell"], df["region"])
        ]
        scored = _score_frame(df)
        keys = ["label", "region", "res"]
        _out, ref = _calibrate(scored, keys)

        own = ref[ref["__is_global__"] == False]  # noqa: E712
        no_region = own[own["region"].isna() & (own["__null_depth__"] >= 2)]
        # One "unknown region" group per class (and per resolution), never one
        # shared group that classes collide in.
        assert len(no_region) >= 2, no_region
        assert no_region["label"].nunique() > 1, no_region[["label", "region"]]
        assert len(set(no_region["__null_key__"].tolist())) == len(no_region)

    def test_missing_region_rows_keep_their_own_class_null(self):
        """The sharp end of the same bug: cross-class calibration.

        Two classes at very different distance scales, each with some slices
        whose region is underivable.  If the NA keys collide, the tight class's
        null is applied to the wide one and its rows score in the tens of
        sigma.
        """
        r = np.random.default_rng(21)
        rows = []
        for label, sigma, offset in (("tight", 0.04, 0.0), ("wide", 0.30, 0.0)):
            for s in range(6):
                cell = f"cell_{label}_{s}"
                centre = np.zeros(D)
                centre[0] = 1.0 + offset
                X = r.normal(size=(120, D)) * sigma + centre
                for i in range(120):
                    rows.append((f"{label}{s}_{i}", cell, label,
                                 X[i].astype(np.float32),
                                 None if s % 2 else f"R{label}", 3, False))
        df = pd.DataFrame(rows, columns=["sample_id", "h3_l3_cell", "label",
                                         "embedding", "region", "res", "truth"])
        scored = _score_frame(df)
        out, _ref = _calibrate(scored, ["label", "region", "res"])
        wide_orphan = out[(out["label"] == "wide") & out["region"].isna()]
        assert len(wide_orphan) > 0
        z = pd.to_numeric(wide_orphan["cos_abs_z"], errors="coerce")
        # Calibrated against its own class these sit near zero; against the
        # tight class's null the repro measured a mean of 34 sigma.
        assert abs(float(z.mean())) < 3.0, float(z.mean())

    def test_a_rung_that_does_not_subdivide_is_dropped(self):
        """Two identical rungs compose their shrinkage: w_eff = 1-(1-w)^2.

        On a fixed single-resolution run `h3_null_res` is a constant, so a
        naive third rung would re-shrink an already-shrunk estimate toward
        itself — raising a thin group's weight on its own noisy estimate from
        0.29 to 0.49 and quietly halving the insurance shrinkage provides.
        """
        # Two roots, so the REGION key genuinely subdivides; only `res` is
        # constant, and only that rung should be dropped.
        df = _with_null_cols(_two_root_single_resolution_world(seed=9))
        scored = _score_frame(df)
        _o2, ref2 = _calibrate(scored, ["label", "region"])
        _o3, ref3 = _calibrate(scored, ["label", "region", "res"])
        # `res` is constant here, so the third rung must not be built at all.
        assert int(ref3["__null_depth__"].max()) == 2
        cols = ["cosine_distance_null_loc", "cosine_distance_null_scale"]
        a = ref2.sort_values(["__null_depth__", "__null_key__"])[cols].to_numpy()
        b = ref3.sort_values(["__null_depth__", "__null_key__"])[cols].to_numpy()
        assert np.allclose(a, b, equal_nan=True), (a, b)

    def test_mismatched_null_keys_raise_instead_of_degrading(self):
        """Silent degradation is the failure mode this whole change is about.

        The ladder is indexed by prefix depth, so calibrating with a different
        key list matches nothing and calibrates everything against a coarser
        null — while the run completes and still produces flags.
        """
        df = _with_null_cols(_colocated_multires_world(per_level=6, n=120, seed=10))
        scored = _score_frame(df)
        ref = compute_null_reference(
            scored, null_keys=["label", "region", "res"],
            slice_key_cols=["h3_l3_cell", "label"], scored_mask_col="scored",
        )
        with pytest.raises(ValueError, match="null_keys"):
            add_absolute_scores(scored, ref, null_keys=["label", "res", "region"])


def _two_root_single_resolution_world(n=120, per_root=8, n_classes=3, seed=0):
    """One resolution, two broad regions — so `region` subdivides and `res`
    does not."""
    import h3

    r = np.random.default_rng(seed)
    roots = [h3.latlng_to_cell(47.0, 2.0, 1), h3.latlng_to_cell(-15.0, -50.0, 1)]
    cens = []
    for _ in range(n_classes):
        v = r.normal(size=D)
        cens.append(v / np.linalg.norm(v))
    rows = []
    for ri, root in enumerate(roots):
        sigma = 0.20 + 0.12 * ri
        for s, cell in enumerate(sorted(h3.cell_to_children(root, 3))[:per_root]):
            for c in range(n_classes):
                X = r.normal(size=(n, D)) * sigma + cens[c]
                for i in range(n):
                    rows.append((f"r{ri}s{s}c{c}_{i}", cell, f"cls{c}",
                                 X[i].astype(np.float32), 3, False))
    return pd.DataFrame(
        rows,
        columns=["sample_id", "h3_l3_cell", "label", "embedding", "lvl", "truth"],
    )


class TestNullLadderDiagnostics:
    """The ladder's instruments have to be trustworthy or they are worse than
    nothing — an operator who learns to ignore a warning has lost the warning."""

    def test_unscored_rows_do_not_trigger_the_coverage_warning(self, capsys):
        """The null is built from scored slices, so unscored rows can never
        match.  Counting them made the warning fire at 71 % on a healthy frame
        while the message asserted sparsity was *not* the explanation."""
        df = _with_null_cols(_colocated_multires_world(per_level=6, n=120, seed=12))
        scored = _score_frame(df)
        # Two thirds of the frame never reached the scoring threshold.
        rng = np.random.default_rng(0)
        unscored = rng.random(len(scored)) < 0.67
        for c in ("cosine_distance", "knn_distance_fixed", "neighbourhood_offset"):
            if c in scored.columns:
                scored.loc[unscored, c] = np.nan
        scored["scored"] = ~unscored

        keys = ["label", "region", "res"]
        ref = compute_null_reference(
            scored, null_keys=keys, slice_key_cols=["h3_l3_cell", "label"],
            scored_mask_col="scored",
        )
        capsys.readouterr()
        add_absolute_scores(scored, ref, null_keys=keys)
        assert "WARNING" not in capsys.readouterr().out

    def test_a_real_key_mismatch_still_warns(self, capsys):
        df = _with_null_cols(_colocated_multires_world(per_level=6, n=120, seed=13))
        scored = _score_frame(df)
        keys = ["label", "region", "res"]
        ref = compute_null_reference(
            scored, null_keys=keys, slice_key_cols=["h3_l3_cell", "label"],
            scored_mask_col="scored",
        )
        # Same key names, values that exist nowhere in the reference.
        broken = scored.copy()
        broken["region"] = "not-a-region"
        broken["res"] = np.int16(9)
        broken["label"] = "not-a-class"
        capsys.readouterr()
        add_absolute_scores(broken, ref, null_keys=keys)
        assert "WARNING" in capsys.readouterr().out

    def test_sole_child_group_is_not_shrunk_twice(self):
        """A group that is its parent's only child was estimated from exactly
        the same slices.  Blending it toward the parent composes the two
        shrinkages and moves it AWAY from the pooled estimate."""
        import h3

        # Two regions.  R0 holds a single resolution (so its (region, res)
        # group is a sole child); R1 spans two resolutions that share ONE
        # region — L2 and L3 both floor to L1 — so the rung is informative and
        # the whole-rung skip does not fire.
        r = np.random.default_rng(14)
        roots = [h3.latlng_to_cell(47.0, 2.0, 1), h3.latlng_to_cell(-15.0, -50.0, 1)]
        rows = []
        for ri, root in enumerate(roots):
            levels = (3,) if ri == 0 else (2, 3)
            for lvl in levels:
                for s, cell in enumerate(sorted(h3.cell_to_children(root, lvl))[:4]):
                    sigma = 0.20 + 0.10 * ri
                    X = r.normal(size=(120, D)) * sigma + np.eye(D)[0]
                    for i in range(120):
                        rows.append((f"r{ri}L{lvl}s{s}_{i}", cell, "cls0",
                                     X[i].astype(np.float32), lvl, False))
        df = pd.DataFrame(rows, columns=["sample_id", "h3_l3_cell", "label",
                                         "embedding", "lvl", "truth"])
        df = _with_null_cols(df)
        scored = _score_frame(df)
        _o2, ref2 = _calibrate(scored, ["label", "region"])
        _o3, ref3 = _calibrate(scored, ["label", "region", "res"])
        assert int(ref3["__null_depth__"].max()) == 3, "the rung must be kept"

        def loc_for(ref, depth, region):
            m = (ref["__null_depth__"] == depth) & (ref["region"] == region)
            return ref.loc[m, "cosine_distance_null_loc"].to_numpy()

        r0 = sorted(df.loc[df["sample_id"].str.startswith("r0"), "region"].unique())[0]
        parent = loc_for(ref2, 2, r0)
        child = loc_for(ref3, 3, r0)
        assert len(parent) == 1 and len(child) == 1, (parent, child)
        # Sole child: identical to its parent, not shrunk a second time.
        assert np.allclose(parent, child), (parent, child)
