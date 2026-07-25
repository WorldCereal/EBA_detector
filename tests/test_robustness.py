"""Tests for the robustness improvements: robust centroid (anti-masking),
flag-gated confidence, MAD degeneracy, slice-trust gating, parcel-aware
scoring, the synthetic-noise validation module, and the experiment scenarios.

Synthetic data only; no DuckDB / worldcereal / catboost. Runs in seconds.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# make experiments/ importable (sibling of src/)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from EBA_detector.anomaly_utils import (
    robust_centroid,
    compute_scores_for_slice,
    add_confidence_from_score,
    flag_anomalies,
)
from EBA_detector.robust_extensions import (
    compute_slice_trust,
    apply_trust_to_confidence,
    downgrade_flags_low_trust,
    parcel_aware_slice_scores,
    aggregate_parcel_scores,
)
from EBA_detector.validation import (
    NoiseSpec, inject_label_noise, score_embeddings_df, evaluate_detection,
)


def _unit(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


# --------------------------------------------------------------------------
# Robust centroid — masking fix
# --------------------------------------------------------------------------

class TestRobustCentroid:
    def test_trimmed_resists_masking(self):
        """With a contaminating outlier cluster, the trimmed centroid should
        stay closer to the inlier core than the plain mean, so the outliers'
        distance to the centroid is LARGER (less masked)."""
        rng = np.random.RandomState(0)
        D = 16
        core = _unit(np.tile(np.r_[1.0, np.zeros(D - 1)], (90, 1)) + 0.02 * rng.randn(90, D))
        # 10% contamination pointing in an orthogonal direction
        out = _unit(np.tile(np.r_[0.0, 1.0, np.zeros(D - 2)], (10, 1)) + 0.02 * rng.randn(10, D))
        X = np.vstack([core, out]).astype(np.float32)

        c_mean = robust_centroid(X, mode="mean")
        c_trim = robust_centroid(X, mode="trimmed", trim_frac=0.10)

        def dist_to(c):
            cn = c / (np.linalg.norm(c) + 1e-12)
            return 1.0 - (_unit(out) @ cn)

        # outliers are farther from the trimmed centroid than from the mean
        assert dist_to(c_trim).mean() > dist_to(c_mean).mean()
        # trimmed centroid is closer to the inlier core direction
        core_dir = _unit(core).mean(0); core_dir /= np.linalg.norm(core_dir)
        assert (c_trim @ core_dir) / np.linalg.norm(c_trim) > \
               (c_mean @ core_dir) / np.linalg.norm(c_mean)

    def test_median_mode_and_small_slice(self):
        rng = np.random.RandomState(1)
        X = rng.randn(3, 8).astype(np.float32)
        # n < 5 falls back to mean
        assert np.allclose(robust_centroid(X, mode="trimmed"), X.mean(0))
        med = robust_centroid(rng.randn(40, 8).astype(np.float32), mode="median")
        assert med.shape == (8,)

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            robust_centroid(np.random.randn(20, 4).astype(np.float32), mode="bogus")


# --------------------------------------------------------------------------
# Flag-gated confidence + MAD degeneracy
# --------------------------------------------------------------------------

class TestConfidenceGating:
    def test_unflagged_keep_confidence_one(self):
        df = pd.DataFrame({
            "mean_score": [0.99, 0.99, 0.2],
            "flagged": [True, False, False],
        })
        out = add_confidence_from_score(df, score_col="mean_score", flagged_col="flagged")
        # unflagged rows -> exactly 1.0 regardless of their (high) score
        assert out.loc[1, "confidence"] == pytest.approx(1.0)
        assert out.loc[2, "confidence"] == pytest.approx(1.0)
        # flagged high-score row -> penalised below 1
        assert out.loc[0, "confidence"] < 1.0

    def test_ungated_matches_legacy(self):
        df = pd.DataFrame({"mean_score": [0.99], "flagged": [False]})
        out = add_confidence_from_score(df, score_col="mean_score", flagged_col=None)
        assert out.loc[0, "confidence"] < 1.0  # legacy: penalised even if unflagged


class TestMadDegeneracy:
    def test_mad_zero_flags_nothing(self):
        # >50% identical S -> MAD == 0 -> no flags raised
        s = np.array([0.5] * 60 + list(np.linspace(0.6, 1.0, 40)), dtype=np.float32)
        df = pd.DataFrame({
            "S": s, "ewoc_code": ["c"] * 100, "h3_l3_cell": ["x"] * 100,
            "S_rank": s, "S_rank_min": s, "S_z": s,
        })
        flagged, _ = flag_anomalies(df, label_col="ewoc_code", threshold_mode="mad", mad_k=3.0)
        assert flagged["flagged"].sum() == 0


# --------------------------------------------------------------------------
# Slice trust
# --------------------------------------------------------------------------

class TestSliceTrust:
    def _two_label_context(self, separated: bool, ctx="c", n=60):
        rng = np.random.RandomState(0)
        D = 16
        if separated:
            a = _unit(np.tile(np.r_[1.0, np.zeros(D-1)], (n, 1)) + 0.05*rng.randn(n, D))
            b = _unit(np.tile(np.r_[0.0, 1.0, np.zeros(D-2)], (n, 1)) + 0.05*rng.randn(n, D))
        else:
            base = np.r_[1.0, np.zeros(D-1)]
            a = _unit(np.tile(base, (n, 1)) + 1.0*rng.randn(n, D))
            b = _unit(np.tile(base, (n, 1)) + 1.0*rng.randn(n, D))
        rows = [{"sample_id": f"a{i}", "label": "A", "ctx": ctx, "embedding": a[i]} for i in range(n)]
        rows += [{"sample_id": f"b{i}", "label": "B", "ctx": ctx, "embedding": b[i]} for i in range(n)]
        return pd.DataFrame(rows)

    def test_separated_high_entangled_low(self):
        hi = compute_slice_trust(self._two_label_context(True), label_col="label", context_cols=["ctx"])
        lo = compute_slice_trust(self._two_label_context(False), label_col="label", context_cols=["ctx"])
        assert hi["slice_trust"].iloc[0] > 0.8
        assert lo["slice_trust"].iloc[0] < 0.5

    def test_apply_trust_attenuates_lowtrust_flags(self):
        df = pd.DataFrame({
            "confidence": [0.2, 0.2],
            "slice_trust": [0.9, 0.05],
            "flagged": [True, True],
        })
        out = apply_trust_to_confidence(df, conf_col="confidence", min_trust=0.3)
        # high trust keeps the penalty; low trust pulls confidence back up
        assert out["confidence"].iloc[0] == pytest.approx(0.2, abs=1e-6)
        assert out["confidence"].iloc[1] > 0.6

    def test_downgrade_flags_low_trust(self):
        df = pd.DataFrame({
            "anomaly_flag": ["candidate", "candidate", "suspect"],
            "slice_trust": [0.9, 0.1, 0.1],
        })
        out = downgrade_flags_low_trust(df, suspect_min_trust=0.3, candidate_min_trust=0.5)
        assert out["anomaly_flag"].tolist() == ["candidate", "flagged", "flagged"]


# --------------------------------------------------------------------------
# Parcel awareness
# --------------------------------------------------------------------------

class TestParcelAware:
    def test_wrong_parcel_scores_high(self):
        rng = np.random.RandomState(0)
        D = 16
        rows = []
        # 8 good parcels near the core
        for p in range(8):
            c = _unit((np.r_[1.0, np.zeros(D-1)] + 0.05*rng.randn(D))[None])[0]
            for j in range(6):
                rows.append({"sample_id": f"g{p}_{j}", "parcel": f"g{p}",
                             "embedding": _unit((c + 0.02*rng.randn(D))[None])[0]})
        # 1 wrong parcel far away (siblings cluster together -> masks itself in plain kNN)
        cw = _unit((np.r_[0.0, 1.0, np.zeros(D-2)])[None])[0]
        for j in range(6):
            rows.append({"sample_id": f"w_{j}", "parcel": "w",
                         "embedding": _unit((cw + 0.02*rng.randn(D))[None])[0]})
        df = pd.DataFrame(rows)
        scored = parcel_aware_slice_scores(df, group_col="parcel", knn_k=5)
        agg = aggregate_parcel_scores(scored, group_col="parcel", score_col="S_parcel")
        worst = agg.sort_values("parcel_median_S_parcel").iloc[-1]
        assert worst["parcel"] == "w"  # the wrong parcel is the most anomalous


# --------------------------------------------------------------------------
# Synthetic-noise validation
# --------------------------------------------------------------------------

def _structured_df(seed=0):
    rng = np.random.RandomState(seed)
    D = 24
    cc = {c: _unit(rng.randn(1, D))[0] for c in ["maize", "wheat", "rice"]}
    rows, sid = [], 0
    for cell in ["A", "B"]:
        for cls in ["maize", "wheat", "rice"]:
            X = _unit(np.tile(cc[cls], (90, 1)) + 0.12 * rng.randn(90, D))
            for j in range(90):
                rows.append({"sample_id": f"s{sid}", "embedding": X[j].astype(np.float32),
                             "label": cls, "h3_l3_cell": cell, "parcel": f"{cell}{cls}p{j//6}"})
                sid += 1
    return pd.DataFrame(rows)


class TestValidation:
    def test_inject_label_noise_truth(self):
        df = _structured_df()
        spec = NoiseSpec(mode="within_context", rate=0.1, seed=0)
        noisy = inject_label_noise(df, spec)
        assert "noise_truth" in noisy and "label_noisy" in noisy
        # corrupted rows have a changed label; clean rows unchanged
        changed = (noisy["label"] != noisy["label_noisy"])
        assert (changed == noisy["noise_truth"]).all()
        assert 0 < noisy["noise_truth"].sum() <= len(df)

    def test_parcel_mode_corrupts_whole_group(self):
        df = _structured_df()
        spec = NoiseSpec(mode="parcel", rate=0.1, group_col="parcel", seed=1)
        noisy = inject_label_noise(df, spec)
        # within a corrupted parcel, all rows are flagged corrupted
        for pid, g in noisy.groupby("parcel"):
            assert g["noise_truth"].nunique() == 1

    def test_detector_recovers_planted_errors(self):
        df = _structured_df()
        spec = NoiseSpec(mode="within_context", rate=0.1, seed=0)
        noisy = inject_label_noise(df, spec)
        scored = score_embeddings_df(noisy, label_col="label_noisy", h3_col="h3_l3_cell")
        scored["noise_truth"] = scored["sample_id"].map(
            noisy.set_index("sample_id")["noise_truth"]).fillna(False)
        m = evaluate_detection(scored, score_col="S")
        # on well-separated synthetic data the detector should recover errors well
        assert m["auroc"] > 0.85
        assert m["average_precision"] > 0.6

    def test_evaluate_detection_degenerate(self):
        df = pd.DataFrame({"S": [0.1, 0.2, 0.3], "noise_truth": [False, False, False]})
        m = evaluate_detection(df, score_col="S")
        assert np.isnan(m["auroc"])  # no positives -> undefined


# --------------------------------------------------------------------------
# Experiment scenarios
# --------------------------------------------------------------------------

class TestScenarios:
    def _df(self, n=400):
        rng = np.random.RandomState(0)
        return pd.DataFrame({
            "label": rng.choice(["a", "b", "c"], n),
            "flag": rng.choice(["normal", "flagged", "suspect", "candidate"], n,
                               p=[.8, .1, .07, .03]),
            "conf": np.clip(rng.beta(8, 1, n), 0, 1),
            "quality": rng.choice([50, 95, 100], n),
        })

    def test_drop_and_weight(self):
        from scenarios import Scenario, apply_scenario_to_train
        df = self._df()
        s_drop = Scenario("d", drop_mode="drop_suspect")
        out = apply_scenario_to_train(df, s_drop, flag_col="flag", conf_col="conf")
        assert not out["flag"].isin(["suspect", "candidate"]).any()
        assert (out["sample_weight"] == 1.0).all()

        s_w = Scenario("w", weight_mode="conf_power", weight_power=2.0)
        outw = apply_scenario_to_train(df, s_w, flag_col="flag", conf_col="conf")
        assert len(outw) == len(df)  # weighting keeps all rows
        assert outw["sample_weight"].max() <= 1.0 and outw["sample_weight"].min() >= 0.0

    def test_build_test_views(self):
        from scenarios import build_test_views
        df = self._df()
        views = build_test_views(df, flag_col="flag", quality_col="quality", quality_threshold=0.9)
        assert set(views) == {"clean", "full", "minus_flagged"}
        assert (views["clean"]["quality"] >= 90).all()
        assert not views["minus_flagged"]["flag"].isin(["flagged", "suspect", "candidate"]).any()
        assert len(views["full"]) == len(df)
