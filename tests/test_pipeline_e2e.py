"""End-to-end tests of ``run_pipeline`` on synthetic embeddings.

The orchestration layer used to be impossible to import without a full
worldcereal install, so nothing exercised the parts where the individually
correct helpers are wired together — which is exactly where several of the
defects lived (the discarded confidence fusion, the context key, the terminal
flag states).  ``map_classes`` is now imported lazily, so this file can drive
the real pipeline with ``embeddings_df=`` and no DuckDB, no network.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("geopandas")
pytest.importorskip("h3")

from EBA_detector.anomaly import run_pipeline  # noqa: E402

D = 24
# Real H3 level-3 cells, far enough apart not to be neighbours.
CELLS = [
    "832da1fffffffff",
    "832db0fffffffff",
    "8309a4fffffffff",
    "83754efffffffff",
    "831e30fffffffff",
    "8326b3fffffffff",
]


def make_embeddings(
    n_per_cell: int = 90,
    contamination: float = 0.0,
    seed: int = 0,
    n_bad_embeddings: int = 0,
):
    """Build an embeddings frame in the shape ``run_pipeline`` expects."""
    r = np.random.default_rng(seed)
    rows = []
    import h3 as _h3

    for ci, cell in enumerate(CELLS):
        lat, lon = _h3.cell_to_latlng(cell)
        centre = r.normal(size=D) * 0.15 + np.eye(D)[0]
        X = r.normal(size=(n_per_cell, D)) * 0.22 + centre
        n_out = int(contamination * n_per_cell)
        if n_out:
            wrong = r.normal(size=D) * 0.15 + np.eye(D)[1]
            X[:n_out] = r.normal(size=(n_out, D)) * 0.22 + wrong
        for i in range(n_per_cell):
            rows.append(
                {
                    "sample_id": f"c{ci}_s{i}",
                    "ewoc_code": 1101010000,
                    "model_hash": "m1",
                    "ref_id": f"ds{ci % 2}",
                    "h3_l3_cell": cell,
                    "lat": lat + r.normal() * 1e-4,
                    "lon": lon + r.normal() * 1e-4,
                    "truth": i < n_out,
                    **{f"embedding_{j}": float(X[i, j]) for j in range(D)},
                }
            )

    df = pd.DataFrame(rows)
    for i in range(n_bad_embeddings):
        for j in range(D):
            df.loc[i, f"embedding_{j}"] = 0.0  # zero-norm: failed inference
    embed_cols = [f"embedding_{j}" for j in range(D)]
    return df, embed_cols


MAPPING = {"LANDCOVER10": {"1101010000": "temporary_crops"}}


def run(df, embed_cols, **kw):
    params = dict(
        embeddings_db_path="unused",
        label_domain="LANDCOVER10",
        class_mappings_name="LANDCOVER10",
        mapping_file=MAPPING,
        h3_level=3,
        group_cols=[],
        min_slice_size=50,
        min_scoring_slice_size=50,
        merge_small_slice=False,
        max_full_pairwise_n=0,
        write_outputs=False,
        embeddings_df=(df, embed_cols),
        skip_classes=None,
    )
    params.update(kw)
    return run_pipeline(**params)


class TestEndToEnd:
    def test_clean_data_produces_no_candidates(self):
        df, cols = make_embeddings(contamination=0.0)
        out, _summary = run(df, cols)

        assert len(out) == len(df)
        assert set(out["anomaly_flag"].unique()) <= {
            "normal", "flagged", "suspect", "candidate", "unscored", "unscorable",
            "unmapped",
        }
        # The core promise: a clean population must not manufacture strong calls.
        assert (out["anomaly_flag"] == "candidate").sum() == 0
        assert (out["anomaly_flag"] == "suspect").sum() == 0

    def test_contaminated_data_produces_flags_with_evidence(self):
        df, cols = make_embeddings(contamination=0.15, seed=3)
        out, _summary = run(df, cols)

        flagged = out[out["anomaly_flag"].isin(["flagged", "suspect", "candidate"])]
        assert len(flagged) > 0

        truth = out["truth"].fillna(False).to_numpy(dtype=bool)
        hit = out["anomaly_flag"].isin(["flagged", "suspect", "candidate"]).to_numpy()
        precision = (hit & truth).sum() / max(hit.sum(), 1)
        assert precision > 0.7

    def test_evidence_columns_survive_to_the_output(self):
        """A reviewer opening a flagged point on a basemap must be able to see
        WHY it was flagged.  All of these used to be dropped before writing."""
        df, cols = make_embeddings(contamination=0.10, seed=5)
        out, _summary = run(df, cols)
        for col in (
            "abs_z",
            "cosine_distance",
            "knn_distance",
            "neighbourhood_offset",
            "escalation_votes",
            "scored",
            "weak_support",
        ):
            assert col in out.columns, f"{col} missing from output"

    def test_confidence_fusion_reaches_the_shipped_column(self):
        """`confidence_alt` used to be computed and then silently discarded."""
        df, cols = make_embeddings(contamination=0.15, seed=7)
        out, _summary = run(df, cols)
        assert "confidence_nonoutlier" in out.columns
        assert out["confidence_nonoutlier"].between(0.0, 1.0).all()
        # fusion actually ran and fed the output
        assert "confidence_base" in out.columns

    def test_degenerate_embeddings_become_unscorable_not_candidates(self):
        """Zero-norm vectors score cosine_distance 1.0 — the maximum — so before
        the quality gate they were guaranteed to be flagged."""
        df, cols = make_embeddings(contamination=0.0, n_bad_embeddings=4, seed=11)
        out, _summary = run(df, cols)

        bad = out[out["sample_id"].isin([f"c0_s{i}" for i in range(4)])]
        assert (bad["anomaly_flag"] == "unscorable").all()
        assert not bad["flagged"].fillna(False).any()
        assert (bad["confidence_nonoutlier"] == 1.0).all()

    def test_unmapped_codes_are_terminal_not_dropped(self):
        """Silently dropping them made every incremental update rediscover them."""
        df, cols = make_embeddings(contamination=0.0, seed=13)
        df.loc[df.index[:5], "ewoc_code"] = 9999999999  # absent from the legend
        out, _summary = run(df, cols)

        assert len(out) == len(df), "rows were dropped instead of held aside"
        assert (out["anomaly_flag"] == "unmapped").sum() == 5

    def test_small_slices_are_reported_as_unscored(self):
        df, cols = make_embeddings(n_per_cell=20, contamination=0.0, seed=17)
        out, _summary = run(df, cols, min_slice_size=10, min_scoring_slice_size=50)
        assert (out["anomaly_flag"] == "unscored").all()
        assert not out["flagged"].fillna(False).any()
        # never down-weight something we did not examine
        assert (out["confidence_nonoutlier"] == 1.0).all()

    def test_output_is_deterministic_under_row_shuffling(self):
        df, cols = make_embeddings(contamination=0.10, seed=19)
        out_a, _ = run(df, cols)
        shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        out_b, _ = run(shuffled, cols)

        a = out_a.set_index("sample_id")["anomaly_flag"].sort_index()
        b = out_b.set_index("sample_id")["anomaly_flag"].sort_index()
        pd.testing.assert_series_equal(a, b)

    def test_purity_veto_blocks_escalation(self):
        """A point whose neighbours agree with its label must not reach
        suspect/candidate however far it sits from the centroid.

        Two labels are needed for purity to mean anything: in a single-label
        context every point trivially has purity 1.0, and the veto correctly
        stands down rather than muting the whole region.
        """
        df, cols = make_embeddings(contamination=0.15, seed=23)
        # give half of each cell a second label so purity is informative
        df["ewoc_code"] = np.where(
            df.groupby("h3_l3_cell").cumcount() % 2 == 0, 1101010000, 1201010000
        )
        mapping = {
            "LANDCOVER10": {
                "1101010000": "temporary_crops",
                "1201010000": "grassland",
            }
        }
        strict, _ = run(df, cols, mapping_file=mapping, purity_veto=0.0)
        assert (strict["anomaly_flag"] == "candidate").sum() == 0
        assert (strict["anomaly_flag"] == "suspect").sum() == 0

    def test_purity_veto_stands_down_when_uninformative(self):
        """A single-label context must not have all escalation muted."""
        df, cols = make_embeddings(contamination=0.15, seed=23)
        out, _ = run(df, cols, purity_veto=0.0)
        assert not out["purity_veto"].fillna(False).any()

    def test_legacy_relative_only_mode_still_runs(self):
        """The ablation path must stay available for the paper."""
        df, cols = make_embeddings(contamination=0.05, seed=29)
        out, _ = run(df, cols, require_absolute=False, threshold_mode="mad")
        assert "anomaly_flag" in out.columns


class TestTemporalControl:
    def test_time_col_joins_the_slice_key(self):
        """Without temporal control, a minority-year sample is distant for
        phenological reasons and gets flagged as a label error."""
        df, cols = make_embeddings(contamination=0.0, n_per_cell=120, seed=31)

        # Give one sixth of each cell a different year AND a shifted embedding,
        # mimicking a different season rather than a wrong label.
        rng = np.random.default_rng(2)
        df["year"] = "2021"
        minority = df.groupby("h3_l3_cell").head(20).index
        df.loc[minority, "year"] = "2018"
        shift = rng.normal(size=D) * 0.5
        for j in range(D):
            df.loc[minority, f"embedding_{j}"] += shift[j]

        out_without, _ = run(df, cols, min_slice_size=50)
        out_with, _ = run(df, cols, min_slice_size=50, time_col="year")

        def minority_flag_rate(out):
            m = out["sample_id"].isin(df.loc[minority, "sample_id"])
            return out.loc[m, "flagged"].fillna(False).mean()

        assert minority_flag_rate(out_with) <= minority_flag_rate(out_without)
        assert "time_minority_frac" in out_with.columns


class TestSecondRoundFixes:
    def test_skip_classes_rows_get_a_terminal_state(self):
        """NaN is never terminal, so `find_unscored_samples` rediscovered these
        rows on every incremental update.  With skip_classes=["ignore"] covering
        roughly half of CROPTYPE24 that is a large recurring cost."""
        df, cols = make_embeddings(contamination=0.05, seed=3)
        df.loc[df.index[:20], "ewoc_code"] = 1201010000
        mapping = {
            "LANDCOVER10": {"1101010000": "temporary_crops", "1201010000": "ignore"}
        }
        out, _ = run(df, cols, mapping_file=mapping, skip_classes=["ignore"])

        assert (out["anomaly_flag"] == "skipped").sum() == 20
        assert not out["anomaly_flag"].isna().any()

        from EBA_detector.anomaly_utils import TERMINAL_FLAG_VALUES

        assert set(out["anomaly_flag"].unique()) <= TERMINAL_FLAG_VALUES

    def test_quality_reason_reaches_the_output(self):
        """'unscorable' without a reason cannot be triaged."""
        df, cols = make_embeddings(contamination=0.0, n_bad_embeddings=4, seed=11)
        out, _ = run(df, cols)
        bad = out[out["anomaly_flag"] == "unscorable"]
        assert len(bad) == 4
        assert set(bad["quality_reason"].dropna()) == {"zero_norm"}

    def test_context_is_geographic_not_per_dataset(self):
        """The context must not inherit group_cols.

        A slice is per-dataset so one dataset's labelling convention cannot
        contaminate another's reference cloud.  But 'what else is on the ground
        around this point?' is a geographic question: keeping ref_id in the
        context makes single-crop datasets report context_n_labels == 1, which
        removes the purity and margin votes AND trips the no-corroboration cap,
        silently making suspect/candidate unreachable.
        """
        df, cols = make_embeddings(contamination=0.05, seed=5)
        # two single-crop datasets interleaved within every cell
        alt = df.groupby("h3_l3_cell").cumcount() % 2 == 0
        df["ref_id"] = np.where(alt, "ds_maize", "ds_grass")
        df["ewoc_code"] = np.where(alt, 1101010000, 1201010000)
        mapping = {
            "LANDCOVER10": {
                "1101010000": "temporary_crops",
                "1201010000": "grassland",
            }
        }
        kw = dict(
            mapping_file=mapping, group_cols=["ref_id"],
            min_slice_size=40, min_scoring_slice_size=40,
        )
        geographic, _ = run(df, cols, **kw)
        per_dataset, _ = run(df, cols, context_group_cols=["ref_id"], **kw)

        assert (geographic["context_n_labels"] >= 2).all()
        assert geographic["corroborated"].fillna(False).all()
        # the rejected alternative loses the corroborating evidence entirely
        assert not per_dataset["corroborated"].fillna(False).any()

    def test_weak_support_demotes_exactly_one_level(self):
        """Chained .loc assignments dropped candidate->suspect->flagged."""
        from EBA_detector.anomaly import _assign_anomaly_categories

        frame = pd.DataFrame(
            {
                "flagged": [True, True],
                "abs_z": [10.0, 10.0],
                "knn_same_label_frac_ctx": [0.1, 0.1],
                "context_n_labels": [3, 3],
                "alt_margin_ctx": [-1.0, -1.0],
                "S_rank": [0.999, 0.999],
                "S": [0.99, 0.99],
                "rank_percentile": [0.999, 0.999],
                "undersized_slice": [True, False],
                "scored": [True, True],
            }
        )
        out = _assign_anomaly_categories(frame)
        assert out["combined_anomaly"].tolist() == ["suspect", "candidate"]
