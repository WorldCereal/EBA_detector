"""Outlier-treatment scenarios shared by both experiment tiers.

A *scenario* describes how the outlier-detection outputs are applied to the
**training** data of a downstream crop-type model.  The same scenario objects
drive both the fast CatBoost proxy (``run_catboost_proxy.py``) and the real
WorldCereal Presto fine-tuning harness (``run_worldcereal_finetune.py``), so
the two tiers stay comparable.

Design rules that keep the experiments honest
---------------------------------------------
1. **Scenarios touch the TRAIN split only.**  The evaluation/test split is held
   fixed across all scenarios so differences are attributable to the training
   treatment, not to an easier test set.
2. **The primary test set is detector-independent.**  It is defined by
   high annotation quality (``quality_score``) or a curated gold set — never by
   the outlier scores being evaluated.  Filtering the test set with the same
   detector would be circular and inflate the apparent benefit.
3. **Three test *views* are reported** (see ``TEST_VIEWS``):
   ``clean`` (primary), ``full`` (all eval points, incl. noisy), and
   ``minus_flagged`` (eval points the detector flags removed).  Comparing the
   three separates "the model got better" from "the test set got easier".

The anomaly columns produced by the pipeline (see
``EBA_detector.anomaly_utils.ANOMALY_COLUMNS`` and
``worldcereal.train.OUTLIER_COLUMNS``) are:

* ``*_anomaly_flag``           : ``normal|flagged|suspect|candidate`` plus the
  non-judged terminal states ``unscored|unscorable|unmapped|skipped`` (rows the
  detector could not form an opinion about).  Those are deliberately absent
  from every ``DROP_SETS`` entry: "we did not look" must never cause a training
  sample to be removed.
* ``*_confidence_nonoutlier``  : continuous P(not outlier) in [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# Severity ordering of the categorical flag (low → high).
FLAG_SEVERITY: Dict[str, int] = {
    "normal": 0,
    "flagged": 1,
    "suspect": 2,
    "candidate": 3,
}

# Terminal states that are NOT positions on the severity ladder: the detector
# could not judge the sample (slice too small, embedding rejected, code absent
# from the legend) or was told to skip it.  Never map these to 0 — treating
# "unscored" as "normal" is precisely the conflation these states exist to
# remove.  Keep them, and report them separately.
NON_JUDGED_FLAGS: set = {"unscored", "unscorable", "unmapped", "skipped"}

# Which categories are removed for each hard-drop level (matches
# worldcereal.train.finetuning_utils.identify_true_outliers).
DROP_SETS: Dict[str, set] = {
    "keep": set(),
    "drop_candidate": {"candidate"},
    "drop_suspect": {"candidate", "suspect"},
    "drop_flagged": {"candidate", "suspect", "flagged"},
}

# The three test "views" used for evaluation (see module docstring).
TEST_VIEWS = ("clean", "full", "minus_flagged")


@dataclass(frozen=True)
class Scenario:
    """One training-data treatment.

    Attributes
    ----------
    name
        Unique identifier (used in result tables / filenames).
    drop_mode
        Hard removal level, one of :data:`DROP_SETS` keys.  Removes train rows
        whose ``flag_col`` is in the corresponding severity set.
    weight_mode
        Continuous down-weighting applied to *surviving* train rows:
        ``"none"``, ``"conf_linear"`` (w = confidence), ``"conf_power"``
        (w = confidence**power), or ``"conf_filter"`` (hard threshold on
        confidence, w in {0,1}).
    weight_power
        Exponent for ``conf_power``.
    conf_threshold
        Threshold for ``conf_filter`` (rows below it are dropped from train).
    description
        Human-readable note for the paper.
    """

    name: str
    drop_mode: str = "keep"
    weight_mode: str = "none"
    weight_power: float = 1.0
    conf_threshold: Optional[float] = None
    description: str = ""


# ---------------------------------------------------------------------------
# Default scenario suite for the paper
# ---------------------------------------------------------------------------

DEFAULT_SCENARIOS: List[Scenario] = [
    Scenario("baseline", "keep", "none",
             description="No outlier treatment — reference model."),
    # Hard removal at increasing severity
    Scenario("drop_candidate", "drop_candidate", "none",
             description="Remove only the most extreme (candidate) outliers."),
    Scenario("drop_suspect", "drop_suspect", "none",
             description="Remove candidate+suspect outliers."),
    Scenario("drop_flagged", "drop_flagged", "none",
             description="Remove all flagged outliers (most aggressive)."),
    # Continuous down-weighting (keep all rows, weight by confidence)
    Scenario("downweight_linear", "keep", "conf_linear",
             description="Weight each train sample by confidence_nonoutlier."),
    Scenario("downweight_power2", "keep", "conf_power", weight_power=2.0,
             description="Weight by confidence^2 (sharper penalty)."),
    Scenario("downweight_power4", "keep", "conf_power", weight_power=4.0,
             description="Weight by confidence^4 (aggressive penalty)."),
    # Confidence threshold filter
    Scenario("filter_conf_0.90", "keep", "conf_filter", conf_threshold=0.90,
             description="Drop train rows with confidence_nonoutlier < 0.90."),
    Scenario("filter_conf_0.95", "keep", "conf_filter", conf_threshold=0.95,
             description="Drop train rows with confidence_nonoutlier < 0.95."),
    # Hybrid: remove the worst, softly down-weight the rest
    Scenario("drop_suspect+downweight", "drop_suspect", "conf_linear",
             description="Remove candidate+suspect, down-weight remaining by confidence."),
]


# ---------------------------------------------------------------------------
# Pure functions (no CatBoost / torch) — unit-testable
# ---------------------------------------------------------------------------


def apply_scenario_to_train(
    train_df: pd.DataFrame,
    scenario: Scenario,
    *,
    flag_col: str,
    conf_col: str,
    weight_floor: float = 0.0,
) -> pd.DataFrame:
    """Apply *scenario* to the TRAIN split.

    Returns a copy of *train_df* restricted to surviving rows, with a
    ``sample_weight`` column added (all 1.0 unless a weighting mode applies).
    Never mutates the input.
    """
    df = train_df.copy()

    # 1. hard removal by categorical flag
    drop_set = DROP_SETS.get(scenario.drop_mode)
    if drop_set is None:
        raise ValueError(f"Unknown drop_mode: {scenario.drop_mode}")
    if drop_set and flag_col in df.columns:
        keep_mask = ~df[flag_col].astype(str).isin(drop_set)
        df = df[keep_mask].copy()

    # 2. confidence-based weighting / filtering on survivors
    conf = (
        pd.to_numeric(df[conf_col], errors="coerce").fillna(1.0).clip(0.0, 1.0)
        if conf_col in df.columns
        else pd.Series(1.0, index=df.index)
    )

    if scenario.weight_mode == "none":
        w = pd.Series(1.0, index=df.index)
    elif scenario.weight_mode == "conf_linear":
        w = conf
    elif scenario.weight_mode == "conf_power":
        if scenario.weight_power <= 0:
            raise ValueError("weight_power must be > 0")
        w = conf ** float(scenario.weight_power)
    elif scenario.weight_mode == "conf_filter":
        if scenario.conf_threshold is None:
            raise ValueError("conf_filter requires conf_threshold")
        df = df[conf >= float(scenario.conf_threshold)].copy()
        conf = conf.loc[df.index]
        w = pd.Series(1.0, index=df.index)
    else:
        raise ValueError(f"Unknown weight_mode: {scenario.weight_mode}")

    w = w.clip(lower=weight_floor, upper=1.0).astype(float)
    df["sample_weight"] = w.to_numpy()
    return df


def build_test_views(
    eval_df: pd.DataFrame,
    *,
    flag_col: str,
    quality_col: Optional[str] = None,
    quality_threshold: float = 0.9,
    gold_mask_col: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Construct the three evaluation views from a single eval split.

    Parameters
    ----------
    eval_df
        The fixed held-out split (test).  Must NOT have been filtered by the
        detector before this call.
    flag_col
        Categorical anomaly flag column (used for the ``minus_flagged`` view).
    quality_col
        Optional annotation-quality column.  If provided, the ``clean`` view is
        the subset with ``quality >= quality_threshold`` — a detector-
        independent gold set.
    gold_mask_col
        Optional boolean column explicitly marking curated gold-standard test
        samples; takes precedence over *quality_col* for the ``clean`` view.

    Returns
    -------
    dict
        ``{"clean": df, "full": df, "minus_flagged": df}``.
    """
    views: Dict[str, pd.DataFrame] = {}
    views["full"] = eval_df.copy()

    if gold_mask_col and gold_mask_col in eval_df.columns:
        clean = eval_df[eval_df[gold_mask_col].astype(bool)].copy()
    elif quality_col and quality_col in eval_df.columns:
        q = pd.to_numeric(eval_df[quality_col], errors="coerce").fillna(0.0)
        # quality scores may be on a 0-100 scale; normalise heuristically
        if q.max() > 1.5:
            q = q / 100.0
        clean = eval_df[q >= quality_threshold].copy()
    else:
        # no quality info: fall back to "full" but warn via empty-safe copy
        clean = eval_df.copy()
    views["clean"] = clean

    if flag_col in eval_df.columns:
        keep = ~eval_df[flag_col].astype(str).isin(DROP_SETS["drop_flagged"])
        views["minus_flagged"] = eval_df[keep].copy()
    else:
        views["minus_flagged"] = eval_df.copy()

    return views


def scenario_grid_summary(scenarios: Sequence[Scenario] = DEFAULT_SCENARIOS) -> pd.DataFrame:
    """Return a tidy table describing the scenario suite (for the paper appendix)."""
    return pd.DataFrame(
        [
            {
                "name": s.name,
                "drop_mode": s.drop_mode,
                "weight_mode": s.weight_mode,
                "weight_power": s.weight_power,
                "conf_threshold": s.conf_threshold,
                "description": s.description,
            }
            for s in scenarios
        ]
    )
