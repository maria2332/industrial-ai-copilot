"""Tests for grouped cross-validation and the model-selection rule."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline

from industrial_ai.ml.labels import LABEL_COLUMN, add_failure_label, add_rul_train
from industrial_ai.ml.models import build_pipeline, model_inputs
from industrial_ai.ml.selection import cross_validate_by_unit, simplest_within_one_std
from industrial_ai.ml.split import unit_group_kfold


class UnitMemorizer(ClassifierMixin, BaseEstimator):
    """Spy model: scores 1 for rows of units seen in training and 0 otherwise."""

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        self.seen_units_ = set(X["unit"])
        return self

    def predict_proba(self, X):
        seen = X["unit"].isin(self.seen_units_).to_numpy(dtype=float)
        return np.column_stack([1 - seen, seen])


@pytest.fixture
def labelled(make_trajectories):
    trajectories = make_trajectories({unit: 30 for unit in range(1, 9)})
    return add_failure_label(add_rul_train(trajectories), horizon=10)


def test_out_of_fold_scores_cover_every_row(labelled):
    X, y = model_inputs(labelled), labelled[LABEL_COLUMN]

    result = cross_validate_by_unit(
        build_pipeline("logistic_regression"), X, y, unit_group_kfold(labelled, n_splits=4)
    )

    assert result.oof_scores.shape == (len(X),)
    assert not np.isnan(result.oof_scores).any()
    assert len(result.fold_pr_auc) == len(result.fold_roc_auc) == 4
    assert set(result.summary()) == {
        "pr_auc_mean",
        "pr_auc_std",
        "roc_auc_mean",
        "roc_auc_std",
        "fit_seconds",
    }


def test_each_row_is_scored_by_a_model_that_never_saw_its_unit(labelled):
    X, y = model_inputs(labelled), labelled[LABEL_COLUMN]

    result = cross_validate_by_unit(
        Pipeline([("model", UnitMemorizer())]), X, y, unit_group_kfold(labelled, n_splits=4)
    )

    assert (result.oof_scores == 0).all()


def test_overlapping_folds_are_rejected(labelled):
    X, y = model_inputs(labelled), labelled[LABEL_COLUMN]
    leaky_folds = [(np.arange(len(X)), np.arange(10))]

    with pytest.raises(ValueError, match="both train and validation"):
        cross_validate_by_unit(Pipeline([("model", UnitMemorizer())]), X, y, leaky_folds)


def test_folds_must_cover_every_row(labelled):
    X, y = model_inputs(labelled), labelled[LABEL_COLUMN]
    incomplete_folds = unit_group_kfold(labelled, n_splits=4)[:3]

    with pytest.raises(ValueError, match="cover every row"):
        cross_validate_by_unit(Pipeline([("model", UnitMemorizer())]), X, y, incomplete_folds)


def test_frame_and_target_must_have_the_same_length(labelled):
    X, y = model_inputs(labelled), labelled[LABEL_COLUMN]

    with pytest.raises(ValueError, match="same length"):
        cross_validate_by_unit(build_pipeline("logistic_regression"), X, y.iloc[:-1], [])


def _summary(means: list[float], std: float = 0.01) -> pd.DataFrame:
    return pd.DataFrame(
        {"pr_auc_mean": means, "pr_auc_std": [std] * len(means)},
        index=["simple", "medium", "complex"],
    )


def test_simpler_model_wins_when_inside_the_band():
    # band: 0.975 - 0.01 = 0.965 -> "simple" (0.95) is outside, "medium" (0.97) inside
    chosen = simplest_within_one_std(_summary([0.95, 0.97, 0.975]), ["simple", "medium", "complex"])

    assert chosen == "medium"


def test_best_model_wins_when_it_is_clearly_better():
    chosen = simplest_within_one_std(_summary([0.90, 0.92, 0.98]), ["simple", "medium", "complex"])

    assert chosen == "complex"


def test_candidates_outside_the_order_are_ignored():
    chosen = simplest_within_one_std(_summary([0.99, 0.90, 0.91]), ["medium", "complex"])

    assert chosen == "medium"


def test_selection_needs_at_least_one_ordered_candidate():
    with pytest.raises(ValueError):
        simplest_within_one_std(_summary([0.9, 0.9, 0.9]), ["other"])
