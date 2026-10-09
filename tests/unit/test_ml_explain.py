"""Tests for model explanations."""

import numpy as np
import pandas as pd
import pytest

from industrial_ai.ml.explain import (
    feature_groups,
    grouped_permutation_importance,
    linear_contributions,
    top_contributions,
)
from industrial_ai.ml.features import FeatureConfig
from industrial_ai.ml.labels import LABEL_COLUMN, add_failure_label, add_rul_train
from industrial_ai.ml.models import build_pipeline, model_inputs
from industrial_ai.ml.split import unit_group_kfold

CONFIG = FeatureConfig(sensors=("sensor_2", "sensor_4", "sensor_7"), include_cycle=True)


@pytest.fixture
def labelled(make_trajectories):
    """Eight units of 60 cycles; only sensor_4 rises when failure approaches."""
    trajectories = add_rul_train(make_trajectories({unit: 60 for unit in range(1, 9)}))
    trajectories["sensor_4"] += 0.5 * (15 - trajectories["rul"]).clip(lower=0)
    return add_failure_label(trajectories, horizon=10)


@pytest.fixture
def fitted(labelled):
    pipeline = build_pipeline("logistic_regression", feature_config=CONFIG)
    return pipeline.fit(model_inputs(labelled), labelled[LABEL_COLUMN])


def test_contributions_add_up_to_the_log_odds(labelled, fitted):
    X = model_inputs(labelled)

    contributions = linear_contributions(fitted, X)
    intercept = fitted.named_steps["model"].intercept_[0]

    np.testing.assert_allclose(contributions.sum(axis=1) + intercept, fitted.decision_function(X))
    assert list(contributions.columns) == list(fitted.named_steps["inputs"].get_feature_names_out())


def test_top_contributions_are_sorted_by_absolute_value(labelled, fitted):
    top = top_contributions(fitted, model_inputs(labelled), n=3)

    assert len(top) == 3 * len(labelled)
    for _, row_ranks in top.groupby("row"):
        magnitudes = row_ranks.sort_values("rank")["contribution"].abs().to_numpy()
        assert np.all(np.diff(magnitudes) <= 0)


def test_contributions_need_the_scaled_logistic_regression(labelled):
    forest = build_pipeline("random_forest", feature_config=CONFIG)
    forest.fit(model_inputs(labelled), labelled[LABEL_COLUMN])

    with pytest.raises(ValueError, match="logistic regression"):
        linear_contributions(forest, model_inputs(labelled))


def test_feature_groups_put_each_sensor_together():
    names = ["sensor_2_mean10", "sensor_2_trend10", "sensor_11_delta_baseline", "cycle"]

    assert feature_groups(names) == {
        "sensor_2": ["sensor_2_mean10", "sensor_2_trend10"],
        "sensor_11": ["sensor_11_delta_baseline"],
        "cycle": ["cycle"],
    }


def test_permutation_importance_finds_the_informative_sensor(labelled):
    importance = grouped_permutation_importance(
        build_pipeline("logistic_regression", feature_config=CONFIG),
        model_inputs(labelled),
        labelled[LABEL_COLUMN],
        unit_group_kfold(labelled, n_splits=4),
        n_repeats=3,
    )

    assert set(importance.index) == {"sensor_2", "sensor_4", "sensor_7", "cycle"}
    assert importance.index[0] == "sensor_4"
    assert list(importance.columns) == ["pr_auc_drop_mean", "pr_auc_drop_std"]


def test_permutation_importance_rejects_overlapping_folds(labelled):
    leaky_folds = [(np.arange(len(labelled)), np.arange(10))]

    with pytest.raises(ValueError, match="both train and validation"):
        grouped_permutation_importance(
            build_pipeline("logistic_regression", feature_config=CONFIG),
            model_inputs(labelled),
            labelled[LABEL_COLUMN],
            leaky_folds,
        )


def test_contributions_keep_the_input_index(labelled, fitted):
    X = model_inputs(labelled).set_index(pd.Index(range(1000, 1000 + len(labelled))))

    assert linear_contributions(fitted, X).index.equals(X.index)
