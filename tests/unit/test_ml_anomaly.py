"""Tests for the healthy-reference anomaly detector."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from industrial_ai.ml.anomaly import (
    METHODS,
    AnomalyDetector,
    out_of_fold_margins,
    select_feature_columns,
)
from industrial_ai.ml.split import unit_group_kfold

DRIFT_START = 80
# Like real degradation, several sensors drift together, some up and some down;
# sensor_4 drifts twice as fast as the others.
DRIFT_DIRECTIONS = {"sensor_2": 1, "sensor_3": 1, "sensor_4": 2, "sensor_7": -1, "sensor_12": -1}


@pytest.fixture
def drifting(make_trajectories):
    """Six units of 120 cycles whose sensors start drifting after cycle 80."""
    trajectories = make_trajectories({unit: 120 for unit in range(1, 7)})
    drift = 0.1 * (trajectories["cycle"] - DRIFT_START).clip(lower=0)
    for sensor, direction in DRIFT_DIRECTIONS.items():
        trajectories[sensor] = trajectories[sensor] + direction * drift
    return trajectories


@pytest.mark.parametrize("method", METHODS)
def test_detector_only_learns_from_the_healthy_window(drifting, method):
    altered = drifting.copy()
    altered.loc[altered["cycle"] > 60, "sensor_7"] += 100.0

    original = AnomalyDetector(method=method).fit(drifting)
    changed = AnomalyDetector(method=method).fit(altered)

    assert original.threshold_ == changed.threshold_
    pd.testing.assert_series_equal(original.mean_, changed.mean_)


@pytest.mark.parametrize("method", METHODS)
def test_threshold_matches_the_false_alarm_budget_on_the_reference(drifting, method):
    detector = AnomalyDetector(method=method, false_alarm_rate=0.05).fit(drifting)
    healthy = detector.in_healthy_window(drifting).to_numpy()

    alarm_share = detector.is_anomalous(drifting)[healthy].mean()

    assert 0.0 < alarm_share <= 0.07


@pytest.mark.parametrize("method", METHODS)
def test_drift_is_detected(drifting, method):
    detector = AnomalyDetector(method=method).fit(drifting)
    late = (drifting["cycle"] > 110).to_numpy()

    assert detector.is_anomalous(drifting)[late].mean() > 0.9


def test_default_features_are_deviations_without_age_or_levels(drifting):
    detector = AnomalyDetector().fit(drifting)

    assert len(detector.feature_names_) == 28
    assert all(name.endswith(("_delta_baseline", "_trend10")) for name in detector.feature_names_)


def test_feature_selection_by_kind():
    names = ["sensor_2_mean10", "sensor_2_trend10", "sensor_2_delta_baseline", "cycle"]

    assert select_feature_columns(names, ["mean"]) == ["sensor_2_mean10"]
    assert select_feature_columns(names, ["trend", "delta_baseline"]) == names[1:3]
    assert select_feature_columns(names, ["cycle"]) == []


def test_default_healthy_window_starts_when_features_are_complete():
    assert AnomalyDetector().healthy_window() == (20, 60)


def test_top_deviation_points_to_the_drifting_sensor(drifting):
    detector = AnomalyDetector(method="max_zscore").fit(drifting)
    last_rows = drifting.index[drifting["cycle"] == 120]

    # features need each unit's full history, so the whole frame is explained
    top = detector.top_deviations(drifting, n=3)

    assert len(top) == 3 * len(drifting)
    first_ranked = top[top["row"].isin(last_rows) & (top["rank"] == 1)]["feature"]
    assert len(first_ranked) == len(last_rows)
    assert first_ranked.str.startswith("sensor_4_").all()


def test_margin_is_score_minus_threshold(drifting):
    detector = AnomalyDetector().fit(drifting)

    np.testing.assert_allclose(
        detector.margin(drifting), detector.anomaly_score(drifting) - detector.threshold_
    )


def test_out_of_fold_margins_cover_every_row(drifting):
    folds = unit_group_kfold(drifting, n_splits=3)

    margins = out_of_fold_margins(AnomalyDetector(method="max_zscore"), drifting, folds)

    assert margins.shape == (len(drifting),)
    assert not np.isnan(margins).any()


def test_out_of_fold_margins_reject_overlapping_folds(drifting):
    leaky_folds = [(np.arange(len(drifting)), np.arange(10))]

    with pytest.raises(ValueError, match="both train and validation"):
        out_of_fold_margins(AnomalyDetector(), drifting, leaky_folds)


def test_units_shorter_than_the_healthy_window_are_rejected(make_trajectories):
    with pytest.raises(ValueError, match="healthy window"):
        AnomalyDetector().fit(make_trajectories({1: 15, 2: 15}))


@pytest.mark.parametrize(
    "params",
    [
        {"method": "one_class_svm"},
        {"false_alarm_rate": 0.0},
        {"healthy_first_cycle": 60, "healthy_last_cycle": 60},
        {"feature_kinds": ("cycle",)},
    ],
)
def test_invalid_configurations_are_rejected(drifting, params):
    with pytest.raises(ValueError):
        AnomalyDetector(**params).fit(drifting)


def test_detector_can_be_cloned_unfitted(drifting):
    detector = AnomalyDetector(method="max_zscore", false_alarm_rate=0.02).fit(drifting)

    copy = clone(detector)

    assert copy.get_params() == detector.get_params()
    assert not hasattr(copy, "threshold_")
