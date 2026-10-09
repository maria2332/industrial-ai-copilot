"""Tests for feature engineering. The leakage tests are the most important ones."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from industrial_ai.ml.features import (
    FeatureConfig,
    UnitHistoryFeatures,
    build_features,
    feature_names,
)

CONFIG = FeatureConfig(sensors=("sensor_2", "sensor_3"), window=3, trend_lag=2, baseline_cycles=4)


def test_features_only_use_past_cycles(make_trajectories):
    trajectories = make_trajectories({1: 12})
    altered = trajectories.copy()
    altered.loc[altered["cycle"] > 8, "sensor_2"] += 1000.0

    original = build_features(trajectories, CONFIG)
    changed = build_features(altered, CONFIG)

    past = trajectories["cycle"] <= 8
    pd.testing.assert_frame_equal(original[past], changed[past])
    assert not original[~past].equals(changed[~past])


def test_features_of_a_unit_ignore_other_units(make_trajectories):
    trajectories = make_trajectories({1: 10, 2: 10})
    altered = trajectories.copy()
    altered.loc[altered["unit"] == 2, "sensor_2"] += 1000.0

    unit_1 = trajectories["unit"] == 1
    pd.testing.assert_frame_equal(
        build_features(trajectories, CONFIG)[unit_1], build_features(altered, CONFIG)[unit_1]
    )


def test_output_keeps_the_input_row_order(make_trajectories):
    trajectories = make_trajectories({1: 8, 2: 6})
    shuffled = trajectories.sample(frac=1.0, random_state=0)

    from_shuffled = build_features(shuffled, CONFIG)

    assert from_shuffled.index.equals(shuffled.index)
    pd.testing.assert_frame_equal(from_shuffled.sort_index(), build_features(trajectories, CONFIG))


def test_known_values_on_a_linear_ramp(make_trajectories):
    trajectories = make_trajectories({1: 30})
    trajectories["sensor_2"] = trajectories["cycle"].astype(float)
    config = FeatureConfig(sensors=("sensor_2",), window=10, trend_lag=10, baseline_cycles=20)

    features = build_features(trajectories, config).set_index(trajectories["cycle"])

    # cycle 25: mean of cycles 16..25 = 20.5; ten cycles earlier: mean of 6..15 = 10.5
    assert features.loc[25, "sensor_2_mean10"] == pytest.approx(20.5)
    assert features.loc[25, "sensor_2_trend10"] == pytest.approx(10.0)
    # baseline = mean of cycles 1..20 = 10.5
    assert features.loc[25, "sensor_2_delta_baseline"] == pytest.approx(10.0)
    # early cycles: no trend available yet, and the baseline is still being formed
    assert features.loc[5, "sensor_2_trend10"] == 0.0
    assert features.loc[1, "sensor_2_delta_baseline"] == 0.0


def test_trend_is_zero_until_both_windows_are_complete(make_trajectories):
    trajectories = make_trajectories({1: 30})
    trajectories["sensor_2"] = trajectories["cycle"].astype(float)
    config = FeatureConfig(sensors=("sensor_2",), window=10, trend_lag=10, baseline_cycles=20)

    trend = build_features(trajectories, config)["sensor_2_trend10"].set_axis(trajectories["cycle"])

    # before cycle 20 the older rolling mean covers fewer than 10 cycles: no trend yet
    assert (trend.loc[:19] == 0.0).all()
    # cycle 20: mean of 11..20 (15.5) minus mean of 1..10 (5.5)
    assert trend.loc[20] == pytest.approx(10.0)


def test_incomplete_history_is_rejected(make_trajectories):
    trajectories = make_trajectories({1: 10})

    with pytest.raises(ValueError, match="complete history"):
        build_features(trajectories[trajectories["cycle"] > 3], CONFIG)
    with pytest.raises(ValueError, match="complete history"):
        build_features(trajectories[trajectories["cycle"] != 5], CONFIG)


def test_missing_sensor_column_is_reported(make_trajectories):
    trajectories = make_trajectories({1: 5}).drop(columns="sensor_3")

    with pytest.raises(KeyError, match="sensor_3"):
        build_features(trajectories, CONFIG)


def test_feature_names_match_output_columns(make_trajectories):
    trajectories = make_trajectories({1: 6})
    with_cycle = FeatureConfig(sensors=("sensor_2",), include_cycle=True)

    assert list(build_features(trajectories, CONFIG).columns) == feature_names(CONFIG)
    assert feature_names(with_cycle)[-1] == "cycle"
    assert len(feature_names(FeatureConfig())) == 14 * 3


@pytest.mark.parametrize("field", ["window", "trend_lag", "baseline_cycles"])
def test_config_rejects_non_positive_parameters(field):
    with pytest.raises(ValueError):
        FeatureConfig(**{field: 0})


def test_transformer_works_inside_a_pipeline(make_trajectories):
    trajectories = make_trajectories({1: 15, 2: 15, 3: 15})
    target = (trajectories["cycle"] > 10).astype(int)
    pipeline = make_pipeline(
        UnitHistoryFeatures(CONFIG), StandardScaler(), LogisticRegression(max_iter=1000)
    )

    fitted = clone(pipeline).fit(trajectories, target)
    probabilities = fitted.predict_proba(trajectories)[:, 1]

    assert probabilities.shape == (len(trajectories),)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert list(fitted[0].get_feature_names_out()) == feature_names(CONFIG)
