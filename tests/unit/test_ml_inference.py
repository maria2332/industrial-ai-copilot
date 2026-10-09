"""Tests for inference with saved artifacts."""

import pytest

from industrial_ai.ml.anomaly import AnomalyDetector
from industrial_ai.ml.features import FeatureConfig
from industrial_ai.ml.inference import CLASSIFIER_NAME, DETECTOR_NAME, Predictor
from industrial_ai.ml.labels import LABEL_COLUMN, add_failure_label, add_rul_train
from industrial_ai.ml.models import build_pipeline, model_inputs
from industrial_ai.ml.registry import save_artifact

CONFIG = FeatureConfig(sensors=("sensor_2", "sensor_4", "sensor_7"), include_cycle=True)


@pytest.fixture
def labelled(make_trajectories):
    trajectories = add_rul_train(make_trajectories({unit: 80 for unit in range(1, 7)}))
    trajectories["sensor_4"] += 0.5 * (15 - trajectories["rul"]).clip(lower=0)
    return add_failure_label(trajectories, horizon=10)


@pytest.fixture
def saved(tmp_path, labelled):
    X, y = model_inputs(labelled), labelled[LABEL_COLUMN]
    classifier = build_pipeline("logistic_regression", feature_config=CONFIG).fit(X, y)
    detector = AnomalyDetector(method="max_zscore").fit(X)
    save_artifact(
        classifier,
        CLASSIFIER_NAME,
        "1.0.0",
        {"threshold": 0.5, "horizon_cycles": 10},
        models_dir=tmp_path,
    )
    save_artifact(
        detector, DETECTOR_NAME, "1.0.0", {"alarm_consecutive_cycles": 3}, models_dir=tmp_path
    )
    return tmp_path


def test_failure_risk_describes_the_last_cycle(saved, labelled):
    predictor = Predictor.load(models_dir=saved)
    history = labelled[labelled["unit"] == 1]

    risk = predictor.predict_failure(history, top_n=3)

    expected = predictor.classifier.predict_proba(model_inputs(history))[-1, 1]
    assert risk.cycle == 80
    assert risk.probability == pytest.approx(expected)
    assert risk.high_risk == (risk.probability >= 0.5)
    assert risk.horizon_cycles == 10
    assert len(risk.top_features) == 3
    assert risk.model_version == "1.0.0"
    assert set(risk.as_dict()) >= {"probability", "threshold", "top_features"}


def test_anomaly_status_requires_consecutive_cycles(saved, labelled):
    predictor = Predictor.load(models_dir=saved)
    history = labelled[labelled["unit"] == 2]
    margins = predictor.detector.margin(model_inputs(history))

    status = predictor.detect_anomaly(history)

    assert status.cycle == 80
    assert status.margin == pytest.approx(margins[-1])
    assert status.anomalous_now == (margins[-1] >= 0)
    assert status.sustained_alarm == bool((margins[-3:] >= 0).all())
    assert status.consecutive_cycles_required == 3
    assert len(status.top_deviations) == 3


def test_history_order_does_not_matter(saved, labelled):
    predictor = Predictor.load(models_dir=saved)
    history = labelled[labelled["unit"] == 3]

    shuffled = predictor.predict_failure(history.sample(frac=1.0, random_state=0))

    assert shuffled.probability == pytest.approx(predictor.predict_failure(history).probability)


def test_one_unit_at_a_time(saved, labelled):
    predictor = Predictor.load(models_dir=saved)

    with pytest.raises(ValueError, match="exactly one unit"):
        predictor.predict_failure(labelled[labelled["unit"].isin([1, 2])])


def test_incomplete_history_is_rejected(saved, labelled):
    predictor = Predictor.load(models_dir=saved)
    history = labelled[(labelled["unit"] == 1) & (labelled["cycle"] > 5)]

    with pytest.raises(ValueError, match="complete history"):
        predictor.predict_failure(history)
