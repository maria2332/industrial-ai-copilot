"""Tests for the candidate model pipelines."""

import numpy as np
import pytest
from sklearn.base import clone

from industrial_ai.ml.features import FeatureConfig
from industrial_ai.ml.labels import LABEL_COLUMN, add_failure_label, add_rul_train
from industrial_ai.ml.models import INPUT_COLUMNS, MODEL_NAMES, build_pipeline, model_inputs

SKLEARN_MODELS = [name for name in MODEL_NAMES if name != "xgboost"]


@pytest.fixture
def labelled(make_trajectories):
    trajectories = make_trajectories({unit: 40 for unit in range(1, 5)})
    return add_failure_label(add_rul_train(trajectories), horizon=10)


def _assert_fits_and_returns_probabilities(pipeline, labelled):
    X, y = model_inputs(labelled), labelled[LABEL_COLUMN]

    scores = clone(pipeline).fit(X, y).predict_proba(X)[:, 1]

    assert scores.shape == (len(X),)
    assert np.all((scores >= 0) & (scores <= 1))


@pytest.mark.parametrize("model_name", SKLEARN_MODELS)
@pytest.mark.parametrize("inputs", ["features", "raw"])
def test_candidates_fit_and_return_probabilities(labelled, model_name, inputs):
    _assert_fits_and_returns_probabilities(build_pipeline(model_name, inputs=inputs), labelled)


def test_xgboost_candidate_fits_and_returns_probabilities(labelled):
    pytest.importorskip("xgboost")
    _assert_fits_and_returns_probabilities(build_pipeline("xgboost"), labelled)


def test_only_logistic_regression_is_scaled():
    for name in SKLEARN_MODELS:
        has_scaler = "scale" in build_pipeline(name).named_steps
        assert has_scaler == (name == "logistic_regression"), name


def test_raw_inputs_are_the_selected_sensors_of_the_current_cycle(labelled):
    X = model_inputs(labelled)
    input_step = build_pipeline("random_forest", inputs="raw").named_steps["inputs"]

    transformed = input_step.fit_transform(X)

    np.testing.assert_array_equal(transformed, X[list(FeatureConfig().sensors)].to_numpy())


def test_hist_gradient_boosting_never_holds_out_random_rows():
    assert build_pipeline("hist_gradient_boosting").named_steps["model"].early_stopping is False


def test_model_inputs_exclude_the_labels(labelled):
    X = model_inputs(labelled)

    assert list(X.columns) == list(INPUT_COLUMNS)
    assert "rul" not in X.columns
    assert LABEL_COLUMN not in X.columns


def test_model_inputs_report_missing_columns(labelled):
    with pytest.raises(KeyError, match="sensor_4"):
        model_inputs(labelled.drop(columns="sensor_4"))


def test_unknown_model_or_inputs_are_rejected():
    with pytest.raises(ValueError, match="unknown model"):
        build_pipeline("support_vector_machine")
    with pytest.raises(ValueError, match="unknown inputs"):
        build_pipeline("random_forest", inputs="images")
