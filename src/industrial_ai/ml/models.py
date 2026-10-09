"""Candidate failure-risk models.

Each candidate is a complete scikit-learn Pipeline that goes from the raw sensor history of
one or more units to a probability of failure within the horizon. Hyperparameters are fixed,
reasonable defaults and are not tuned (see D-020 in docs/decisions.md).
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from industrial_ai.ml.data import ID_COLUMNS, SENSOR_COLUMNS
from industrial_ai.ml.features import FeatureConfig, UnitHistoryFeatures

RANDOM_STATE = 42

# Columns a model receives. Labels (rul, failure_within_horizon) are never part of the input.
INPUT_COLUMNS: tuple[str, ...] = (*ID_COLUMNS, *SENSOR_COLUMNS)

INPUT_KINDS = ("features", "raw")


def _logistic_regression() -> ClassifierMixin:
    return LogisticRegression(max_iter=2000)


def _random_forest() -> ClassifierMixin:
    return RandomForestClassifier(
        n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
    )


def _hist_gradient_boosting() -> ClassifierMixin:
    # Early stopping is disabled on purpose: scikit-learn would hold out a random 10% of the
    # training rows, and random rows are exactly the leaky split we avoid (see D-003).
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )


def _xgboost() -> ClassifierMixin:
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise ImportError('xgboost is not installed: run pip install -e ".[ml]"') from error
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


# Ordered from simplest to most complex; model selection prefers the earliest candidate
# that performs as well as the best one (see simplest_within_one_std).
_ESTIMATORS: dict[str, Callable[[], ClassifierMixin]] = {
    "logistic_regression": _logistic_regression,
    "random_forest": _random_forest,
    "hist_gradient_boosting": _hist_gradient_boosting,
    "xgboost": _xgboost,
}
MODEL_NAMES: tuple[str, ...] = tuple(_ESTIMATORS)


def model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the columns a model may see, so labels can never leak into the input."""
    missing = [column for column in INPUT_COLUMNS if column not in frame.columns]
    if missing:
        raise KeyError(f"missing input columns: {missing}")
    return frame[list(INPUT_COLUMNS)]


def build_pipeline(
    model_name: str, inputs: str = "features", feature_config: FeatureConfig | None = None
) -> Pipeline:
    """Build an unfitted Pipeline: input step, optional scaling and the classifier.

    inputs="features": the engineered features of each unit's history (UnitHistoryFeatures).
    inputs="raw": the selected sensors of the current cycle only, as a reference baseline.
    Only logistic regression is scaled: tree models are insensitive to feature scale, while a
    regularised linear model is not.
    """
    if model_name not in _ESTIMATORS:
        raise ValueError(f"unknown model {model_name!r}; expected one of {MODEL_NAMES}")
    if inputs not in INPUT_KINDS:
        raise ValueError(f"unknown inputs {inputs!r}; expected one of {INPUT_KINDS}")

    config = feature_config or FeatureConfig()
    if inputs == "features":
        input_step = UnitHistoryFeatures(config)
    else:
        input_step = ColumnTransformer(
            [("sensors", "passthrough", list(config.sensors))],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    steps: list[tuple[str, object]] = [("inputs", input_step)]
    if model_name == "logistic_regression":
        steps.append(("scale", StandardScaler()))
    steps.append(("model", _ESTIMATORS[model_name]()))
    return Pipeline(steps)
