"""Explanations of the failure-risk model.

- Local and global, for the selected linear model: exact per-feature contributions to the
  log-odds. For a logistic regression on standardised features, the contribution of feature j
  is coef_j * z_j, where z_j is the feature in standard deviations from its training mean; the
  contributions plus the intercept add up exactly to the model's log-odds. With independent
  features this is what SHAP (interventional) computes for a linear model, so no extra
  dependency is needed.
- Global, for any pipeline: permutation importance grouped by sensor and measured on held-out
  units, so that correlated features of the same sensor are permuted together.

These are attributions of the model's output, not causes of the failure.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline

from industrial_ai.ml.split import check_units_disjoint


def _linear_parts(pipeline: Pipeline) -> tuple[object, object, LogisticRegression]:
    steps = pipeline.named_steps
    if not {"inputs", "scale", "model"} <= set(steps) or not isinstance(
        steps["model"], LogisticRegression
    ):
        raise ValueError("exact contributions need the scaled logistic regression pipeline")
    return steps["inputs"], steps["scale"], steps["model"]


def linear_contributions(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """Contribution of every feature to the log-odds of every row (fitted pipeline).

    Row sums plus the model's intercept equal its decision_function exactly. Positive values
    push towards failure within the horizon, negative values away from it.
    """
    inputs, scale, model = _linear_parts(pipeline)
    features = inputs.transform(X)
    names = list(inputs.get_feature_names_out())
    standardised = scale.transform(features)
    return pd.DataFrame(standardised * model.coef_[0], index=X.index, columns=names)


def top_contributions(pipeline: Pipeline, X: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """The n features with the largest absolute contribution, for each row."""
    contributions = linear_contributions(pipeline, X)
    values = contributions.to_numpy()
    order = np.argsort(-np.abs(values), axis=1)[:, :n]
    rows = [
        (index, rank + 1, contributions.columns[column], float(values[position, column]))
        for position, index in enumerate(contributions.index)
        for rank, column in enumerate(order[position])
    ]
    return pd.DataFrame(rows, columns=["row", "rank", "feature", "contribution"])


def feature_groups(names: Sequence[str]) -> dict[str, list[str]]:
    """Group feature names by sensor ("sensor_11": its three features); other names alone."""
    groups: dict[str, list[str]] = {}
    for name in names:
        parts = name.split("_")
        key = "_".join(parts[:2]) if name.startswith("sensor_") else name
        groups.setdefault(key, []).append(name)
    return groups


def grouped_permutation_importance(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    target: pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Drop in PR-AUC on held-out units when all features of one group are shuffled together.

    For each fold, a fresh copy of the pipeline is fitted on the training units; the features
    of the validation units are computed once, and each group's columns are permuted jointly
    across rows (breaking their link with the target but keeping their mutual correlation).
    Returns the mean and standard deviation of the drop over folds and repeats, largest first.
    """
    rng = np.random.default_rng(random_state)
    drops: dict[str, list[float]] = {}
    for train_idx, validation_idx in folds:
        check_units_disjoint(frame, train_idx, validation_idx)
        fitted = clone(pipeline).fit(frame.iloc[train_idx], target.iloc[train_idx])
        input_step, rest = fitted[:1], fitted[1:]
        features = input_step.transform(frame.iloc[validation_idx])
        if not isinstance(features, pd.DataFrame):
            features = pd.DataFrame(features, columns=input_step.get_feature_names_out())
        y_validation = target.iloc[validation_idx]
        baseline = average_precision_score(y_validation, rest.predict_proba(features)[:, 1])

        for group, columns in feature_groups(list(features.columns)).items():
            for _ in range(n_repeats):
                permuted = features.copy()
                order = rng.permutation(len(features))
                permuted[columns] = features[columns].to_numpy()[order]
                score = average_precision_score(y_validation, rest.predict_proba(permuted)[:, 1])
                drops.setdefault(group, []).append(baseline - score)

    summary = pd.DataFrame(
        {
            "pr_auc_drop_mean": {group: np.mean(values) for group, values in drops.items()},
            "pr_auc_drop_std": {group: np.std(values) for group, values in drops.items()},
        }
    )
    return summary.sort_values("pr_auc_drop_mean", ascending=False)
