"""Unsupervised anomaly detection against a healthy reference.

The detector learns what healthy operation looks like from early-life cycles only and never
sees a failure label; labels are used only to evaluate it. Its threshold is also label-free:
the score exceeded by a fixed share of the healthy reference cycles (the false-alarm budget).

Two methods share the same features, reference window and threshold rule:
- "max_zscore": the largest absolute deviation of any feature from its healthy mean, in
  standard deviations; the logic of a classic control chart, used as the simple baseline;
- "isolation_forest": scikit-learn's IsolationForest fitted on the healthy reference.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import IsolationForest

from industrial_ai.ml.features import FeatureConfig, build_features, feature_names
from industrial_ai.ml.split import check_units_disjoint

METHODS = ("max_zscore", "isolation_forest")
DEFAULT_FEATURE_KINDS = ("delta_baseline", "trend")


def select_feature_columns(names: Sequence[str], kinds: Sequence[str]) -> list[str]:
    """Keep the per-sensor features whose kind is listed, e.g. "trend" keeps "sensor_4_trend10".

    The unit's age (`cycle`) is never selected: every late cycle would look anomalous simply
    because the healthy reference only contains young units.
    """
    patterns = [re.compile(rf"^sensor_\d+_{re.escape(kind)}\d*$") for kind in kinds]
    return [name for name in names if any(pattern.match(name) for pattern in patterns)]


class AnomalyDetector(BaseEstimator):
    """Detector fitted on the healthy reference cycles of each training unit.

    healthy_first_cycle: first reference cycle; by default the first cycle whose features are
        complete (baseline and trend windows filled).
    healthy_last_cycle: last cycle assumed healthy (see D-021).
    false_alarm_rate: share of healthy reference cycles allowed above the threshold.
    """

    def __init__(
        self,
        method: str = "isolation_forest",
        feature_kinds: Sequence[str] = DEFAULT_FEATURE_KINDS,
        healthy_first_cycle: int | None = None,
        healthy_last_cycle: int = 60,
        false_alarm_rate: float = 0.01,
        feature_config: FeatureConfig | None = None,
        random_state: int = 42,
    ) -> None:
        self.method = method
        self.feature_kinds = feature_kinds
        self.healthy_first_cycle = healthy_first_cycle
        self.healthy_last_cycle = healthy_last_cycle
        self.false_alarm_rate = false_alarm_rate
        self.feature_config = feature_config
        self.random_state = random_state

    def _config(self) -> FeatureConfig:
        return self.feature_config or FeatureConfig()

    def healthy_window(self) -> tuple[int, int]:
        """First and last cycle of the healthy reference."""
        config = self._config()
        first = self.healthy_first_cycle or max(
            config.window + config.trend_lag, config.baseline_cycles
        )
        return first, self.healthy_last_cycle

    def in_healthy_window(self, X: pd.DataFrame) -> pd.Series:
        """True for the rows inside the healthy reference window."""
        first, last = self.healthy_window()
        return X["cycle"].between(first, last)

    def _validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown method {self.method!r}; expected one of {METHODS}")
        if not 0.0 < self.false_alarm_rate < 1.0:
            raise ValueError("false_alarm_rate must be in (0, 1)")
        first, last = self.healthy_window()
        if not 1 <= first < last:
            raise ValueError("the healthy window needs 1 <= first cycle < last cycle")

    def _features(self, X: pd.DataFrame) -> pd.DataFrame:
        all_features = build_features(X, self._config())
        return all_features[self.feature_names_]

    def _raw_score(self, features: pd.DataFrame) -> np.ndarray:
        if self.method == "max_zscore":
            return self._zscores(features).abs().max(axis=1).to_numpy()
        # score_samples is higher for normal points, so it is negated: higher = more anomalous
        return -self.model_.score_samples(features)

    def _zscores(self, features: pd.DataFrame) -> pd.DataFrame:
        return (features - self.mean_) / self.std_

    def fit(self, X: pd.DataFrame, y: object = None) -> AnomalyDetector:
        """Learn the healthy reference from the cycles inside the healthy window only."""
        self._validate()
        names = select_feature_columns(feature_names(self._config()), self.feature_kinds)
        if not names:
            raise ValueError(f"no features match the kinds {tuple(self.feature_kinds)}")
        self.feature_names_ = names

        healthy = self.in_healthy_window(X).to_numpy()
        if not healthy.any():
            raise ValueError("no cycles inside the healthy window: units are too short")
        reference = self._features(X)[healthy]

        self.mean_ = reference.mean()
        std = reference.std()
        self.std_ = std.where(std > 0, 1.0)
        if self.method == "isolation_forest":
            self.model_ = IsolationForest(
                n_estimators=200, random_state=self.random_state, n_jobs=-1
            ).fit(reference)

        reference_scores = self._raw_score(reference)
        self.threshold_ = float(np.quantile(reference_scores, 1.0 - self.false_alarm_rate))
        return self

    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """Anomaly score of every row (higher = further from healthy operation)."""
        return self._raw_score(self._features(X))

    def margin(self, X: pd.DataFrame) -> np.ndarray:
        """Score minus threshold: an alarm is raised where the margin is >= 0."""
        return self.anomaly_score(X) - self.threshold_

    def is_anomalous(self, X: pd.DataFrame) -> np.ndarray:
        return self.margin(X) >= 0

    def zscores(self, X: pd.DataFrame) -> pd.DataFrame:
        """Signed deviation of each feature from its healthy mean, in standard deviations."""
        return self._zscores(self._features(X))

    def top_deviations(self, X: pd.DataFrame, n: int = 3) -> pd.DataFrame:
        """The n features that deviate most from the healthy reference, for each row.

        This describes how the current state differs from healthy operation; it does not say
        why the deviation happens.
        """
        z = self.zscores(X)
        order = np.argsort(-np.abs(z.to_numpy()), axis=1)[:, :n]
        rows = [
            (index, rank + 1, z.columns[column], float(z.iat[position, column]))
            for position, index in enumerate(z.index)
            for rank, column in enumerate(order[position])
        ]
        return pd.DataFrame(rows, columns=["row", "rank", "feature", "zscore"])


def out_of_fold_margins(
    detector: AnomalyDetector,
    frame: pd.DataFrame,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Margin of every row from a detector fitted only on the other units' healthy cycles.

    Each fold has its own threshold, so margins (score - threshold) are what is comparable
    across folds: an alarm is a margin >= 0.
    """
    margins = np.full(len(frame), np.nan)
    for train_idx, validation_idx in folds:
        check_units_disjoint(frame, train_idx, validation_idx)
        fitted = clone(detector).fit(frame.iloc[train_idx])
        margins[validation_idx] = fitted.margin(frame.iloc[validation_idx])
    if np.isnan(margins).any():
        raise ValueError("the folds must cover every row exactly once")
    return margins
