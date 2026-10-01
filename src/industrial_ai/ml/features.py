"""Feature engineering for C-MAPSS trajectories.

Every feature is computed per unit and only from the current and past cycles, so it is
computed at prediction time exactly as during training. No feature holds state learned from
the training set; stateful steps such as scaling live in the model Pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# Selected in the EDA (notebook 01): sensors 1, 5, 10, 16, 18 and 19 are constant in FD001
# and sensor 6 only takes two values, so none of them carries degradation signal.
SELECTED_SENSORS: tuple[str, ...] = tuple(
    f"sensor_{i}" for i in (2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21)
)


@dataclass(frozen=True)
class FeatureConfig:
    """Parameters of the feature set.

    window: cycles averaged by the rolling mean that smooths sensor noise.
    trend_lag: cycles between the two rolling means compared by the trend feature.
    baseline_cycles: first cycles of each unit that define its own healthy reference level.
    include_cycle: whether the age of the unit is used as a feature.
    """

    sensors: tuple[str, ...] = SELECTED_SENSORS
    window: int = 10
    trend_lag: int = 10
    baseline_cycles: int = 20
    include_cycle: bool = False

    def __post_init__(self) -> None:
        if min(self.window, self.trend_lag, self.baseline_cycles) < 1:
            raise ValueError("window, trend_lag and baseline_cycles must be positive")
        if not self.sensors:
            raise ValueError("at least one sensor is required")


def feature_names(config: FeatureConfig) -> list[str]:
    """Names of the feature columns, in the order produced by build_features."""
    kinds = (f"mean{config.window}", f"trend{config.trend_lag}", "delta_baseline")
    names = [f"{sensor}_{kind}" for sensor in config.sensors for kind in kinds]
    return [*names, "cycle"] if config.include_cycle else names


def _check_complete_history(frame: pd.DataFrame) -> None:
    if not frame.index.is_unique:
        raise ValueError("the input index must be unique")
    ordered = frame.sort_values(["unit", "cycle"])
    starts = ordered.groupby("unit")["cycle"].min()
    steps = ordered.groupby("unit")["cycle"].diff().dropna()
    if (starts != 1).any() or (steps != 1).any():
        raise ValueError("each unit needs its complete history: contiguous cycles from 1")


def build_features(frame: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """Compute the feature matrix: one row per input row, in the same order as the input.

    For each selected sensor:
    - mean{window}: rolling mean over the last `window` cycles (noise reduction);
    - trend{lag}: rolling mean now minus rolling mean `lag` cycles ago (0 until available);
    - delta_baseline: rolling mean minus the unit's mean over its first `baseline_cycles`
      cycles (expanding mean while those cycles are still being observed).
    """
    config = config or FeatureConfig()
    missing = [column for column in ("unit", "cycle", *config.sensors) if column not in frame]
    if missing:
        raise KeyError(f"missing columns: {missing}")
    _check_complete_history(frame)

    ordered = frame.sort_values(["unit", "cycle"])
    units = ordered["unit"]
    sensors = ordered[list(config.sensors)].astype(float)

    rolling_mean = sensors.groupby(units).transform(
        lambda values: values.rolling(config.window, min_periods=1).mean()
    )
    trend = (rolling_mean - rolling_mean.groupby(units).shift(config.trend_lag)).fillna(0.0)

    early = sensors.copy()
    early.loc[ordered["cycle"] > config.baseline_cycles, :] = np.nan
    baseline = early.groupby(units).transform(lambda values: values.expanding().mean())
    delta_baseline = rolling_mean - baseline

    parts = [
        rolling_mean.add_suffix(f"_mean{config.window}"),
        trend.add_suffix(f"_trend{config.trend_lag}"),
        delta_baseline.add_suffix("_delta_baseline"),
    ]
    if config.include_cycle:
        parts.append(ordered[["cycle"]])
    features = pd.concat(parts, axis=1)[feature_names(config)].astype(float)
    return features.loc[frame.index]


class UnitHistoryFeatures(TransformerMixin, BaseEstimator):
    """Scikit-learn wrapper of build_features, so the saved Pipeline goes from raw history
    to prediction and training and inference cannot compute features differently."""

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config

    def _resolved_config(self) -> FeatureConfig:
        return self.config or FeatureConfig()

    def fit(self, X: pd.DataFrame, y: object = None) -> UnitHistoryFeatures:
        """Nothing is learned from the data: the features are stateless."""
        self.feature_names_out_ = np.array(feature_names(self._resolved_config()), dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return build_features(X, self._resolved_config())

    def get_feature_names_out(self, input_features: object = None) -> np.ndarray:
        return np.array(feature_names(self._resolved_config()), dtype=object)
