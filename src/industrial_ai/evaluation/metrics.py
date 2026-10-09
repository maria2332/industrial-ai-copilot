"""Metrics for binary alarms: decision threshold, confusion-matrix metrics and unit-level alarms.

Generic on purpose: the same functions evaluate the failure classifier and, later, the anomaly
detector (an anomaly score with a threshold is also an alarm).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def select_threshold(y_true: ArrayLike, scores: ArrayLike, min_recall: float) -> float:
    """Return the threshold with the highest precision among those with recall >= min_recall.

    A row is predicted positive when score >= threshold. Ties keep the lowest threshold, which
    has the highest recall.
    """
    if not 0.0 < min_recall <= 1.0:
        raise ValueError("min_recall must be in (0, 1]")
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # The last precision/recall pair (recall 0) has no threshold attached.
    # The lowest threshold flags every row (recall 1), so at least one threshold is eligible.
    precision, recall = precision[:-1], recall[:-1]
    eligible = recall >= min_recall
    best = int(np.argmax(np.where(eligible, precision, -1.0)))
    return float(thresholds[best])


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def threshold_metrics(y_true: ArrayLike, scores: ArrayLike, threshold: float) -> dict[str, float]:
    """Metrics of the alarm `score >= threshold`, plus threshold-free ranking metrics."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()

    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    both_classes = len(np.unique(y_true)) == 2
    return {
        "threshold": float(threshold),
        "prevalence": float(y_true.mean()),
        "precision": precision,
        "recall": recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "false_positive_rate": _ratio(fp, fp + tn),
        "false_negative_rate": _ratio(fn, fn + tp),
        "accuracy": _ratio(tp + tn, len(y_true)),
        "pr_auc": float(average_precision_score(y_true, scores)) if both_classes else np.nan,
        "roc_auc": float(roc_auc_score(y_true, scores)) if both_classes else np.nan,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def last_cycle_mask(frame: pd.DataFrame) -> pd.Series:
    """True for the last observed cycle of each unit."""
    return frame["cycle"] == frame.groupby("unit")["cycle"].transform("max")


def first_alarms(
    frame: pd.DataFrame, scores: ArrayLike, threshold: float, consecutive: int = 1
) -> pd.DataFrame:
    """First cycle at which each unit raises an alarm, and its RUL at that moment.

    An alarm is raised once the score has been >= threshold for `consecutive` cycles in a
    row, which filters isolated spikes. `scores` are aligned with the rows of `frame`, which
    needs unit, cycle and rul. Units that never alarm get NaN.
    """
    if consecutive < 1:
        raise ValueError("consecutive must be at least 1")
    scores = np.asarray(scores, dtype=float)
    if len(scores) != len(frame):
        raise ValueError("scores must have one value per row of frame")

    data = frame[["unit", "cycle", "rul"]].assign(alarm=scores >= threshold)
    data = data.sort_values(["unit", "cycle"])
    resets = (~data["alarm"]).groupby(data["unit"]).cumsum()
    streak = data["alarm"].astype(int).groupby([data["unit"], resets]).cumsum()

    first = data[streak >= consecutive].groupby("unit")[["cycle", "rul"]].first()
    first = first.rename(columns={"cycle": "first_alarm_cycle", "rul": "rul_at_first_alarm"})
    units = pd.Index(sorted(data["unit"].unique()), name="unit")
    return first.reindex(units).astype(float)


def summarize_first_alarms(rul_at_first_alarm: pd.Series, horizon: int) -> dict[str, float]:
    """Classify each unit's first alarm by how long before failure it came.

    missed: no alarm before failure; in window: RUL <= horizon; early: up to twice the horizon
    before failure; premature: earlier than that. For a classifier a premature alarm is most
    likely false; for an anomaly detector it may also be the onset of degradation.
    """
    rul = rul_at_first_alarm.astype(float)
    return {
        "units": float(len(rul)),
        "missed (no alarm)": float(rul.isna().sum()),
        f"in window (RUL ≤ {horizon})": float((rul <= horizon).sum()),
        f"early ({horizon} < RUL ≤ {2 * horizon})": float(
            rul.between(horizon + 1, 2 * horizon).sum()
        ),
        f"premature (RUL > {2 * horizon})": float((rul > 2 * horizon).sum()),
        "median RUL at first alarm": float(rul.median()),
        "least warning (min RUL at first alarm)": float(rul.min()),
    }


def reliability_table(y_true: ArrayLike, scores: ArrayLike, n_bins: int = 10) -> pd.DataFrame:
    """Predicted probability vs observed frequency, in equal-width probability bins.

    A calibrated model has observed_rate close to mean_predicted in every bin. Empty bins are
    dropped.
    """
    y_true = np.asarray(y_true, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if len(y_true) != len(scores):
        raise ValueError("y_true and scores must have the same length")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = pd.cut(scores, edges, include_lowest=True)
    table = (
        pd.DataFrame({"bin": bins, "score": scores, "label": y_true})
        .groupby("bin", observed=True)
        .agg(
            count=("label", "size"),
            mean_predicted=("score", "mean"),
            observed_rate=("label", "mean"),
        )
    )
    return table


def expected_calibration_error(table: pd.DataFrame) -> float:
    """Average gap between predicted and observed rates, weighted by the rows in each bin."""
    weights = table["count"] / table["count"].sum()
    return float((weights * (table["mean_predicted"] - table["observed_rate"]).abs()).sum())
