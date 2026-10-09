"""Tests for alarm metrics. The small example can be checked by hand:

scores    0.1  0.2  0.3  0.4  0.6  0.7  0.8  0.9
label      0    0    1    0    1    1    0    1

threshold 0.6 -> flags 0.6, 0.7, 0.8, 0.9: TP 3, FP 1, FN 1, TN 3 (precision 0.75, recall 0.75)
threshold 0.3 -> flags the six highest:     TP 4, FP 2, FN 0, TN 2 (precision 0.67, recall 1.00)
"""

import math

import numpy as np
import pandas as pd
import pytest

from industrial_ai.evaluation.metrics import (
    first_alarms,
    last_cycle_mask,
    select_threshold,
    summarize_first_alarms,
    threshold_metrics,
)

SCORES = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
LABELS = np.array([0, 0, 1, 0, 1, 1, 0, 1])


@pytest.mark.parametrize(("min_recall", "expected"), [(0.75, 0.6), (1.0, 0.3)])
def test_threshold_has_best_precision_among_those_reaching_the_recall(min_recall, expected):
    assert select_threshold(LABELS, SCORES, min_recall) == pytest.approx(expected)


@pytest.mark.parametrize("min_recall", [0.0, 1.5])
def test_threshold_rejects_invalid_recall_targets(min_recall):
    with pytest.raises(ValueError):
        select_threshold(LABELS, SCORES, min_recall)


def test_threshold_metrics_match_the_hand_computed_example():
    metrics = threshold_metrics(LABELS, SCORES, threshold=0.6)

    assert (metrics["tp"], metrics["fp"], metrics["tn"], metrics["fn"]) == (3, 1, 3, 1)
    for name in ("precision", "recall", "f1", "accuracy"):
        assert metrics[name] == pytest.approx(0.75), name
    assert metrics["false_positive_rate"] == pytest.approx(0.25)
    assert metrics["false_negative_rate"] == pytest.approx(0.25)
    assert metrics["prevalence"] == pytest.approx(0.5)


def test_threshold_metrics_with_a_single_class_report_nan_instead_of_failing():
    metrics = threshold_metrics([0, 0, 0], [0.1, 0.2, 0.9], threshold=0.5)

    assert metrics["fp"] == 1
    assert math.isnan(metrics["recall"])
    assert math.isnan(metrics["pr_auc"])
    assert math.isnan(metrics["roc_auc"])


def test_last_cycle_mask_marks_the_last_row_of_each_unit(make_trajectories):
    mask = last_cycle_mask(make_trajectories({1: 3, 2: 2}))

    assert mask.tolist() == [False, False, True, False, True]


@pytest.fixture
def alarm_history():
    """Unit 1 alarms at cycles 1, 3, 4 and 6; unit 2 never alarms."""
    frame = pd.DataFrame(
        {
            "unit": [1] * 6 + [2] * 3,
            "cycle": [1, 2, 3, 4, 5, 6, 1, 2, 3],
            "rul": [5, 4, 3, 2, 1, 0, 2, 1, 0],
        }
    )
    scores = np.array([0.9, 0.1, 0.8, 0.8, 0.2, 0.9, 0.1, 0.2, 0.3])
    return frame, scores


@pytest.mark.parametrize(
    ("consecutive", "cycle", "rul"), [(1, 1.0, 5.0), (2, 4.0, 2.0), (3, np.nan, np.nan)]
)
def test_first_alarm_requires_consecutive_cycles(alarm_history, consecutive, cycle, rul):
    frame, scores = alarm_history

    alarms = first_alarms(frame, scores, threshold=0.5, consecutive=consecutive)

    np.testing.assert_array_equal(alarms.loc[1].to_numpy(), [cycle, rul])
    assert alarms.loc[2].isna().all()


def test_first_alarm_does_not_depend_on_row_order(alarm_history):
    frame, scores = alarm_history
    order = np.random.default_rng(0).permutation(len(frame))

    shuffled = first_alarms(frame.iloc[order], scores[order], threshold=0.5, consecutive=2)

    pd.testing.assert_frame_equal(shuffled, first_alarms(frame, scores, 0.5, consecutive=2))


def test_first_alarm_validates_its_inputs(alarm_history):
    frame, scores = alarm_history

    with pytest.raises(ValueError, match="consecutive"):
        first_alarms(frame, scores, threshold=0.5, consecutive=0)
    with pytest.raises(ValueError, match="one value per row"):
        first_alarms(frame, scores[:-1], threshold=0.5)


def test_first_alarm_summary_classifies_each_unit():
    rul = pd.Series([np.nan, 5.0, 30.0, 31.0, 60.0, 61.0, 120.0], index=range(1, 8))

    summary = summarize_first_alarms(rul, horizon=30)

    assert summary["units"] == 7
    assert summary["missed (no alarm)"] == 1
    assert summary["in window (RUL ≤ 30)"] == 2
    assert summary["early (30 < RUL ≤ 60)"] == 2
    assert summary["premature (RUL > 60)"] == 2
    assert summary["median RUL at first alarm"] == pytest.approx(45.5)
    assert summary["least warning (min RUL at first alarm)"] == 5
