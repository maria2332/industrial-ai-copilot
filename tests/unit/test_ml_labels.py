"""Tests for RUL and failure-horizon labels."""

import pandas as pd
import pytest

from industrial_ai.ml.labels import (
    LABEL_COLUMN,
    RUL_COLUMN,
    add_failure_label,
    add_rul_test,
    add_rul_train,
)


def test_train_rul_counts_down_to_zero_at_failure(make_trajectories):
    labelled = add_rul_train(make_trajectories({1: 5, 2: 3}))

    assert labelled.loc[labelled["unit"] == 1, RUL_COLUMN].tolist() == [4, 3, 2, 1, 0]
    assert labelled.groupby("unit")[RUL_COLUMN].min().eq(0).all()


def test_test_rul_is_reconstructed_from_rul_at_last_cycle(make_trajectories):
    trajectories = make_trajectories({1: 3, 2: 2})
    rul_at_last_cycle = pd.Series([10, 0], index=pd.Index([1, 2], name="unit"))

    labelled = add_rul_test(trajectories, rul_at_last_cycle)

    assert labelled.loc[labelled["unit"] == 1, RUL_COLUMN].tolist() == [12, 11, 10]
    assert labelled.loc[labelled["unit"] == 2, RUL_COLUMN].tolist() == [1, 0]


def test_test_rul_requires_every_unit(make_trajectories):
    with pytest.raises(ValueError, match="missing"):
        add_rul_test(make_trajectories({1: 3, 2: 2}), pd.Series([5], index=[1]))


@pytest.mark.parametrize(("rul", "expected"), [(31, 0), (30, 1), (0, 1)])
def test_failure_label_boundary(rul, expected):
    frame = pd.DataFrame({RUL_COLUMN: [rul]})

    assert add_failure_label(frame, horizon=30)[LABEL_COLUMN].item() == expected


def test_failure_label_rejects_invalid_horizon():
    with pytest.raises(ValueError):
        add_failure_label(pd.DataFrame({RUL_COLUMN: [1]}), horizon=0)


def test_failure_label_requires_rul_column():
    with pytest.raises(KeyError):
        add_failure_label(pd.DataFrame({"cycle": [1]}), horizon=30)


def test_label_functions_do_not_mutate_their_input(make_trajectories):
    trajectories = make_trajectories({1: 3})
    before = trajectories.copy()

    add_failure_label(add_rul_train(trajectories), horizon=2)

    pd.testing.assert_frame_equal(trajectories, before)
