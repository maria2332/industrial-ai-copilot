"""Tests for unit-grouped cross-validation."""

import numpy as np
import pytest

from industrial_ai.ml.split import check_units_disjoint, unit_group_kfold


def test_folds_never_share_units(make_trajectories):
    trajectories = make_trajectories({unit: 5 + unit for unit in range(1, 11)})

    for train_idx, validation_idx in unit_group_kfold(trajectories, n_splits=5):
        check_units_disjoint(trajectories, train_idx, validation_idx)


def test_every_unit_is_validated_exactly_once(make_trajectories):
    trajectories = make_trajectories({unit: 4 for unit in range(1, 11)})

    validated = np.concatenate(
        [trajectories["unit"].iloc[idx].unique() for _, idx in unit_group_kfold(trajectories, 5)]
    )

    assert sorted(validated) == list(range(1, 11))


def test_overlapping_units_are_detected(make_trajectories):
    trajectories = make_trajectories({1: 4, 2: 4})

    with pytest.raises(ValueError, match="both train and validation"):
        check_units_disjoint(trajectories, np.array([0, 1, 4]), np.array([2, 5]))


def test_invalid_number_of_splits(make_trajectories):
    with pytest.raises(ValueError):
        unit_group_kfold(make_trajectories({1: 3, 2: 3}), n_splits=3)
