"""Cross-validation splits that keep every unit entirely on one side of the split."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


def unit_group_kfold(frame: pd.DataFrame, n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    """Folds in which each unit appears in exactly one validation fold.

    Simulates the deployment question: how does the model behave on an unseen engine?
    """
    n_units = frame["unit"].nunique()
    if not 2 <= n_splits <= n_units:
        raise ValueError(f"n_splits must be between 2 and the number of units ({n_units})")
    return list(GroupKFold(n_splits=n_splits).split(frame, groups=frame["unit"]))


def check_units_disjoint(
    frame: pd.DataFrame, train_idx: np.ndarray, validation_idx: np.ndarray
) -> None:
    """Raise if any unit has rows on both sides of a split."""
    shared = set(frame["unit"].iloc[train_idx]) & set(frame["unit"].iloc[validation_idx])
    if shared:
        raise ValueError(f"units in both train and validation: {sorted(shared)[:10]}")
