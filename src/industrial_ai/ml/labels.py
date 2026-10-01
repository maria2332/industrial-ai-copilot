"""Remaining-useful-life (RUL) and failure-horizon labels for C-MAPSS trajectories.

Convention: RUL is the number of operating cycles left after the current one. In the
training set every unit runs to failure, so its last recorded cycle has RUL = 0.
"""

from __future__ import annotations

import pandas as pd

RUL_COLUMN = "rul"
LABEL_COLUMN = "failure_within_horizon"


def add_rul_train(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the RUL of run-to-failure trajectories: last cycle of the unit minus current cycle."""
    out = frame.copy()
    last_cycle = out.groupby("unit")["cycle"].transform("max")
    out[RUL_COLUMN] = (last_cycle - out["cycle"]).astype("int64")
    return out


def add_rul_test(frame: pd.DataFrame, rul_at_last_cycle: pd.Series) -> pd.DataFrame:
    """Add the RUL of truncated test trajectories.

    The RUL file only gives the remaining life after the last observed cycle of each unit,
    so every earlier cycle is reconstructed as: RUL_last + (last observed cycle - cycle).
    """
    missing_units = set(frame["unit"].unique()) - set(rul_at_last_cycle.index)
    if missing_units:
        raise ValueError(f"RUL missing for units: {sorted(missing_units)[:10]}")

    out = frame.copy()
    last_cycle = out.groupby("unit")["cycle"].transform("max")
    rul_last = out["unit"].map(rul_at_last_cycle)
    out[RUL_COLUMN] = (rul_last + last_cycle - out["cycle"]).astype("int64")
    return out


def add_failure_label(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Add a binary label: 1 if the unit fails within `horizon` cycles (RUL <= horizon)."""
    if horizon < 1:
        raise ValueError("horizon must be a positive number of cycles")
    if RUL_COLUMN not in frame.columns:
        raise KeyError(f"column {RUL_COLUMN!r} not found; add RUL before labelling")

    out = frame.copy()
    out[LABEL_COLUMN] = (out[RUL_COLUMN] <= horizon).astype("int64")
    return out
