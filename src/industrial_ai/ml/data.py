"""Loading and structural validation of the NASA C-MAPSS turbofan dataset.

Each file contains one row per (unit, cycle): the engine id, the operating cycle,
3 operational settings and 21 sensor channels, separated by whitespace.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from industrial_ai.config.paths import RAW_CMAPSS_DIR

ID_COLUMNS = ["unit", "cycle"]
SETTING_COLUMNS = [f"setting_{i}" for i in range(1, 4)]
SENSOR_COLUMNS = [f"sensor_{i}" for i in range(1, 22)]
ALL_COLUMNS = ID_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
SPLITS = ("train", "test")

# Sensor meaning as listed in Saxena et al. (2008), Table 2, following the column order
# commonly assumed in the literature. The dataset readme does not name the sensors itself.
SENSOR_DESCRIPTIONS: dict[str, str] = {
    "sensor_1": "T2 - total temperature at fan inlet (°R)",
    "sensor_2": "T24 - total temperature at LPC outlet (°R)",
    "sensor_3": "T30 - total temperature at HPC outlet (°R)",
    "sensor_4": "T50 - total temperature at LPT outlet (°R)",
    "sensor_5": "P2 - pressure at fan inlet (psia)",
    "sensor_6": "P15 - total pressure in bypass duct (psia)",
    "sensor_7": "P30 - total pressure at HPC outlet (psia)",
    "sensor_8": "Nf - physical fan speed (rpm)",
    "sensor_9": "Nc - physical core speed (rpm)",
    "sensor_10": "epr - engine pressure ratio (P50/P2)",
    "sensor_11": "Ps30 - static pressure at HPC outlet (psia)",
    "sensor_12": "phi - ratio of fuel flow to Ps30 (pps/psi)",
    "sensor_13": "NRf - corrected fan speed (rpm)",
    "sensor_14": "NRc - corrected core speed (rpm)",
    "sensor_15": "BPR - bypass ratio",
    "sensor_16": "farB - burner fuel-air ratio",
    "sensor_17": "htBleed - bleed enthalpy",
    "sensor_18": "Nf_dmd - demanded fan speed (rpm)",
    "sensor_19": "PCNfR_dmd - demanded corrected fan speed (rpm)",
    "sensor_20": "W31 - HPT coolant bleed (lbm/s)",
    "sensor_21": "W32 - LPT coolant bleed (lbm/s)",
}


class DataValidationError(ValueError):
    """Raised when a C-MAPSS file does not have the expected structure."""


def _check_subset(subset: str) -> None:
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}; expected one of {SUBSETS}")


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python scripts/download_data.py` first.")


def load_trajectories(
    subset: str = "FD001", split: str = "train", data_dir: Path = RAW_CMAPSS_DIR
) -> pd.DataFrame:
    """Load and validate the train or test trajectories of a C-MAPSS subset.

    Returns one row per (unit, cycle), sorted by unit and cycle.
    Raises DataValidationError if the file does not have the expected structure.
    """
    _check_subset(subset)
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {SPLITS}")
    path = Path(data_dir) / f"{split}_{subset}.txt"
    _require_file(path)

    frame = pd.read_csv(path, sep=r"\s+", header=None)
    if frame.shape[1] != len(ALL_COLUMNS):
        raise DataValidationError(
            f"{path.name}: expected {len(ALL_COLUMNS)} columns, found {frame.shape[1]}"
        )
    frame.columns = ALL_COLUMNS

    problems = validate_trajectories(frame)
    if problems:
        raise DataValidationError(f"{path.name}: " + "; ".join(problems))

    frame = frame.astype({"unit": "int64", "cycle": "int64"})
    return frame.sort_values(ID_COLUMNS).reset_index(drop=True)


def load_rul(subset: str = "FD001", data_dir: Path = RAW_CMAPSS_DIR) -> pd.Series:
    """Load the true remaining useful life after the last observed cycle of each test unit.

    The returned Series is indexed by unit (starting at 1, in file order).
    """
    _check_subset(subset)
    path = Path(data_dir) / f"RUL_{subset}.txt"
    _require_file(path)

    values = pd.read_csv(path, sep=r"\s+", header=None).iloc[:, 0]
    if values.isna().any() or (values < 0).any():
        raise DataValidationError(f"{path.name}: RUL values must be non-negative numbers")
    index = pd.RangeIndex(1, len(values) + 1, name="unit")
    return pd.Series(values.to_numpy(dtype="int64"), index=index, name="rul_at_last_cycle")


def validate_trajectories(frame: pd.DataFrame) -> list[str]:
    """Return a list of structural problems found in a trajectories frame (empty if valid)."""
    missing_columns = [column for column in ALL_COLUMNS if column not in frame.columns]
    if missing_columns:
        return [f"missing columns: {missing_columns}"]

    problems: list[str] = []
    n_missing = int(frame[ALL_COLUMNS].isna().sum().sum())
    if n_missing:
        problems.append(f"{n_missing} missing values")

    measurements = frame[SETTING_COLUMNS + SENSOR_COLUMNS].to_numpy(dtype=float)
    if np.isinf(measurements).any():
        problems.append("infinite values in settings or sensors")

    if (frame["unit"] < 1).any():
        problems.append("unit ids must be positive")

    ordered = frame.sort_values(ID_COLUMNS)
    first_cycle = ordered.groupby("unit")["cycle"].min()
    late_starters = first_cycle[first_cycle != 1].index.tolist()
    if late_starters:
        problems.append(f"units not starting at cycle 1: {late_starters[:10]}")

    steps = ordered.groupby("unit")["cycle"].diff().dropna()
    if (steps != 1).any():
        problems.append("cycles are not contiguous within some units (gaps or duplicates)")

    return problems
