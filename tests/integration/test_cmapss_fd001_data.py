"""Checks on the real FD001 files. Skipped when the data is not downloaded (e.g. in CI)."""

import pytest

from industrial_ai.config.paths import RAW_CMAPSS_DIR
from industrial_ai.ml.data import load_rul, load_trajectories

pytestmark = pytest.mark.skipif(
    not (RAW_CMAPSS_DIR / "train_FD001.txt").exists(),
    reason="C-MAPSS not downloaded: run `python scripts/download_data.py`",
)


def test_fd001_train_has_expected_size():
    train = load_trajectories("FD001", "train")

    assert len(train) == 20_631
    assert train["unit"].nunique() == 100


def test_fd001_test_and_rul_files_are_consistent():
    test = load_trajectories("FD001", "test")
    rul = load_rul("FD001")

    assert len(test) == 13_096
    assert test["unit"].nunique() == len(rul) == 100
