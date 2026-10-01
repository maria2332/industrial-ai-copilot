"""Tests for loading and validating C-MAPSS files."""

import numpy as np
import pandas as pd
import pytest

from industrial_ai.ml.data import (
    ALL_COLUMNS,
    DataValidationError,
    load_rul,
    load_trajectories,
    validate_trajectories,
)


def test_load_trajectories_parses_whitespace_format(make_trajectories, write_cmapss_file):
    frame = make_trajectories({1: 5, 2: 3})
    path = write_cmapss_file("train_FD001.txt", frame)

    loaded = load_trajectories("FD001", "train", data_dir=path.parent)

    assert list(loaded.columns) == ALL_COLUMNS
    assert len(loaded) == 8
    assert loaded["unit"].dtype == np.int64
    np.testing.assert_allclose(loaded["sensor_1"], frame["sensor_1"])


def test_load_trajectories_missing_file_explains_how_to_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data.py"):
        load_trajectories("FD001", "train", data_dir=tmp_path)


def test_load_trajectories_rejects_unknown_subset_or_split(tmp_path):
    with pytest.raises(ValueError, match="subset"):
        load_trajectories("FD009", "train", data_dir=tmp_path)
    with pytest.raises(ValueError, match="split"):
        load_trajectories("FD001", "validation", data_dir=tmp_path)


def test_load_trajectories_rejects_wrong_column_count(make_trajectories, write_cmapss_file):
    frame = make_trajectories({1: 3}).iloc[:, :-1]
    path = write_cmapss_file("train_FD001.txt", frame)

    with pytest.raises(DataValidationError, match="columns"):
        load_trajectories("FD001", "train", data_dir=path.parent)


def test_validation_accepts_clean_data(make_trajectories):
    assert validate_trajectories(make_trajectories({1: 4, 2: 6})) == []


def test_validation_detects_missing_values(make_trajectories):
    frame = make_trajectories({1: 4})
    frame.loc[2, "sensor_7"] = np.nan

    assert any("missing values" in problem for problem in validate_trajectories(frame))


def test_validation_detects_cycle_gap(make_trajectories):
    frame = make_trajectories({1: 5})
    frame = frame[frame["cycle"] != 3]

    assert any("contiguous" in problem for problem in validate_trajectories(frame))


def test_validation_detects_duplicated_cycle(make_trajectories):
    frame = make_trajectories({1: 3})
    frame = pd.concat([frame, frame.iloc[[1]]], ignore_index=True)

    assert any("contiguous" in problem for problem in validate_trajectories(frame))


def test_validation_detects_unit_not_starting_at_cycle_one(make_trajectories):
    frame = make_trajectories({1: 5})
    frame = frame[frame["cycle"] > 1]

    assert any("cycle 1" in problem for problem in validate_trajectories(frame))


def test_load_rul_is_indexed_by_unit_starting_at_one(write_cmapss_file):
    path = write_cmapss_file("RUL_FD001.txt", pd.DataFrame({"rul": [112, 98, 69]}))

    rul = load_rul("FD001", data_dir=path.parent)

    assert rul.index.tolist() == [1, 2, 3]
    assert rul.tolist() == [112, 98, 69]


def test_load_rul_rejects_negative_values(write_cmapss_file):
    path = write_cmapss_file("RUL_FD001.txt", pd.DataFrame({"rul": [10, -1]}))

    with pytest.raises(DataValidationError):
        load_rul("FD001", data_dir=path.parent)
