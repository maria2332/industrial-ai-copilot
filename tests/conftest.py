"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from industrial_ai.ml.data import ALL_COLUMNS


def _trajectories(lengths: dict[int, int], seed: int = 0) -> pd.DataFrame:
    """Random trajectories with the C-MAPSS column layout: {unit: number of cycles}."""
    rng = np.random.default_rng(seed)
    frames = []
    for unit, n_cycles in lengths.items():
        values = rng.normal(size=(n_cycles, len(ALL_COLUMNS) - 2))
        frame = pd.DataFrame(values, columns=ALL_COLUMNS[2:])
        frame.insert(0, "cycle", np.arange(1, n_cycles + 1))
        frame.insert(0, "unit", unit)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def make_trajectories() -> Callable[..., pd.DataFrame]:
    """Factory of synthetic trajectories, so tests never need the real dataset."""
    return _trajectories


@pytest.fixture
def write_cmapss_file(tmp_path: Path) -> Callable[[str, pd.DataFrame], Path]:
    """Write a frame like the original files: space-separated, trailing spaces, no header."""

    def _write(name: str, frame: pd.DataFrame) -> Path:
        path = tmp_path / name
        rows = [
            " ".join(str(value) for value in row) + "  " for row in frame.itertuples(index=False)
        ]
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return path

    return _write
