"""Tests for versioned artifact storage."""

import json

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from industrial_ai.ml.registry import (
    environment,
    list_versions,
    load_artifact,
    parse_version,
    save_artifact,
)


@pytest.fixture
def model():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 2))
    return LogisticRegression().fit(X, (X[:, 0] > 0).astype(int)), X


def test_save_and_load_round_trip(tmp_path, model):
    estimator, X = model

    directory = save_artifact(estimator, "clf", "1.0.0", {"threshold": 0.7}, models_dir=tmp_path)
    loaded, metadata = load_artifact("clf", "1.0.0", models_dir=tmp_path)

    np.testing.assert_allclose(loaded.predict_proba(X), estimator.predict_proba(X))
    assert (directory / "model.joblib").exists()
    assert metadata["threshold"] == 0.7
    assert metadata["name"] == "clf"
    assert metadata["version"] == "1.0.0"
    assert metadata["environment"] == environment()
    assert "created_at" in metadata


def test_versions_are_immutable(tmp_path, model):
    save_artifact(model[0], "clf", "1.0.0", {}, models_dir=tmp_path)

    with pytest.raises(FileExistsError):
        save_artifact(model[0], "clf", "1.0.0", {}, models_dir=tmp_path)
    save_artifact(model[0], "clf", "1.0.0", {"note": "forced"}, models_dir=tmp_path, overwrite=True)
    assert load_artifact("clf", "1.0.0", models_dir=tmp_path)[1]["note"] == "forced"


def test_latest_version_uses_numeric_order(tmp_path, model):
    for version_id in ("1.9.0", "1.10.0", "1.2.0"):
        save_artifact(model[0], "clf", version_id, {}, models_dir=tmp_path)

    assert list_versions("clf", models_dir=tmp_path) == ["1.2.0", "1.9.0", "1.10.0"]
    assert load_artifact("clf", models_dir=tmp_path)[1]["version"] == "1.10.0"


def test_library_mismatch_warns(tmp_path, model):
    directory = save_artifact(model[0], "clf", "1.0.0", {}, models_dir=tmp_path)
    metadata_file = directory / "metadata.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    metadata["environment"]["scikit-learn"] = "0.0.1"
    metadata_file.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.warns(UserWarning, match="scikit-learn 0.0.1"):
        load_artifact("clf", "1.0.0", models_dir=tmp_path)


def test_missing_artifacts_explain_how_to_create_them(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        load_artifact("clf", models_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        load_artifact("clf", "2.0.0", models_dir=tmp_path)


@pytest.mark.parametrize("text", ["1.0", "v1.0.0", "1.0.0-beta", ""])
def test_invalid_versions_are_rejected(text):
    with pytest.raises(ValueError, match="MAJOR.MINOR.PATCH"):
        parse_version(text)
