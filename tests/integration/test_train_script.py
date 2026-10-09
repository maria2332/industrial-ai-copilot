"""End-to-end check of scripts/train.py on small synthetic files in the C-MAPSS format."""

import importlib.util
import json
from pathlib import Path

import pytest

from industrial_ai.ml.inference import CLASSIFIER_NAME, DETECTOR_NAME, Predictor
from industrial_ai.ml.labels import add_rul_train

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("train_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def data_dir(make_trajectories, write_cmapss_file):
    """Ten run-to-failure training units and four truncated test units."""
    train = add_rul_train(make_trajectories({unit: 90 for unit in range(1, 11)}, seed=1))
    train["sensor_4"] += 0.5 * (40 - train["rul"]).clip(lower=0)
    write_cmapss_file("train_FD001.txt", train.drop(columns="rul"))

    test = make_trajectories({unit: 60 + 5 * unit for unit in range(1, 5)}, seed=2)
    path = write_cmapss_file("test_FD001.txt", test)
    (path.parent / "RUL_FD001.txt").write_text("10\n25\n50\n80\n", encoding="utf-8")
    return path.parent


def test_training_saves_both_artifacts(tmp_path, data_dir):
    models_dir = tmp_path / "models"
    args = ["--data-dir", str(data_dir), "--models-dir", str(models_dir), "--folds", "2"]

    assert _load_script().main([*args, "--version", "0.1.0"]) == 0

    for name in (CLASSIFIER_NAME, DETECTOR_NAME):
        directory = models_dir / name / "0.1.0"
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        assert (directory / "model.joblib").exists()
        assert metadata["data"]["train_units"] == 10
        assert set(metadata["data"]["files_sha256"]) == {"train", "test", "RUL"}
    classifier_metadata = json.loads(
        (models_dir / CLASSIFIER_NAME / "0.1.0" / "metadata.json").read_text(encoding="utf-8")
    )
    assert 0.0 < classifier_metadata["threshold"] < 1.0
    assert classifier_metadata["horizon_cycles"] == 30
    assert classifier_metadata["cross_validation"]["out_of_fold_at_threshold"]["recall"] >= 0.9


def test_trained_artifacts_serve_predictions(tmp_path, data_dir, make_trajectories):
    models_dir = tmp_path / "models"
    args = ["--data-dir", str(data_dir), "--models-dir", str(models_dir), "--folds", "2"]
    _load_script().main([*args, "--version", "0.1.0"])

    predictor = Predictor.load(models_dir=models_dir)
    history = make_trajectories({7: 70}, seed=3)

    assert 0.0 <= predictor.predict_failure(history).probability <= 1.0
    assert predictor.detect_anomaly(history).consecutive_cycles_required == 3


def test_existing_versions_are_not_overwritten_by_default(tmp_path, data_dir):
    models_dir = tmp_path / "models"
    args = ["--data-dir", str(data_dir), "--models-dir", str(models_dir), "--folds", "2"]
    script = _load_script()
    script.main([*args, "--version", "0.1.0"])

    with pytest.raises(FileExistsError):
        script.main([*args, "--version", "0.1.0"])
    assert script.main([*args, "--version", "0.1.0", "--overwrite"]) == 0
