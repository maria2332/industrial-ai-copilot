"""Versioned storage of trained artifacts: models/<name>/<version>/{model.joblib, metadata.json}.

- A version is immutable: saving over an existing one fails unless explicitly forced.
- metadata.json records what is needed to trust and reproduce the artifact: creation time,
  library versions, a fingerprint of the training data, the decision policy (thresholds)
  and the evaluation results.
- joblib files are pickles: loading one executes code, so only load artifacts you created.
  They are also tied to library versions, which is why a mismatch raises a warning.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import warnings
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import joblib

from industrial_ai.config.paths import MODELS_DIR

DEFAULT_VERSION = "1.0.0"
MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
TRACKED_PACKAGES = ("scikit-learn", "pandas", "numpy", "joblib")
_VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")


def parse_version(text: str) -> tuple[int, int, int]:
    """Parse a MAJOR.MINOR.PATCH version, so that 1.10.0 sorts after 1.9.0."""
    match = _VERSION_PATTERN.match(text)
    if not match:
        raise ValueError(f"invalid version {text!r}: expected MAJOR.MINOR.PATCH, e.g. 1.0.0")
    major, minor, patch = (int(part) for part in match.groups())
    return major, minor, patch


def environment() -> dict[str, str]:
    """Python and library versions the artifact is created with."""
    packages = {}
    for package in TRACKED_PACKAGES:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not installed"
    return {"python": platform.python_version(), **packages}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def list_versions(name: str, models_dir: Path = MODELS_DIR) -> list[str]:
    """Saved versions of an artifact, oldest first."""
    directory = Path(models_dir) / name
    if not directory.exists():
        return []
    found = [
        child.name
        for child in directory.iterdir()
        if child.is_dir() and _VERSION_PATTERN.match(child.name)
    ]
    return sorted(found, key=parse_version)


def save_artifact(
    estimator: object,
    name: str,
    version_id: str,
    metadata: dict[str, Any],
    models_dir: Path = MODELS_DIR,
    overwrite: bool = False,
) -> Path:
    """Save an estimator and its metadata; return the artifact directory."""
    parse_version(version_id)
    directory = Path(models_dir) / name / version_id
    if directory.exists():
        if not overwrite:
            raise FileExistsError(
                f"{name} {version_id} already exists; use a new version or overwrite=True"
            )
        shutil.rmtree(directory)
    directory.mkdir(parents=True)

    joblib.dump(estimator, directory / MODEL_FILE)
    record = {
        "name": name,
        "version": version_id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": environment(),
        **metadata,
    }
    (directory / METADATA_FILE).write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    return directory


def load_artifact(
    name: str, version_id: str | None = None, models_dir: Path = MODELS_DIR
) -> tuple[object, dict[str, Any]]:
    """Load an estimator and its metadata (the latest version when version_id is None)."""
    if version_id is None:
        versions = list_versions(name, models_dir)
        if not versions:
            raise FileNotFoundError(
                f"no saved versions of {name!r} in {models_dir}: run `python scripts/train.py`"
            )
        version_id = versions[-1]
    directory = Path(models_dir) / name / version_id
    if not directory.exists():
        raise FileNotFoundError(f"{name} {version_id} not found in {models_dir}")

    metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
    saved, current = metadata.get("environment", {}), environment()
    mismatched = [
        f"{package} {saved.get(package)} (now {current[package]})"
        for package in TRACKED_PACKAGES
        if saved.get(package) != current[package]
    ]
    if mismatched:
        warnings.warn(
            f"{name} {version_id} was saved with different library versions: "
            + ", ".join(mismatched)
            + ". Retrain it or pin those versions.",
            UserWarning,
            stacklevel=2,
        )
    return joblib.load(directory / MODEL_FILE), metadata
