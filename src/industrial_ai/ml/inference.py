"""Inference with the saved artifacts: the entry point for the API and the agent tools.

Both models need the complete history of one unit (from cycle 1), because the features
compare each cycle with the unit's own past. The result describes the last cycle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from industrial_ai.config.paths import MODELS_DIR
from industrial_ai.ml.anomaly import AnomalyDetector
from industrial_ai.ml.explain import top_contributions
from industrial_ai.ml.models import model_inputs
from industrial_ai.ml.registry import load_artifact

CLASSIFIER_NAME = "failure_classifier"
DETECTOR_NAME = "anomaly_detector"


@dataclass(frozen=True)
class FailureRisk:
    """Risk of failure within the horizon at the last cycle of a unit's history."""

    cycle: int
    probability: float
    threshold: float
    high_risk: bool
    horizon_cycles: int
    top_features: list[tuple[str, float]]
    model_version: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AnomalyStatus:
    """Deviation from healthy operation at the last cycle of a unit's history."""

    cycle: int
    margin: float
    anomalous_now: bool
    sustained_alarm: bool
    consecutive_cycles_required: int
    top_deviations: list[tuple[str, float]]
    model_version: str

    def as_dict(self) -> dict:
        return asdict(self)


def _single_unit_history(history: pd.DataFrame) -> pd.DataFrame:
    units = history["unit"].unique()
    if len(units) != 1:
        raise ValueError(f"expected the history of exactly one unit, got {len(units)}")
    return model_inputs(history).sort_values("cycle").reset_index(drop=True)


class Predictor:
    """Loads the trained classifier and detector once and answers per-unit questions."""

    def __init__(
        self,
        classifier: object,
        classifier_metadata: dict,
        detector: AnomalyDetector,
        detector_metadata: dict,
    ) -> None:
        self.classifier = classifier
        self.classifier_metadata = classifier_metadata
        self.detector = detector
        self.detector_metadata = detector_metadata

    @classmethod
    def load(cls, version_id: str | None = None, models_dir: Path = MODELS_DIR) -> Predictor:
        """Load both artifacts of the same version (the latest when version_id is None)."""
        classifier, classifier_metadata = load_artifact(CLASSIFIER_NAME, version_id, models_dir)
        detector, detector_metadata = load_artifact(
            DETECTOR_NAME, classifier_metadata["version"], models_dir
        )
        return cls(classifier, classifier_metadata, detector, detector_metadata)

    def predict_failure(self, history: pd.DataFrame, top_n: int = 3) -> FailureRisk:
        X = _single_unit_history(history)
        probability = float(self.classifier.predict_proba(X)[-1, 1])
        threshold = float(self.classifier_metadata["threshold"])
        contributions = top_contributions(self.classifier, X, n=top_n)
        last = contributions[contributions["row"] == X.index[-1]]
        return FailureRisk(
            cycle=int(X["cycle"].iloc[-1]),
            probability=probability,
            threshold=threshold,
            high_risk=probability >= threshold,
            horizon_cycles=int(self.classifier_metadata["horizon_cycles"]),
            top_features=list(zip(last["feature"], last["contribution"], strict=True)),
            model_version=self.classifier_metadata["version"],
        )

    def detect_anomaly(self, history: pd.DataFrame, top_n: int = 3) -> AnomalyStatus:
        X = _single_unit_history(history)
        margins = self.detector.margin(X)
        required = int(self.detector_metadata["alarm_consecutive_cycles"])
        recent = margins[-required:]
        deviations = self.detector.top_deviations(X, n=top_n)
        last = deviations[deviations["row"] == X.index[-1]]
        return AnomalyStatus(
            cycle=int(X["cycle"].iloc[-1]),
            margin=float(margins[-1]),
            anomalous_now=bool(margins[-1] >= 0),
            sustained_alarm=bool(len(recent) == required and np.all(recent >= 0)),
            consecutive_cycles_required=required,
            top_deviations=list(zip(last["feature"], last["zscore"], strict=True)),
            model_version=self.detector_metadata["version"],
        )
