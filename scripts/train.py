"""Train the failure classifier and the anomaly detector, evaluate them, save versioned artifacts.

Repeats the decisions taken in notebook 02 without the exploration: same features, same
unit folds, same selection rules. The API only loads what this script saves; it never trains.

Usage:
    python scripts/train.py                    # version from MODEL_VERSION, else 1.0.0
    python scripts/train.py --version 1.1.0
    python scripts/train.py --overwrite        # replace an existing version
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from industrial_ai.config.paths import MODELS_DIR, RAW_CMAPSS_DIR
from industrial_ai.evaluation.metrics import (
    expected_calibration_error,
    first_alarms,
    last_cycle_mask,
    reliability_table,
    select_threshold,
    summarize_first_alarms,
    threshold_metrics,
)
from industrial_ai.ml.anomaly import AnomalyDetector, out_of_fold_margins
from industrial_ai.ml.data import load_rul, load_trajectories
from industrial_ai.ml.features import feature_names
from industrial_ai.ml.inference import CLASSIFIER_NAME, DETECTOR_NAME
from industrial_ai.ml.labels import LABEL_COLUMN, add_failure_label, add_rul_test, add_rul_train
from industrial_ai.ml.models import SELECTED_FEATURES, build_pipeline, model_inputs
from industrial_ai.ml.registry import DEFAULT_VERSION, file_sha256, save_artifact
from industrial_ai.ml.selection import cross_validate_by_unit
from industrial_ai.ml.split import unit_group_kfold

SUBSET = "FD001"
SELECTED_MODEL = "logistic_regression"  # D-015
SELECTED_DETECTOR = "max_zscore"  # D-021
MIN_RECALL = 0.90  # D-019
ALARM_CONSECUTIVE_CYCLES = 3  # D-021


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", default=os.environ.get("MODEL_VERSION", DEFAULT_VERSION))
    parser.add_argument("--horizon", type=int, default=30, help="failure horizon in cycles")
    parser.add_argument("--folds", type=int, default=5, help="unit-grouped CV folds")
    parser.add_argument("--data-dir", type=Path, default=RAW_CMAPSS_DIR)
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing version")
    return parser.parse_args(argv)


def _rounded(metrics: dict[str, float]) -> dict[str, float]:
    return {name: round(float(value), 4) for name, value in metrics.items()}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    train = add_failure_label(
        add_rul_train(load_trajectories(SUBSET, "train", args.data_dir)), args.horizon
    )
    test = add_failure_label(
        add_rul_test(
            load_trajectories(SUBSET, "test", args.data_dir), load_rul(SUBSET, args.data_dir)
        ),
        args.horizon,
    )
    X, y = model_inputs(train), train[LABEL_COLUMN]
    X_test, y_test = model_inputs(test), test[LABEL_COLUMN].to_numpy()
    folds = unit_group_kfold(train, n_splits=args.folds)
    last = last_cycle_mask(test).to_numpy()

    data = {
        "subset": SUBSET,
        "files_sha256": {
            name: file_sha256(Path(args.data_dir) / f"{name}_{SUBSET}.txt")
            for name in ("train", "test", "RUL")
        },
        "train_units": int(train["unit"].nunique()),
        "train_rows": len(train),
    }

    # Failure classifier: out-of-fold scores choose the threshold, then refit on all units.
    print(f"Cross-validating {SELECTED_MODEL} ({args.folds} unit folds)...")
    pipeline = build_pipeline(SELECTED_MODEL)
    cv = cross_validate_by_unit(pipeline, X, y, folds)
    threshold = select_threshold(y, cv.oof_scores, MIN_RECALL)
    classifier = build_pipeline(SELECTED_MODEL).fit(X, y)
    test_scores = classifier.predict_proba(X_test)[:, 1]
    classifier_metadata = {
        "model": SELECTED_MODEL,
        "horizon_cycles": args.horizon,
        "threshold": round(threshold, 6),
        "threshold_rule": f"highest precision with out-of-fold recall >= {MIN_RECALL} (D-019)",
        "features": feature_names(SELECTED_FEATURES),
        "data": data,
        "cross_validation": {
            "folds": args.folds,
            **_rounded(cv.summary()),
            "out_of_fold_at_threshold": _rounded(threshold_metrics(y, cv.oof_scores, threshold)),
            "expected_calibration_error": round(
                expected_calibration_error(reliability_table(y, cv.oof_scores)), 4
            ),
        },
        "test": {
            "every_cycle": _rounded(threshold_metrics(y_test, test_scores, threshold)),
            "last_cycle_per_unit": _rounded(
                threshold_metrics(y_test[last], test_scores[last], threshold)
            ),
        },
    }

    # Anomaly detector: out-of-fold evaluation, then refit on the healthy cycles of all units.
    print(f"Cross-validating the {SELECTED_DETECTOR} anomaly detector...")
    detector_template = AnomalyDetector(method=SELECTED_DETECTOR)
    margins = out_of_fold_margins(detector_template, X, folds)
    healthy = detector_template.in_healthy_window(train).to_numpy()
    first_alarm_rul = first_alarms(train, margins, 0.0, ALARM_CONSECUTIVE_CYCLES)
    detector = AnomalyDetector(method=SELECTED_DETECTOR).fit(X)
    detector_metadata = {
        "method": SELECTED_DETECTOR,
        "healthy_window_cycles": list(detector.healthy_window()),
        "false_alarm_budget": detector.false_alarm_rate,
        "threshold": round(detector.threshold_, 6),
        "alarm_consecutive_cycles": ALARM_CONSECUTIVE_CYCLES,
        "features": detector.feature_names_,
        "data": data,
        "cross_validation": {
            "folds": args.folds,
            "false_alarm_rate_healthy_cycles": round(float(np.mean(margins[healthy] >= 0)), 4),
            "first_alarms": _rounded(
                summarize_first_alarms(first_alarm_rul["rul_at_first_alarm"], args.horizon)
            ),
        },
    }

    for estimator, name, metadata in (
        (classifier, CLASSIFIER_NAME, classifier_metadata),
        (detector, DETECTOR_NAME, detector_metadata),
    ):
        path = save_artifact(
            estimator, name, args.version, metadata, args.models_dir, overwrite=args.overwrite
        )
        print(f"Saved {name} {args.version} to {path}")

    oof = classifier_metadata["cross_validation"]
    print(
        f"Classifier: PR-AUC {oof['pr_auc_mean']:.3f} ± {oof['pr_auc_std']:.3f}, "
        f"threshold {threshold:.3f}, test last-cycle recall "
        f"{classifier_metadata['test']['last_cycle_per_unit']['recall']:.3f}"
    )
    alarms = detector_metadata["cross_validation"]
    print(
        f"Detector: healthy false alarms {alarms['false_alarm_rate_healthy_cycles']:.3f}, "
        f"median warning {alarms['first_alarms']['median RUL at first alarm']:.0f} cycles"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
