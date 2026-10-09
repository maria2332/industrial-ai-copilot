"""Model comparison with cross-validation grouped by unit.

Every candidate is evaluated with the same folds, and each row's out-of-fold score comes
from a model that never saw that row's unit. Those scores are what the decision threshold
is chosen on, so the threshold is never tuned on data the model was trained with.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from industrial_ai.ml.split import check_units_disjoint


@dataclass(frozen=True)
class CrossValidationResult:
    """Out-of-fold scores and per-fold metrics of one candidate."""

    oof_scores: np.ndarray
    fold_pr_auc: np.ndarray
    fold_roc_auc: np.ndarray
    fit_seconds: float

    def summary(self) -> dict[str, float]:
        """Mean and standard deviation across folds, plus total training time."""
        return {
            "pr_auc_mean": float(self.fold_pr_auc.mean()),
            "pr_auc_std": float(self.fold_pr_auc.std()),
            "roc_auc_mean": float(self.fold_roc_auc.mean()),
            "roc_auc_std": float(self.fold_roc_auc.std()),
            "fit_seconds": self.fit_seconds,
        }


def cross_validate_by_unit(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    target: pd.Series,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
) -> CrossValidationResult:
    """Fit a fresh copy of the pipeline on each training fold and score its validation fold.

    `frame` holds the model inputs (unit, cycle and sensors). The folds must keep units
    disjoint and cover every row exactly once (as unit_group_kfold does).
    """
    if len(frame) != len(target):
        raise ValueError("frame and target must have the same length")

    oof_scores = np.full(len(frame), np.nan)
    fold_pr_auc, fold_roc_auc = [], []
    start = time.perf_counter()
    for train_idx, validation_idx in folds:
        check_units_disjoint(frame, train_idx, validation_idx)
        if not np.isnan(oof_scores[validation_idx]).all():
            raise ValueError("a row appears in more than one validation fold")

        model = clone(pipeline).fit(frame.iloc[train_idx], target.iloc[train_idx])
        scores = model.predict_proba(frame.iloc[validation_idx])[:, 1]
        oof_scores[validation_idx] = scores

        y_validation = target.iloc[validation_idx]
        fold_pr_auc.append(average_precision_score(y_validation, scores))
        fold_roc_auc.append(roc_auc_score(y_validation, scores))

    if np.isnan(oof_scores).any():
        raise ValueError("the folds must cover every row exactly once")
    return CrossValidationResult(
        oof_scores=oof_scores,
        fold_pr_auc=np.array(fold_pr_auc),
        fold_roc_auc=np.array(fold_roc_auc),
        fit_seconds=time.perf_counter() - start,
    )


def simplest_within_one_std(summary: pd.DataFrame, simplest_first: Sequence[str]) -> str:
    """Pick the simplest candidate whose mean PR-AUC is within one standard deviation of the best.

    `summary` is indexed by candidate name with columns pr_auc_mean and pr_auc_std. The band
    is the best candidate's mean minus its own standard deviation across folds. Differences
    smaller than the fold-to-fold noise are not treated as evidence for a more complex model.
    """
    candidates = [name for name in simplest_first if name in summary.index]
    if not candidates:
        raise ValueError("none of the ordered candidates is in the summary")

    ranked = summary.loc[candidates]
    best = ranked["pr_auc_mean"].idxmax()
    bar = ranked.loc[best, "pr_auc_mean"] - ranked.loc[best, "pr_auc_std"]
    return next(name for name in candidates if ranked.loc[name, "pr_auc_mean"] >= bar)
