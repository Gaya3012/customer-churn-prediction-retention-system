"""Model scoring and evaluation report helpers."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model: Any, features: Any, target: Any) -> dict[str, Any]:
    """Calculate threshold metrics, ROC-AUC, and a 2x2 confusion matrix."""
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    matrix = confusion_matrix(target, predictions, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1_score": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def select_best_model(results: dict[str, dict[str, Any]]) -> str:
    """Choose the model with the highest ROC-AUC, then F1 and accuracy."""
    if not results:
        raise ValueError("At least one model result is required.")
    return max(
        results,
        key=lambda name: (
            results[name]["roc_auc"],
            results[name]["f1_score"],
            results[name]["accuracy"],
        ),
    )
