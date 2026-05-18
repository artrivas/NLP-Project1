"""Classification metrics for text classification experiments."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute accuracy and macro/weighted precision, recall, and F1 metrics."""
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
    }


def compute_metrics_for_trainer(eval_pred: Any) -> dict[str, float]:
    """Compute metrics from Hugging Face Trainer evaluation predictions."""
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]

    predictions = np.argmax(logits, axis=-1)
    return compute_classification_metrics(np.asarray(labels), np.asarray(predictions))
