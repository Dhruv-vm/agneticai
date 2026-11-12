"""Phase 4.3 four-class, selective, coverage, and statistical-interval metrics."""

from __future__ import annotations

import math
from typing import Any

from src.attribution.models import AttributionLabel
from src.evaluation.metrics import safe_div

ALL_CLASSES: tuple[str, ...] = (
    AttributionLabel.DENIAL_SENSITIVE.value,
    AttributionLabel.DENIAL_INDEPENDENT.value,
    AttributionLabel.AMBIGUOUS.value,
    AttributionLabel.NOT_EVALUABLE.value,
)


def wilson_ci(successes: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    """Return 95% Wilson score confidence interval ``(low, high)`` or ``(None, None)``."""
    if total <= 0:
        return None, None
    p = successes / total
    denom = 1.0 + (z * z) / total
    center = (p + (z * z) / (2.0 * total)) / denom
    half = (z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * total)) / total)) / denom
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def compute_four_class_metrics(
    records: list[dict[str, Any]],
    pred_key: str = "predicted_class",
    gt_key: str = "ground_truth_class",
) -> dict[str, Any]:
    """Compute full 4-class confusion matrix, macro-F1, coverage, and selective accuracy."""
    matrix: dict[str, dict[str, int]] = {
        gt: {pred: 0 for pred in ALL_CLASSES} for gt in ALL_CLASSES
    }
    for rec in records:
        gt = rec[gt_key]
        pred = rec[pred_key]
        matrix[gt][pred] += 1

    total = len(records)
    overall_correct = sum(matrix[c][c] for c in ALL_CLASSES)
    overall_acc = safe_div(overall_correct, total)
    overall_ci = wilson_ci(overall_correct, total)

    per_class: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    recalls_present: list[float] = []

    for cls in ALL_CLASSES:
        tp = matrix[cls][cls]
        fp = sum(matrix[other][cls] for other in ALL_CLASSES if other != cls)
        fn = sum(matrix[cls][other] for other in ALL_CLASSES if other != cls)
        support = tp + fn
        prec = safe_div(tp, tp + fp)
        rec = safe_div(tp, tp + fn)
        f1 = (
            round(2.0 * prec * rec / (prec + rec), 4)
            if prec is not None and rec is not None and (prec + rec) > 0
            else None
        )
        per_class[cls] = {
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(prec, 4) if prec is not None else None,
            "recall": round(rec, 4) if rec is not None else None,
            "f1": f1,
        }
        if support > 0:
            if f1 is not None:
                f1_values.append(f1)
            if rec is not None:
                recalls_present.append(rec)

    macro_f1 = round(sum(f1_values) / len(f1_values), 4) if f1_values else None
    balanced_acc = (
        round(sum(recalls_present) / len(recalls_present), 4)
        if recalls_present
        else None
    )

    # Selective prediction: predictions in {DENIAL_SENSITIVE, DENIAL_INDEPENDENT}
    decisive_classes = {
        AttributionLabel.DENIAL_SENSITIVE.value,
        AttributionLabel.DENIAL_INDEPENDENT.value,
    }
    covered = [r for r in records if r[pred_key] in decisive_classes]
    covered_correct = sum(1 for r in covered if r[pred_key] == r[gt_key])
    coverage = safe_div(len(covered), total)
    selective_acc = safe_div(covered_correct, len(covered))
    selective_ci = wilson_ci(covered_correct, len(covered))

    amb_count = sum(
        1 for r in records if r[pred_key] == AttributionLabel.AMBIGUOUS.value
    )
    ne_count = sum(
        1 for r in records if r[pred_key] == AttributionLabel.NOT_EVALUABLE.value
    )

    # Binary DENIAL_SENSITIVE detection & FAR over non-DENIAL_SENSITIVE ground truth
    ds_cls = AttributionLabel.DENIAL_SENSITIVE.value
    di_cls = AttributionLabel.DENIAL_INDEPENDENT.value
    ds_metrics = per_class[ds_cls]
    far_den = sum(matrix[di_cls].values())
    far_num = matrix[di_cls][ds_cls]
    far = safe_div(far_num, far_den)

    return {
        "total": total,
        "confusion_matrix_4x4": matrix,
        "overall_4class_accuracy": round(overall_acc, 4) if overall_acc is not None else None,
        "overall_4class_accuracy_ci95": overall_ci,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "per_class": per_class,
        "binary_denial_sensitive_precision": ds_metrics["precision"],
        "binary_denial_sensitive_recall": ds_metrics["recall"],
        "binary_denial_sensitive_f1": ds_metrics["f1"],
        "false_attribution_rate_on_independent": round(far, 4) if far is not None else None,
        "coverage": round(coverage, 4) if coverage is not None else None,
        "selective_accuracy": round(selective_acc, 4) if selective_acc is not None else None,
        "selective_accuracy_ci95": selective_ci,
        "ambiguous_rate": round(safe_div(amb_count, total) or 0.0, 4),
        "not_evaluable_rate": round(safe_div(ne_count, total) or 0.0, 4),
    }
