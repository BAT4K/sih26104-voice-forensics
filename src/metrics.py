"""Evaluation metrics for anti-spoofing.

The field reports Equal Error Rate, not accuracy.  Accuracy is
threshold-dependent and hides the operating point, which is the only thing a
bank actually cares about.  Everything here is pure NumPy so it can run on a
laptop with no GPU and no framework dependency.
"""

from __future__ import annotations

import numpy as np


def _roc(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the ROC without sklearn.

    Args:
        scores: Higher means more likely to be the positive class.
        labels: 1 for positive, 0 for negative.

    Returns:
        ``(fpr, tpr, thresholds)`` sorted by decreasing threshold.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(int)
    if scores.shape != labels.shape:
        raise ValueError(f"scores {scores.shape} and labels {labels.shape} must match")
    if scores.size == 0:
        raise ValueError("empty score array")

    order = np.argsort(-scores, kind="mergesort")
    scores, labels = scores[order], labels[order]

    tps = np.cumsum(labels)
    fps = np.cumsum(1 - labels)

    n_pos = max(int(labels.sum()), 1)
    n_neg = max(int((1 - labels).sum()), 1)

    # Keep only the last index of each run of equal scores.
    distinct = np.where(np.diff(scores))[0]
    idx = np.r_[distinct, scores.size - 1]

    tpr = np.r_[0.0, tps[idx] / n_pos]
    fpr = np.r_[0.0, fps[idx] / n_neg]
    thr = np.r_[np.inf, scores[idx]]
    return fpr, tpr, thr


def eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Equal Error Rate and the threshold that achieves it.

    Args:
        scores: Spoof score. Higher must mean "more likely spoof".
        labels: 1 for spoof, 0 for bonafide.

    Returns:
        ``(eer, threshold)`` with the EER as a fraction in [0, 1].
    """
    fpr, tpr, thr = _roc(scores, labels)
    fnr = 1.0 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fnr[i] + fpr[i]) / 2.0), float(thr[i])


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve."""
    fpr, tpr, _ = _roc(scores, labels)
    return float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))


def fpr_at_tpr(scores: np.ndarray, labels: np.ndarray, target_tpr: float) -> tuple[float, float]:
    """False-positive rate at a fixed true-positive rate.

    This is the number a deployment needs and the number papers almost never
    report.  ``target_tpr = 0.90`` answers "if we want to catch 90 percent of
    clones, what fraction of genuine callers do we wrongly flag?".

    Args:
        scores: Spoof score, higher means more likely spoof.
        labels: 1 for spoof, 0 for bonafide.
        target_tpr: Desired detection rate in (0, 1].

    Returns:
        ``(fpr, threshold)``.
    """
    if not 0.0 < target_tpr <= 1.0:
        raise ValueError("target_tpr must be in (0, 1]")
    fpr, tpr, thr = _roc(scores, labels)
    i = int(np.searchsorted(tpr, target_tpr, side="left"))
    i = min(i, len(tpr) - 1)
    return float(fpr[i]), float(thr[i])


def alert_precision(fpr: float, tpr: float, base_rate: float) -> float:
    """Fraction of raised alerts that are genuine attacks.

    At Pindrop's reported banking base rate of 0.17 percent, a detector at
    1 percent FPR and 90 percent TPR produces alerts that are roughly 87
    percent innocent customers.  Nobody publishes this arithmetic; it decides
    whether the system is deployable.

    Args:
        fpr: False-positive rate in [0, 1].
        tpr: True-positive rate in [0, 1].
        base_rate: Prior probability that a call is an attack.

    Returns:
        Precision in [0, 1].
    """
    tp = tpr * base_rate
    fp = fpr * (1.0 - base_rate)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def det_curve(scores: np.ndarray, labels: np.ndarray) -> dict[str, list[float]]:
    """Points for a DET plot (false-positive against false-negative rate)."""
    fpr, tpr, thr = _roc(scores, labels)
    return {"fpr": fpr.tolist(), "fnr": (1.0 - tpr).tolist(), "thresholds": thr.tolist()}


def macro_f1(pred: np.ndarray, true: np.ndarray, num_classes: int) -> float:
    """Macro-averaged F1, used for Tier 2 family attribution."""
    pred = np.asarray(pred).ravel().astype(int)
    true = np.asarray(true).ravel().astype(int)
    scores = []
    for c in range(num_classes):
        tp = float(np.sum((pred == c) & (true == c)))
        fp = float(np.sum((pred == c) & (true != c)))
        fn = float(np.sum((pred != c) & (true == c)))
        denom = 2 * tp + fp + fn
        scores.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(scores))


def summarise(
    spoof_scores: np.ndarray,
    binary_labels: np.ndarray,
    base_rate: float = 0.0017,
) -> dict[str, float]:
    """One call that produces every headline number for a report table."""
    e, thr = eer(spoof_scores, binary_labels)
    out: dict[str, float] = {"eer": e, "eer_threshold": thr, "auc": auc(spoof_scores, binary_labels)}
    for t in (0.90, 0.95, 0.99):
        f, th = fpr_at_tpr(spoof_scores, binary_labels, t)
        out[f"fpr_at_tpr{int(t * 100)}"] = f
        out[f"threshold_at_tpr{int(t * 100)}"] = th
        out[f"alert_precision_at_tpr{int(t * 100)}"] = alert_precision(f, t, base_rate)
    return out
