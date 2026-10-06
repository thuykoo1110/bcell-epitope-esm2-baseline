from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def prevalence(y) -> float:
    return float(np.mean(y))


def pr_auc(y, s) -> float:
    return float(average_precision_score(y, s))


def roc_auc(y, s) -> float:
    return float(roc_auc_score(y, s))


def confusion(y, pred) -> tuple[int, int, int, int]:
    y = y.astype(bool)
    pred = pred.astype(bool)
    tp = int(np.sum(y & pred))
    tn = int(np.sum(~y & ~pred))
    fp = int(np.sum(~y & pred))
    fn = int(np.sum(y & ~pred))
    return tp, tn, fp, fn


def mcc_from_confusion(tp: int, tn: int, fp: int, fn: int) -> float:
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return 0.0
    return float((tp * tn - fp * fn) / denom)


def threshold_metrics(y, s, thr: float) -> dict:
    tp, tn, fp, fn = confusion(y, s >= thr)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "threshold": float(thr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mcc": mcc_from_confusion(tp, tn, fp, fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def best_f1_threshold(y, s) -> float:
    precision, recall, thresholds = precision_recall_curve(y, s)

    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    if len(f1) == 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1))])


def random_scorer_pr_auc(y, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    return pr_auc(y, rng.random(len(y)))


def all_negative_baseline(y) -> dict:
    tp, tn, fp, fn = confusion(y, np.zeros_like(y))
    return {
        "accuracy": float((tp + tn) / len(y)),
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "mcc": mcc_from_confusion(tp, tn, fp, fn),
    }


def full_report(y, s, thr: float) -> dict:
    out = {
        "pr_auc": pr_auc(y, s),
        "roc_auc": roc_auc(y, s),
        "prevalence": prevalence(y),
        "n_residues": int(len(y)),
        "n_positive": int(np.sum(y)),
    }
    out.update(threshold_metrics(y, s, thr))
    return out
