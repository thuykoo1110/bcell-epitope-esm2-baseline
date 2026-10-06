import numpy as np

from epitope import metrics


def test_perfect_scores():
    y = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 1])
    s = y.astype(float)
    assert metrics.pr_auc(y, s) == 1.0
    assert metrics.roc_auc(y, s) == 1.0
    thr = metrics.best_f1_threshold(y, s)
    m = metrics.threshold_metrics(y, s, thr)
    assert m["f1"] == 1.0 and m["mcc"] == 1.0


def test_random_scorer_pr_auc_is_close_to_prevalence():
    rng = np.random.default_rng(0)
    y = (rng.random(200_000) < 0.10).astype(int)
    assert abs(metrics.random_scorer_pr_auc(y, seed=1) - metrics.prevalence(y)) < 0.01


def test_all_negative_baseline_has_high_accuracy_but_zero_mcc():
    y = np.array([1] + [0] * 9)
    b = metrics.all_negative_baseline(y)
    assert b["accuracy"] == 0.9
    assert b["mcc"] == 0.0 and b["f1"] == 0.0


def test_mcc_known_value():
    # tp=6, tn=3, fp=1, fn=2 -> (18 - 2) / sqrt(7 * 8 * 4 * 5)
    expected = 16 / np.sqrt(7 * 8 * 4 * 5)
    assert abs(metrics.mcc_from_confusion(6, 3, 1, 2) - expected) < 1e-12


def test_threshold_metrics_counts():
    y = np.array([1, 1, 0, 0, 0])
    s = np.array([0.9, 0.4, 0.8, 0.2, 0.1])
    m = metrics.threshold_metrics(y, s, 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert abs(m["precision"] - 0.5) < 1e-12 and abs(m["recall"] - 0.5) < 1e-12
