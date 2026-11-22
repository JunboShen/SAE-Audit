"""Scoring: the majority answer of the runs, balanced accuracy, and a paired bootstrap interval over protocols."""
import numpy as np


def majority(answers, fallback=None):
    """Majority of the parsed answers (0/1) of several runs; `fallback` if no run gave an answer."""
    answers = [a for a in answers if a is not None]
    return int(np.mean(answers) > 0.5) if answers else fallback


def balanced_accuracy(pred, y):
    pred, y = np.asarray(pred, int), np.asarray(y, int)
    sens = pred[y == 1].mean() if (y == 1).any() else 0.0
    spec = (1 - pred[y == 0]).mean() if (y == 0).any() else 0.0
    return float(0.5 * (sens + spec))


def paired_bootstrap(y, pred_a, pred_b, draws=3000, seed=0):
    """Difference in balanced accuracy, b minus a, with a 95% interval from resampling protocols."""
    y, pred_a, pred_b = np.asarray(y, int), np.asarray(pred_a, int), np.asarray(pred_b, int)
    rs, d = np.random.RandomState(seed), []
    for _ in range(draws):
        i = rs.randint(0, len(y), len(y))
        if len(set(y[i].tolist())) < 2:
            continue
        d.append(balanced_accuracy(pred_b[i], y[i]) - balanced_accuracy(pred_a[i], y[i]))
    point = balanced_accuracy(pred_b, y) - balanced_accuracy(pred_a, y)
    return point, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

