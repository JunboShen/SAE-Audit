"""The concept probe: a sparse logistic regression over binary concept indicators (Eq. 2 of the paper).

z_j(x) = 1 if concept j is present in protocol x. The probe is fitted once per task on the training trials.
Two rules are used. Completion, duration, adverse events and dropout have 6,000 training trials and use an
L1 penalty directly. Sample size has 1,129 and uses the 40 concepts with the largest L1 weights, refitted with an
L2 penalty.
"""
import warnings

import numpy as np


def _logistic(X, y, **kw):
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():                    # newer scikit-learn versions deprecate the `penalty` argument
        warnings.simplefilter("ignore")
        return LogisticRegression(**kw).fit(X, y)


def _sigmoid(s):
    return 1.0 / (1.0 + np.exp(-s))


def fit_l1(X, y, C=0.05):
    """L1-regularised logistic regression; every concept with a nonzero weight is kept.
    Returns (kept columns, weights, intercept, threshold); the threshold is the median training probability."""
    lr = _logistic(X, y, penalty="l1", C=C, solver="liblinear", max_iter=3000, random_state=0)
    w, b = lr.coef_[0], float(lr.intercept_[0])
    keep = np.where(np.abs(w) > 1e-8)[0]
    threshold = float(np.median(_sigmoid(X @ w + b)))
    return keep, w[keep].astype(float), b, threshold


def fit_top_k(X, y, k=40):
    """Relax the L1 penalty until at least k concepts have a nonzero weight, keep the k largest, refit with L2.
    The threshold is the out-of-fold probability quantile at one minus the training positive rate."""
    for C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        l1 = _logistic(X, y, C=C, penalty="l1", solver="liblinear", max_iter=2000, random_state=0)
        nonzero = np.where(np.abs(l1.coef_[0]) > 1e-8)[0]
        if len(nonzero) >= k:
            break
    keep = nonzero[np.argsort(-np.abs(l1.coef_[0][nonzero]))][:k]
    l2 = _logistic(X[:, keep], y, C=1.0, max_iter=3000)
    scores = np.zeros(len(y))
    folds = np.array_split(np.random.RandomState(0).permutation(len(y)), 5)
    for f in range(5):
        rest = np.concatenate([folds[g] for g in range(5) if g != f])
        fold = _logistic(X[rest][:, keep], y[rest], C=1.0, max_iter=3000)
        scores[folds[f]] = fold.predict_proba(X[folds[f]][:, keep])[:, 1]
    threshold = float(np.quantile(scores, 1 - np.mean(y)))
    return keep, l2.coef_[0].astype(float), float(l2.intercept_[0]), threshold


def fit(task, train_features, train_y, vocabulary, min_mcc=0.7):
    """Fit the probe of a task. Candidates are the verified concepts; see `fit_l1` and `fit_top_k`."""
    from .evidence import MAX_SUPPORT, MIN_SUPPORT, presence
    y = np.asarray(train_y, int)
    verified = sorted(f for f, e in vocabulary.items() if float(e.get("mcc", 0)) >= min_mcc)
    if task.name == "sample_size":
        support = presence(train_features, verified).sum(0)
        pool = [f for f, c in zip(verified, support) if MIN_SUPPORT * len(y) <= c <= MAX_SUPPORT * len(y)]
        keep, w, b, thr = fit_top_k(presence(train_features, pool), y)
    else:
        pool = [f for f in verified if vocabulary[f].get("clinical")]      # concepts that describe clinical content
        keep, w, b, thr = fit_l1(presence(train_features, pool), y)
    concepts = [int(pool[j]) for j in keep]
    return {"task": task.name, "concepts": concepts, "names": [vocabulary[f]["name"] for f in concepts],
            "weights": [float(x) for x in w], "intercept": b, "threshold": thr}


def predict(probe, features):
    """Probability of the positive outcome and the predicted label for each protocol."""
    from .evidence import presence
    z = presence(features, probe["concepts"])
    p = _sigmoid(z @ np.asarray(probe["weights"], float) + float(probe["intercept"]))
    return p, (p >= float(probe["threshold"])).astype(int)
