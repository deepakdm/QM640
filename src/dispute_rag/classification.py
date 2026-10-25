"""Classification-style evaluation of the determination output.

The research-design guide (Step 4.1 / 4.2) specifies, for classification tasks: a confusion
matrix, cross-validation, accuracy / precision / recall / F1, and significance testing. The
determination output is a 5-class classification problem (approve / deny / partial / escalate /
request more info), so those tools apply directly and are implemented here.

Two deliberate departures, both of which belong in the Methods section rather than being
silently skipped:

  * **k-fold cross-validation.** Classic k-fold trains on k-1 folds and tests on the held-out
    fold. Nothing here is trained — the retrieval index is built from the regulatory corpus, not
    from the labeled cases, and the LLM's weights are fixed — so there is no fitting step to
    cross-validate and no risk of the overfitting that k-fold exists to detect. The underlying
    intent (do not over-read one point estimate) is served instead by `kfold_stability`, which
    partitions the gold set into k folds and reports the spread of the metric across them, and
    by the bootstrap confidence intervals in stats.py.

  * **ROC-AUC.** ROC-AUC needs a continuous score per class. The pipeline returns a categorical
    determination with no calibrated probability, so an AUC would be undefined. `roc_auc_escalation`
    is provided for the case where a confidence score is added later; until then, report the
    confusion matrix and macro-F1 and state this limitation explicitly.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

DETERMINATION_CLASSES = [
    "approve_dispute", "deny_dispute", "partial_credit", "escalate", "request_more_info",
]


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[Optional[str]],
                     labels: Sequence[str] | None = None) -> pd.DataFrame:
    """Rows = gold determination, columns = system determination.

    Unparseable answers are counted in a `no_answer` column rather than dropped, so the
    matrix totals always equal the number of cases.
    """
    labels = list(labels or DETERMINATION_CLASSES)
    cols = labels + ["no_answer"]
    m = pd.DataFrame(0, index=labels, columns=cols, dtype=int)
    for t, p in zip(y_true, y_pred):
        if t not in m.index:
            continue
        m.loc[t, p if p in labels else "no_answer"] += 1
    m.index.name = "gold"
    m.columns.name = "predicted"
    return m


def per_class_metrics(y_true: Sequence[str], y_pred: Sequence[Optional[str]],
                      labels: Sequence[str] | None = None) -> pd.DataFrame:
    """Precision, recall, F1 and support per class, plus macro and weighted averages.

    Macro-F1 is the headline number to report: the determination classes are heavily
    imbalanced in any real complaint sample, and plain accuracy hides failure on the rare
    but operationally important classes (escalate, partial_credit).
    """
    labels = list(labels or DETERMINATION_CLASSES)
    rows = []
    for c in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        support = sum(1 for t in y_true if t == c)
        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall and not np.isnan(precision) and not np.isnan(recall)
              else 0.0 if support else np.nan)
        rows.append({"class": c, "precision": precision, "recall": recall,
                     "f1": f1, "support": support})
    df = pd.DataFrame(rows).set_index("class")

    present = df[df.support > 0]
    total = present.support.sum()
    macro = present[["precision", "recall", "f1"]].mean()
    weighted = (present[["precision", "recall", "f1"]]
                .mul(present.support, axis=0).sum() / total) if total else macro * np.nan
    df.loc["macro_avg"] = [macro.precision, macro.recall, macro.f1, total]
    df.loc["weighted_avg"] = [weighted.precision, weighted.recall, weighted.f1, total]
    return df


def accuracy(y_true: Sequence[str], y_pred: Sequence[Optional[str]]) -> float:
    if not len(y_true):
        return float("nan")
    return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)


def kfold_stability(correct: Sequence[int], k: int = 5, seed: int = 42) -> dict:
    """Spread of a metric across k equal partitions of the evaluation set.

    This is an evaluation-set stability check, NOT model cross-validation — see the module
    docstring. Report it as: "accuracy = 0.78, stable across 5 folds (SD = 0.04, range
    0.72-0.83)". A wide spread means the point estimate is being driven by a handful of cases
    and should be reported with that caveat.
    """
    x = np.asarray(correct, dtype=float)
    n = len(x)
    if n < k:
        return {"k": k, "error": f"only {n} cases; need at least k={k}"}
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    scores = [float(x[f].mean()) for f in folds]
    return {
        "k": k, "n": int(n), "fold_scores": scores,
        "mean": float(np.mean(scores)), "sd": float(np.std(scores, ddof=1)),
        "min": float(np.min(scores)), "max": float(np.max(scores)),
        "fold_sizes": [int(len(f)) for f in folds],
    }


def roc_auc_escalation(y_true: Sequence[str], scores: Sequence[float],
                       positive_class: str = "escalate") -> Optional[float]:
    """One-vs-rest ROC-AUC, usable only once the pipeline emits a confidence score.

    Returns None when no scores are available, which is the current state. Do not fabricate
    scores from the categorical output to fill this in — an AUC computed from hard labels is
    just a rescaled accuracy and is misleading.
    """
    if scores is None or not len(scores):
        return None
    from sklearn.metrics import roc_auc_score
    y = [1 if t == positive_class else 0 for t in y_true]
    if len(set(y)) < 2:
        return None
    return float(roc_auc_score(y, list(scores)))


def classification_report_for_run(answers, gold, labels: Sequence[str] | None = None) -> dict:
    """Full classification view of one configuration's first-run answers."""
    first = {a.case_id: a for a in answers if a.run_idx == 0}
    ids = [cid for cid in first if cid in gold]
    y_true = [gold[cid].determination for cid in ids]
    y_pred = [first[cid].determination for cid in ids]
    correct = [int(t == p) for t, p in zip(y_true, y_pred)]
    return {
        "n": len(ids),
        "accuracy": accuracy(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels),
        "per_class": per_class_metrics(y_true, y_pred, labels),
        "kfold_stability": kfold_stability(correct),
    }
