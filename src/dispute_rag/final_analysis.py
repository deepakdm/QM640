"""Final-report analyses that sit on top of the interim confirmatory layer (analysis.py).

All functions take the long per-case table (analysis.load_per_case) of ONE split, so the same code
produces the development (pilot) and the locked test results, which are reported side by side.

Added for the final report, each fixed before the test set is run:

  classification_by_condition   confusion matrix, per-class precision/recall/F1, macro-F1
  majority_baseline             accuracy of always predicting the most common gold label
  needs_human_collapse          secondary analysis: escalate and request_more_info merged into
                                "needs_human" (both route the case to an investigator), so the
                                accuracy question becomes "does it decide when it should, and hand
                                over when it should?"
  without_ambiguous             accuracy restricted to complaints the rater did not flag ambiguous
  clause_sensitivity            correct-clause rate under the strict (primary), parent and section
                                matching rules (refs.CLAUSE_MATCH_MODES)
  condition_summary             one row per condition: every pillar, with Wilson intervals
  best_model                    the pre-specified ranking rule (BEST_MODEL_RULE)
  dev_test_table                development vs test, side by side, with the difference
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import CONTROL, compare_to_control
from .classification import DETERMINATION_CLASSES, confusion_matrix, per_class_metrics
from .refs import CLAUSE_MATCH_MODES, cites_gold_mode
from .stats import mcnemar_exact, wilson_ci

ORDER = ("C0", "A", "B", "C")
NEEDS_HUMAN = {"escalate", "request_more_info"}
SHORT = {"approve_dispute": "Approve", "deny_dispute": "Deny", "partial_credit": "Partial",
         "escalate": "Escalate", "request_more_info": "More info", "no_answer": "No answer",
         "needs_human": "Needs human"}

BEST_MODEL_RULE = (
    "Conditions are ranked on the split being reported by (1) determination accuracy, highest first; "
    "ties broken by (2) macro-F1, highest first; (3) fabricated-citation rate, lowest first; "
    "(4) 95th-percentile response time, lowest first. The top-ranked condition is reported as the "
    "best model on that split. Its lead over the runner-up is tested with an exact McNemar test on "
    "accuracy (alpha = .05, two-sided); if that test is not significant the report states that the "
    "two are statistically indistinguishable. The rule was fixed on the development set; the test "
    "set confirms or overturns it. The no-retrieval baseline C0 is eligible."
)


def _conds(long: pd.DataFrame) -> list[str]:
    present = set(long["condition"])
    return [c for c in ORDER if c in present]


def _labels(y_true, y_pred) -> list[str]:
    seen = set(y_true) | {p for p in y_pred if p}
    return [c for c in DETERMINATION_CLASSES if c in seen]


# ------------------------------------------------------------------------------ classification
def classification_by_condition(long: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for c in _conds(long):
        d = long[long.condition == c]
        y_true = d["gold_determination"].tolist()
        y_pred = [p if isinstance(p, str) else None for p in d["pred_determination"]]
        labels = _labels(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels)
        if cm["no_answer"].sum() == 0:
            cm = cm.drop(columns="no_answer")
        pc = per_class_metrics(y_true, y_pred, labels)
        supported = pc.drop(index=["macro_avg", "weighted_avg"])
        supported = supported[supported.support >= 5]
        out[c] = {"n": len(d), "accuracy": float(np.mean([t == p for t, p in zip(y_true, y_pred)])),
                  "confusion_matrix": cm, "per_class": pc,
                  "macro_f1": float(pc.loc["macro_avg", "f1"]),
                  "macro_f1_support5": float(supported["f1"].mean()) if len(supported) else np.nan,
                  "weighted_f1": float(pc.loc["weighted_avg", "f1"])}
    return out


def majority_baseline(long: pd.DataFrame) -> dict:
    d = long[long.condition == _conds(long)[0]]
    vc = d["gold_determination"].value_counts()
    lo, hi = wilson_ci(int(vc.iloc[0]), int(vc.sum()))
    return {"label": vc.index[0], "count": int(vc.iloc[0]), "n": int(vc.sum()),
            "accuracy": float(vc.iloc[0] / vc.sum()), "wilson_ci": [lo, hi],
            "gold_distribution": vc.to_dict()}


# ------------------------------------------------------------------------------ secondary analyses
def needs_human_collapse(long: pd.DataFrame) -> dict:
    """Accuracy with escalate and request_more_info merged into needs_human; paired tests vs C0."""
    m = long.copy()
    col = lambda s: s.map(lambda v: "needs_human" if v in NEEDS_HUMAN else v)  # noqa: E731
    m["determination_correct"] = (col(m["gold_determination"]) == col(m["pred_determination"])).astype(int)
    res = compare_to_control(m, "determination_correct")
    res["metric"] = "determination_correct_needs_human"
    res["classification"] = {}
    for c in _conds(m):
        d = m[m.condition == c]
        yt, yp = col(d["gold_determination"]).tolist(), col(d["pred_determination"]).tolist()
        labels = [x for x in ("approve_dispute", "deny_dispute", "partial_credit", "needs_human")
                  if x in set(yt) | set(yp)]
        cm = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
        for t, p in zip(yt, yp):
            if t in cm.index and p in cm.columns:
                cm.loc[t, p] += 1
        res["classification"][c] = cm
    return res


def without_ambiguous(long: pd.DataFrame, gold: dict) -> dict:
    keep = long["case_id"].map(lambda i: i in gold and not gold[i].ambiguous)
    res = compare_to_control(long[keep], "determination_correct")
    res["metric"] = "determination_correct_unambiguous"
    return res


def clause_sensitivity(runs_dir: Path, long: pd.DataFrame, gold: dict) -> dict:
    """Correct-clause rate (first run) under each matching rule, with paired tests vs C0."""
    rows = []
    for c in _conds(long):
        cfg = long.loc[long.condition == c, "config"].iloc[0]
        ids = set(long.loc[long.condition == c, "case_id"])
        for line in (Path(runs_dir) / cfg / "answers.jsonl").read_text().splitlines():
            if not line.strip():
                continue
            a = json.loads(line)
            if a["run_idx"] != 0 or a["case_id"] not in ids:
                continue
            g = gold[a["case_id"]].governing_clauses
            rows.append({"case_id": a["case_id"], "condition": c,
                         **{f"clause_{m}": int(cites_gold_mode(a.get("cited_section_refs") or [], g, m))
                            for m in CLAUSE_MATCH_MODES}})
    wide = pd.DataFrame(rows)
    return {m: compare_to_control(wide, f"clause_{m}") for m in CLAUSE_MATCH_MODES}


# ------------------------------------------------------------------------------ summary + best model
def condition_summary(long: pd.DataFrame, cls: dict) -> pd.DataFrame:
    rows = []
    for c in _conds(long):
        d = long[long.condition == c]
        n = len(d)
        k = int(d["determination_correct"].sum())
        lo, hi = wilson_ci(k, n)
        fab = int(d["fabricated_citation"].sum())
        flo, fhi = wilson_ci(fab, n)
        rows.append({
            "condition": c, "n": n, "accuracy": k / n, "accuracy_lo": lo, "accuracy_hi": hi,
            "macro_f1": cls[c]["macro_f1"], "weighted_f1": cls[c]["weighted_f1"],
            "fabricated_rate": fab / n, "fabricated_lo": flo, "fabricated_hi": fhi,
            "correct_clause_rate": d["correct_clause_cited"].mean(),
            "clause_in_context_rate": d["clause_in_context"].mean() if c != CONTROL else np.nan,
            "consistent_rate": d["consistent"].mean(),
            "review_required_rate": d["review_required"].mean(),
            "latency_median_s": d["latency_s"].median(),
            "latency_p95_s": d["latency_s"].quantile(0.95),
            "escalate_share": (d["pred_determination"] == "escalate").mean(),
        })
    return pd.DataFrame(rows).set_index("condition")


def best_model(summary: pd.DataFrame, long: pd.DataFrame) -> dict:
    ranked = summary.sort_values(
        ["accuracy", "macro_f1", "fabricated_rate", "latency_p95_s"],
        ascending=[False, False, True, True])
    best, runner = ranked.index[0], ranked.index[1] if len(ranked) > 1 else None
    out = {"rule": BEST_MODEL_RULE, "ranking": ranked.index.tolist(), "best": best,
           "best_accuracy": float(ranked.loc[best, "accuracy"]),
           "best_macro_f1": float(ranked.loc[best, "macro_f1"])}
    if runner is not None:
        w = long.pivot_table(index="case_id", columns="condition", values="determination_correct",
                             aggfunc="first").dropna()
        t = mcnemar_exact(w[best].astype(int), w[runner].astype(int))
        out.update({"runner_up": runner, "runner_up_accuracy": float(ranked.loc[runner, "accuracy"]),
                    "lead_points": float(ranked.loc[best, "accuracy"] - ranked.loc[runner, "accuracy"]),
                    "mcnemar_vs_runner_up": t,
                    "distinguishable": bool(t["p_value"] < .05)})
        out["statement"] = (
            f"{best} ranks first under the pre-specified rule (accuracy {out['best_accuracy']:.1%}, "
            f"macro-F1 {out['best_macro_f1']:.3f}); its lead over {runner} is "
            f"{100 * out['lead_points']:.1f} points (exact McNemar p = {t['p_value']:.3f}), so the two are "
            + ("statistically distinguishable." if out["distinguishable"] else "statistically indistinguishable."))
    return out


BEST_BY_METRIC = [("accuracy", True, "Determination accuracy"), ("macro_f1", True, "Macro-F1"),
                  ("fabricated_rate", False, "Fabricated-citation rate"),
                  ("correct_clause_rate", True, "Correct clause cited"),
                  ("consistent_rate", True, "Consistency across runs"),
                  ("latency_p95_s", False, "95th-percentile response time (s)")]


def best_by_metric(summary: pd.DataFrame) -> pd.DataFrame:
    """Best condition on each metric (ties listed together), with the runner-up value."""
    rows = []
    for col, higher, label in BEST_BY_METRIC:
        v = summary[col].dropna()
        top = v.max() if higher else v.min()
        best = [c for c in v.index if np.isclose(v[c], top)]
        rest = v.drop(best)
        runner = (rest.max() if higher else rest.min()) if len(rest) else np.nan
        rows.append({"metric": label, "column": col, "direction": "higher" if higher else "lower",
                     "best": " = ".join(best), "best_value": float(top), "runner_up_value": float(runner)})
    return pd.DataFrame(rows)


def dev_test_table(dev: pd.DataFrame, test: pd.DataFrame,
                   metrics=("accuracy", "macro_f1", "fabricated_rate", "correct_clause_rate",
                            "consistent_rate", "latency_p95_s")) -> pd.DataFrame:
    """Separated evaluation results: development (pilot) vs locked test, per condition."""
    rows = []
    for c in [x for x in ORDER if x in dev.index or x in test.index]:
        r = {"condition": c, "n_dev": dev.loc[c, "n"] if c in dev.index else np.nan,
             "n_test": test.loc[c, "n"] if c in test.index else np.nan}
        for m in metrics:
            dv = dev.loc[c, m] if c in dev.index else np.nan
            tv = test.loc[c, m] if c in test.index else np.nan
            r[f"{m}_dev"], r[f"{m}_test"], r[f"{m}_diff"] = dv, tv, tv - dv
        rows.append(r)
    return pd.DataFrame(rows).set_index("condition")


def to_jsonable(obj):
    if isinstance(obj, pd.DataFrame):
        return obj.reset_index().to_dict(orient="records")
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
