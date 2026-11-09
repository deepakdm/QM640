"""Step 7 (final report): outcome-linked EDA — how case characteristics relate to model outcomes.

    python scripts/07_outcome_eda.py --split dev
    python scripts/07_outcome_eda.py --split test
    python scripts/07_outcome_eda.py --split both

Three views, per condition, on the first of three runs (descriptive; the confirmatory driver model
is the GEE in 04_compare_and_business.py):
  1. Accuracy by narrative-length tercile and by rater-flagged ambiguity, with Wilson intervals and
     a chi-square test of independence per condition.
  2. How often the governing clause reached the model (clause_in_context, retrieval conditions
     only), by dispute type.
  3. Over-escalation: share of complaints answered "escalate", by gold determination.

Writes outputs/final/<split>/outcome_eda.json, outcome_*.csv and fig_outcome_*.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2_contingency  # noqa: E402

from dispute_rag.analysis import load_per_case  # noqa: E402
from dispute_rag.cfpb_cases import load_cases  # noqa: E402
from dispute_rag.final_analysis import ORDER, SHORT  # noqa: E402
from dispute_rag.gold import load_gold  # noqa: E402
from dispute_rag.splits import paths  # noqa: E402
from dispute_rag.stats import wilson_ci  # noqa: E402
from dispute_rag.viz import SERIES, TEXT, apply_style, finish  # noqa: E402

TYPE_LABEL = {"unauthorized_transaction": "Unauthorized", "incorrect_amount": "Incorrect amount",
              "error_resolution": "Error resolution"}


def rate_table(df: pd.DataFrame, group: str, outcome: str) -> pd.DataFrame:
    rows = []
    for (c, g), d in df.groupby(["condition", group], observed=True):
        k, n = int(d[outcome].sum()), len(d)
        lo, hi = wilson_ci(k, n)
        rows.append({"condition": c, group: g, "n": n, "k": k, "rate": k / n, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def chi2_by_condition(df: pd.DataFrame, group: str, outcome: str) -> dict:
    out = {}
    for c, d in df.groupby("condition"):
        tab = pd.crosstab(d[group], d[outcome])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            out[c] = None
            continue
        chi2, p, dof, exp = chi2_contingency(tab, correction=False)
        out[c] = {"chi2": float(chi2), "df": int(dof), "p_value": float(p),
                  "min_expected": float(exp.min())}
    return out


def grouped_bars(tab: pd.DataFrame, group: str, levels: list, labels: list, title: str, ylabel: str,
                 out: Path, conds: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    x = np.arange(len(conds))
    w = 0.8 / len(levels)
    for k, (lev, lab) in enumerate(zip(levels, labels)):
        r = tab[tab[group] == lev].set_index("condition").reindex(conds)
        y = r["rate"] * 100
        pos = x + (k - (len(levels) - 1) / 2) * w
        ax.bar(pos, y, w, color=SERIES[k], label=f"{lab}")
        ax.errorbar(pos, y, yerr=[y - r["lo"] * 100, r["hi"] * 100 - y], fmt="none", ecolor=TEXT, capsize=2, lw=0.8)
    ax.set_xticks(x, conds)
    ax.set_ylim(0, 100)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=len(levels), loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(axis="x", visible=False)
    finish(fig, out)


def run(split: str) -> None:
    sp = paths(split)
    long = load_per_case(sp.runs)
    if long.empty:
        print(f"[{split}] no runs — skipped")
        return
    gold = load_gold(sp.gold)
    cases = {c.case_id: c for c in load_cases(sp.cases)}
    long = long[long["case_id"].isin(gold)].copy()
    long["words"] = long["case_id"].map(lambda i: len(cases[i].narrative.split()))
    long["dispute_type"] = long["case_id"].map(lambda i: cases[i].dispute_type)
    long["ambiguous"] = long["case_id"].map(lambda i: "Ambiguous" if gold[i].ambiguous else "Clear")
    long["escalated"] = (long["pred_determination"] == "escalate").astype(int)
    words = long.drop_duplicates("case_id")["words"]
    cuts = np.quantile(words, [1 / 3, 2 / 3])
    long["length_tercile"] = pd.cut(long["words"], [-np.inf, cuts[0], cuts[1], np.inf],
                                    labels=["Short", "Medium", "Long"])
    conds = [c for c in ORDER if c in set(long["condition"])]
    out = sp.final
    out.mkdir(parents=True, exist_ok=True)
    sfx = " (development set)" if split == "dev" else " (test set)"

    # 1. accuracy by length tercile and ambiguity
    by_len = rate_table(long, "length_tercile", "determination_correct")
    by_amb = rate_table(long, "ambiguous", "determination_correct")
    by_len.to_csv(out / "outcome_accuracy_by_length.csv", index=False)
    by_amb.to_csv(out / "outcome_accuracy_by_ambiguity.csv", index=False)
    lens = [f"Short (≤{cuts[0]:.0f} words)", f"Medium ({cuts[0]:.0f}–{cuts[1]:.0f})", f"Long (>{cuts[1]:.0f})"]
    grouped_bars(by_len, "length_tercile", ["Short", "Medium", "Long"], lens,
                 f"Accuracy by narrative length{sfx}", "Correct determinations (%)",
                 out / "fig_outcome_accuracy_by_length.png", conds)
    grouped_bars(by_amb, "ambiguous", ["Clear", "Ambiguous"], ["Not flagged", "Flagged ambiguous"],
                 f"Accuracy by rater-flagged ambiguity{sfx}", "Correct determinations (%)",
                 out / "fig_outcome_accuracy_by_ambiguity.png", conds)

    # 2. governing clause reached the model, by dispute type (retrieval conditions)
    ret = long[long["condition"] != "C0"]
    by_type = rate_table(ret, "dispute_type", "clause_in_context")
    by_type.to_csv(out / "outcome_clause_in_context_by_type.csv", index=False)
    types = [t for t in TYPE_LABEL if t in set(ret["dispute_type"])]
    grouped_bars(by_type, "dispute_type", types, [TYPE_LABEL[t] for t in types],
                 f"Governing clause in retrieved context, by dispute type{sfx}",
                 "Clause in context (%)", out / "fig_outcome_clause_by_type.png", [c for c in conds if c != "C0"])

    # 3. over-escalation by gold label
    by_gold = rate_table(long, "gold_determination", "escalated")
    by_gold.to_csv(out / "outcome_escalation_by_gold.csv", index=False)
    golds = [g for g in ("approve_dispute", "deny_dispute", "request_more_info", "escalate", "partial_credit")
             if (long["gold_determination"] == g).sum() >= len(conds) * 3]
    grouped_bars(by_gold, "gold_determination", golds, [f"Gold: {SHORT[g]}" for g in golds],
                 f"Share answered 'escalate', by gold determination{sfx}", "Answered escalate (%)",
                 out / "fig_outcome_escalation_by_gold.png", conds)

    summary = {
        "split": split, "n_complaints": int(long["case_id"].nunique()),
        "length_tercile_cutpoints_words": [float(c) for c in cuts],
        "chi2_accuracy_by_length": chi2_by_condition(long, "length_tercile", "determination_correct"),
        "chi2_accuracy_by_ambiguity": chi2_by_condition(long, "ambiguous", "determination_correct"),
        "chi2_clause_in_context_by_type": chi2_by_condition(ret, "dispute_type", "clause_in_context"),
        "escalate_share_overall": long.groupby("condition")["escalated"].mean().to_dict(),
        "gold_escalate_share": float((long.drop_duplicates("case_id")["gold_determination"] == "escalate").mean()),
    }
    (out / "outcome_eda.json").write_text(json.dumps(summary, indent=2))

    print(f"\n=== {split}: {summary['n_complaints']} complaints; length terciles at "
          f"{cuts[0]:.0f} and {cuts[1]:.0f} words")
    show = lambda t, g: t.pivot(index="condition", columns=g, values="rate").reindex(conds).round(3)  # noqa: E731
    print("Accuracy by length tercile:\n", show(by_len, "length_tercile").to_string())
    print("Accuracy by ambiguity:\n", show(by_amb, "ambiguous").to_string())
    print("Clause in context by dispute type:\n",
          by_type.pivot(index="condition", columns="dispute_type", values="rate").round(3).to_string())
    print("Answered escalate, by gold label:\n", show(by_gold, "gold_determination").to_string())
    for k in ("chi2_accuracy_by_length", "chi2_accuracy_by_ambiguity", "chi2_clause_in_context_by_type"):
        print(k, {c: (round(v["p_value"], 3) if v else None) for c, v in summary[k].items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "both"], default="both")
    args = ap.parse_args()
    apply_style()
    for s in (["dev", "test"] if args.split == "both" else [args.split]):
        run(s)


if __name__ == "__main__":
    main()
