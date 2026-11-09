"""Step 5 (final report): classification view, secondary analyses, best model and dev-vs-test.

Run after 04_compare_and_business.py for each split that has results:

    python scripts/05_final_analysis.py --split dev     # interim pilot = development set
    python scripts/05_final_analysis.py --split test    # locked test set
    python scripts/05_final_analysis.py --split both    # both, plus the dev-vs-test comparison

Writes outputs/final/<split>/:
    condition_summary.csv            every pillar per condition, Wilson intervals, macro-F1
    confusion_<C>.csv, per_class_<C>.csv
    majority_baseline.json
    best_model.json                  pre-specified ranking rule, McNemar vs runner-up, statement
    secondary_needs_human.json       escalate + request_more_info merged (paired tests vs C0)
    secondary_unambiguous.json       accuracy without rater-flagged ambiguous complaints
    sensitivity_clause_match.json    correct-clause rate under strict / parent / section rules
    sensitivity_clause_match.csv
    fig_confusion_matrices.png, fig_accuracy_macro_f1.png, fig_clause_sensitivity.png
and, with --split both, outputs/final/dev_vs_test.csv and fig_dev_vs_test.png.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dispute_rag.analysis import load_per_case  # noqa: E402
from dispute_rag.final_analysis import (  # noqa: E402
    SHORT, best_by_metric, best_model, classification_by_condition, clause_sensitivity, condition_summary,
    dev_test_table, majority_baseline, needs_human_collapse, to_jsonable, without_ambiguous,
)
from dispute_rag.gold import load_gold  # noqa: E402
from dispute_rag.refs import CLAUSE_MATCH_MODES  # noqa: E402
from dispute_rag.splits import paths  # noqa: E402
from dispute_rag.viz import SERIES, TEXT, TEXT_2, apply_style, finish  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402


def dump(obj, path: Path) -> None:
    path.write_text(json.dumps(to_jsonable(obj), indent=2, default=float))


def fig_confusions(cls: dict, out: Path, title_suffix: str) -> None:
    conds = list(cls)
    ncol = 2 if len(conds) > 1 else 1
    nrow = int(np.ceil(len(conds) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 3.2 * nrow), squeeze=False)
    vmax = max(int(cls[c]["confusion_matrix"].to_numpy().max()) for c in conds)
    for ax, c in zip(axes.flat, conds):
        cm = cls[c]["confusion_matrix"]
        ax.imshow(cm.to_numpy(), cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        for (i, j), v in np.ndenumerate(cm.to_numpy()):
            ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                    color="white" if v > vmax * 0.55 else TEXT)
        ax.set_xticks(range(cm.shape[1]), [SHORT.get(x, x) for x in cm.columns], rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(cm.shape[0]), [SHORT.get(x, x) for x in cm.index], fontsize=8)
        ax.set_title(f"{c}: accuracy {cls[c]['accuracy']:.0%}, macro-F1 {cls[c]['macro_f1']:.2f}", fontsize=9)
        ax.grid(False)
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Gold", fontsize=8)
    for ax in list(axes.flat)[len(conds):]:
        ax.axis("off")
    fig.suptitle(f"Confusion matrices by condition{title_suffix}", x=0.02, ha="left",
                 fontsize=11, fontweight="bold")
    finish(fig, out)


def fig_acc_f1(summary: pd.DataFrame, base: dict, out: Path, title_suffix: str) -> None:
    conds = summary.index.tolist()
    x = np.arange(len(conds))
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    w = 0.38
    acc = summary["accuracy"] * 100
    yerr = [acc - summary["accuracy_lo"] * 100, summary["accuracy_hi"] * 100 - acc]
    ax.bar(x - w / 2, acc, w, color=SERIES[0], label="Accuracy (95% Wilson CI)")
    ax.errorbar(x - w / 2, acc, yerr=yerr, fmt="none", ecolor=TEXT, capsize=3, lw=1)
    ax.bar(x + w / 2, summary["macro_f1"] * 100, w, color=SERIES[1], label="Macro-F1 (x100)")
    ax.axhline(base["accuracy"] * 100, color=TEXT_2, ls="--", lw=1)
    ax.text(len(conds) - 0.5, base["accuracy"] * 100 + 1.5,
            f"majority baseline {base['accuracy']:.0%}", ha="right", fontsize=8, color=TEXT_2)
    ax.set_xticks(x, conds)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Percent")
    ax.set_title(f"Accuracy and macro-F1 by condition{title_suffix}")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="x", visible=False)
    finish(fig, out)


def fig_clause(sens: dict, out: Path, title_suffix: str) -> None:
    conds = [c for c in ("C0", "A", "B", "C") if c in sens["strict"]["per_condition"]]
    x = np.arange(len(conds))
    w = 0.26
    labels = {"strict": "Strict (primary)", "parent": "Parent or child", "section": "Same section"}
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for k, m in enumerate(CLAUSE_MATCH_MODES):
        r = [sens[m]["per_condition"][c]["rate"] * 100 for c in conds]
        ax.bar(x + (k - 1) * w, r, w, color=SERIES[k], label=labels[m])
    ax.set_xticks(x, conds)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Cites a governing clause (%)")
    ax.set_title(f"Correct-clause rate under three matching rules{title_suffix}")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis="x", visible=False)
    finish(fig, out)


def run_split(split: str) -> pd.DataFrame | None:
    sp = paths(split)
    long = load_per_case(sp.runs)
    if long.empty:
        print(f"[{split}] no runs under {sp.runs} — skipped")
        return None
    gold = load_gold(sp.gold)
    long = long[long["case_id"].isin(gold)]
    out = sp.final
    out.mkdir(parents=True, exist_ok=True)
    suffix = " (development set)" if split == "dev" else " (test set)"

    cls = classification_by_condition(long)
    for c, r in cls.items():
        r["confusion_matrix"].to_csv(out / f"confusion_{c}.csv")
        r["per_class"].to_csv(out / f"per_class_{c}.csv")
    base = majority_baseline(long)
    dump(base, out / "majority_baseline.json")
    summary = condition_summary(long, cls)
    summary.to_csv(out / "condition_summary.csv")
    best = best_model(summary, long)
    dump(best, out / "best_model.json")
    bbm = best_by_metric(summary)
    bbm.to_csv(out / "best_by_metric.csv", index=False)
    collapse = needs_human_collapse(long)
    dump(collapse, out / "secondary_needs_human.json")
    unamb = without_ambiguous(long, gold)
    dump(unamb, out / "secondary_unambiguous.json")
    sens = clause_sensitivity(sp.runs, long, gold)
    dump(sens, out / "sensitivity_clause_match.json")
    pd.DataFrame({m: {c: v["rate"] for c, v in sens[m]["per_condition"].items()} for m in sens}).to_csv(
        out / "sensitivity_clause_match.csv")

    fig_confusions(cls, out / "fig_confusion_matrices.png", suffix)
    fig_acc_f1(summary, base, out / "fig_accuracy_macro_f1.png", suffix)
    fig_clause(sens, out / "fig_clause_sensitivity.png", suffix)

    print(f"\n=== {split}: n = {base['n']} complaints; majority baseline "
          f"{base['label']} = {base['accuracy']:.1%}")
    print(summary[["n", "accuracy", "macro_f1", "fabricated_rate", "correct_clause_rate",
                   "escalate_share", "latency_p95_s"]].round(3).to_string())
    print("\nBest model:", best.get("statement", best["best"]))
    print(bbm[["metric", "best", "best_value", "runner_up_value"]].round(3).to_string(index=False))
    print("\nNeeds-human collapse, accuracy:",
          {c: round(v["rate"], 3) for c, v in collapse["per_condition"].items()},
          "Holm p vs C0:", {c: round(v["p_holm"], 3) for c, v in collapse["vs_control"].items()})
    print("Without ambiguous (n = %d):" % unamb["n_cases"],
          {c: round(v["rate"], 3) for c, v in unamb["per_condition"].items()})
    print("Clause match rates:")
    print(pd.read_csv(out / "sensitivity_clause_match.csv", index_col=0).round(3).to_string())
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "both"], default="both")
    args = ap.parse_args()
    apply_style()
    res = {s: run_split(s) for s in (["dev", "test"] if args.split == "both" else [args.split])}
    if args.split == "both" and all(v is not None for v in res.values()):
        tab = dev_test_table(res["dev"], res["test"])
        Path("outputs/final").mkdir(parents=True, exist_ok=True)
        tab.to_csv("outputs/final/dev_vs_test.csv")
        print("\nDevelopment vs test:\n", tab.round(3).to_string())
        conds = tab.index.tolist()
        x = np.arange(len(conds))
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        for k, (s, lab) in enumerate((("dev", "Development (pilot)"), ("test", "Test (locked)"))):
            r = res[s].reindex(conds)
            y = r["accuracy"] * 100
            ax.bar(x + (k - 0.5) * 0.38, y, 0.38, color=SERIES[k], label=lab)
            ax.errorbar(x + (k - 0.5) * 0.38, y, yerr=[y - r["accuracy_lo"] * 100, r["accuracy_hi"] * 100 - y],
                        fmt="none", ecolor=TEXT, capsize=3, lw=1)
        ax.set_xticks(x, conds)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Correct determinations (%)")
        ax.set_title("Accuracy on the development and test sets (95% Wilson CI)")
        ax.legend(fontsize=8)
        ax.grid(axis="x", visible=False)
        finish(fig, Path("outputs/final/fig_dev_vs_test.png"))


if __name__ == "__main__":
    main()
