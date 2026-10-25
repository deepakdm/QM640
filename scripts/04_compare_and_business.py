"""Step 4: confirmatory tests (RQ1, RQ2), driver analysis (RQ3), adoption benchmarks and failure
modes (RQ4), and the cost model.

Reads outputs/runs/<config>/per_case_metrics.csv and failure_records.csv for every condition that
has been run; works with a partial set (e.g. only C0 and A at the interim stage) and skips what
cannot yet be computed.

Outputs (outputs/reports/):
    condition_summary.csv          one row per condition, every pillar
    rq1_accuracy.json              McNemar vs C0 (Holm), bootstrap CI of the gain, Cochran's Q
    rq2_grounding.json             fabricated-citation and correct-clause tests vs C0
    rq3_drivers_odds_ratios.csv    GEE odds ratios; rq3_drivers.json with AMEs and mediation share
    rq4_benchmarks.csv             adoption benchmark table (met / not met / inconclusive)
    rq4_failure_modes.csv          failure categories by condition; rq4_dominant_failure.json
    business_sensitivity.csv       cost model over accuracy; business_tornado.csv
    fig*.png                       report figures
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dispute_rag.analysis import (  # noqa: E402
    CONDITION_LABELS, benchmark_table, compare_to_control, dominant_failure, load_per_case,
)
from dispute_rag.business_model import Assumptions, breakeven_accuracy, cost_per_case, sensitivity, tornado  # noqa: E402
from dispute_rag.cfpb_cases import load_cases  # noqa: E402
from dispute_rag.gold import load_gold  # noqa: E402
from dispute_rag.viz import CONDITION_COLORS, SERIES, TEXT, TEXT_2, apply_style, finish  # noqa: E402

import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--runs", default="outputs/runs")
p.add_argument("--out", default="outputs/reports")
p.add_argument("--n-boot", type=int, default=500, help="cluster-bootstrap reps for RQ3 mediation")
args = p.parse_args()

runs, reports = Path(args.runs), Path(args.out)
reports.mkdir(parents=True, exist_ok=True)
apply_style()

long = load_per_case(runs)
if long.empty:
    sys.exit("No completed runs found under outputs/runs/. Run scripts/03_run_experiment.py first.")
order = [c for c in ("C0", "A", "B", "C") if c in set(long["condition"])]
print("Conditions found:", order)

# Case factors for RQ3
cases = {c.case_id: c for c in load_cases(Path("data/cases/cases.jsonl"))}
gold = load_gold(Path("data/gold/gold_cases.jsonl"))
long["dispute_type"] = long["case_id"].map(lambda i: cases[i].dispute_type if i in cases else None)
long["narrative_words"] = long["case_id"].map(lambda i: len(cases[i].narrative.split()) if i in cases else np.nan)
long["ambiguous"] = long["case_id"].map(lambda i: int(gold[i].ambiguous) if i in gold else np.nan)

# ---- Descriptive ----------------------------------------------------------------------------
summ = {}
for c in order:
    d = long[long.condition == c]
    summ[c] = json.load(open(runs / d["config"].iloc[0] / "summary.json"))
pd.DataFrame(summ).T.to_csv(reports / "condition_summary.csv")
print(pd.DataFrame(summ).T.round(3).to_string())

# ---- RQ1 and RQ2 ----------------------------------------------------------------------------
acc = compare_to_control(long, "determination_correct")
fab = compare_to_control(long, "fabricated_citation")
cla = compare_to_control(long, "correct_clause_cited")
json.dump(acc, open(reports / "rq1_accuracy.json", "w"), indent=2, default=float)
json.dump({"fabricated_citation": fab, "correct_clause_cited": cla},
          open(reports / "rq2_grounding.json", "w"), indent=2, default=float)

def _rate_plot(res, title, fname, ylabel):
    pc = res["per_condition"]
    conds = [c for c in order if c in pc]
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    for i, c in enumerate(conds):
        r, (lo, hi) = pc[c]["rate"], pc[c]["wilson_ci"]
        ax.bar(i, r * 100, color=CONDITION_COLORS[c], width=0.6)
        ax.errorbar(i, r * 100, yerr=[[max(0.0, 100 * (r - lo))], [max(0.0, 100 * (hi - r))]], color=TEXT, capsize=4, lw=1)
        ax.text(i, hi * 100 + 2, f"{r:.0%}", ha="center", fontsize=9, color=TEXT)
    ax.set_xticks(range(len(conds)), conds)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 105)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    finish(fig, reports / fname)

_rate_plot(acc, "Determination accuracy by condition (95% Wilson CI)", "fig7_accuracy_by_condition.png", "Correct determinations (%)")
_rate_plot(fab, "Determinations citing a non-existent section (95% Wilson CI)", "fig8_fabricated_by_condition.png", "Fabricated citation (%)")

# ---- RQ3 ------------------------------------------------------------------------------------
ret = long[long.condition.isin(["A", "B", "C"])].copy()
ret["config"] = ret["condition"]
if ret["condition"].nunique() >= 2 and ret["clause_in_context"].notna().any():
    from dispute_rag.drivers import driver_analysis
    try:
        drv = driver_analysis(ret, n_boot=args.n_boot)
        drv["odds_ratios"].to_csv(reports / "rq3_drivers_odds_ratios.csv")
        drv["vif"].to_csv(reports / "rq3_vif.csv")
        json.dump({k: v for k, v in drv.items() if k not in ("odds_ratios", "odds_ratios_total_model", "vif")},
                  open(reports / "rq3_drivers.json", "w"), indent=2, default=float)
        orr = drv["odds_ratios"].drop(index="const")
        fig, ax = plt.subplots(figsize=(6.5, 0.35 * len(orr) + 1.2))
        y = np.arange(len(orr))[::-1]
        ax.errorbar(orr["odds_ratio"], y, xerr=[orr["odds_ratio"] - orr["or_ci_low"], orr["or_ci_high"] - orr["odds_ratio"]],
                    fmt="o", color=SERIES[0], ecolor=TEXT_2, capsize=3, ms=6)
        ax.axvline(1, color=TEXT, lw=1, ls="--")
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.1, 0.25, 0.5, 1, 2, 4, 10]))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_yticks(y, [t.replace("_", " ").replace("log words c", "log narrative length (centred)") for t in orr.index])
        ax.set_xlabel("Odds ratio for a correct determination (log scale, 95% CI)")
        ax.set_title("Drivers of correct determinations (GEE)")
        finish(fig, reports / "fig9_driver_forest.png")
        print("\nRQ3 engine:", drv["engine"])
        print(drv["odds_ratios"].round(3).to_string())
    except Exception as e:  # small interim samples can fail to converge
        print(f"\nRQ3 driver model skipped: {e!r}")
else:
    print("\nRQ3 needs at least two retrieval conditions — skipped.")

# ---- RQ4 ------------------------------------------------------------------------------------
recs = []
for c in order:
    f = runs / long[long.condition == c]["config"].iloc[0] / "failure_records.csv"
    if f.exists():
        r = pd.read_csv(f, dtype={"case_id": str})
        r["condition"] = c
        recs.append(r)
recs = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame()
if not recs.empty:
    ft = recs.groupby(["condition", "category"]).size().unstack(fill_value=0).reindex(order)
    ft.to_csv(reports / "rq4_failure_modes.csv")
    dom = dominant_failure(recs)
    json.dump(dom, open(reports / "rq4_dominant_failure.json", "w"), indent=2, default=float)
    cats = [c for c in ft.columns if c != "correct"]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    left = np.zeros(len(ft))
    for i, cat in enumerate(cats):
        ax.barh(ft.index, ft[cat], left=left, color=SERIES[i], label=cat.replace("_", " "),
                edgecolor="white", linewidth=1)
        left += ft[cat].to_numpy()
    ax.invert_yaxis()
    ax.set_xlabel("Incorrect determinations")
    ax.set_title("Failure modes by condition")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    ax.grid(axis="y", visible=False)
    finish(fig, reports / "fig10_failure_modes.png")

# ---- Cost model and benchmarks ---------------------------------------------------------------
a = Assumptions()
cand = long[long.condition == "C"]
if not cand.empty:
    a.review_share = float(cand["review_required"].mean())
be = breakeven_accuracy(a)
sensitivity(a).to_csv(reports / "business_sensitivity.csv", index=False)
acc_c = float(cand["determination_correct"].mean()) if not cand.empty else 0.70
tornado(a, accuracy=acc_c).to_csv(reports / "business_tornado.csv", index=False)
print(f"\nBreak-even accuracy (review share {a.review_share:.2f}): {be:.3f}")

if not cand.empty:
    cost_ok = cost_per_case(acc_c, a)["minutes_saved_pct"] > 0
    bt = benchmark_table(long, acc, cla, breakeven=be, cost_ok=cost_ok)
    bt.to_csv(reports / "rq4_benchmarks.csv", index=False)
    print("\nAdoption benchmarks (Config C)")
    print(bt.to_string(index=False))

sens = sensitivity(a, np.arange(0.3, 1.0001, 0.05))
fig, ax = plt.subplots(figsize=(6.5, 3.4))
ax.plot(sens.accuracy * 100, sens.minutes_saved_pct * 100, color=SERIES[0], marker="o", ms=4)
ax.axhline(0, color=TEXT, lw=0.8)
ax.axvline(be * 100, color=TEXT_2, ls="--", lw=1)
ax.text(be * 100, ax.get_ylim()[1] * 0.9, f"  break-even {be:.0%}", fontsize=8, color=TEXT)
for c in order:
    if c == "C0":
        continue
    x = long[long.condition == c]["determination_correct"].mean() * 100
    ax.plot([x], [cost_per_case(x / 100, a)["minutes_saved_pct"] * 100], "o", color=CONDITION_COLORS[c], ms=8)
    ax.annotate(c, (x, cost_per_case(x / 100, a)["minutes_saved_pct"] * 100), xytext=(5, -12),
                textcoords="offset points", fontsize=9, color=TEXT)
ax.set_xlabel("Assistant determination accuracy (%)")
ax.set_ylabel("Handling time saved (%)")
ax.set_title("Modeled time saving by accuracy (assumptions, not observed)")
finish(fig, reports / "fig11_cost_sensitivity.png")
print(f"\nWrote analysis artifacts to {reports}/")
