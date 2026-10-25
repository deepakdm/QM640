"""Step 1b: data-quality checks, cleaning log and exploratory data analysis (EDA).

Runs after 01_prepare_data.py and before any model call. Reads the raw export in chunks (it can
be several GB), the 160-case sample and, if present, the gold labels, and writes every count,
table and figure the interim report's Data Cleaning and EDA sections use.

Usage:
    python scripts/01b_eda.py
    python scripts/01b_eda.py --csv data/raw/cfpb/complaints.csv --out outputs/eda

Outputs (outputs/eda/):
    cleaning_log.csv          rows surviving each filter, with counts and percentages removed
    missingness.csv           missing values per column in the raw export
    eda_summary.json          every number quoted in the report
    fig*_*.png                report figures
    *.csv                     the tables behind each figure
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from dispute_rag.cfpb_cases import (
    COLUMN_ALIASES, DEFAULT_END, DEFAULT_START, FUNNEL_STEPS, _MAX_NARRATIVE_WORDS,
    _MIN_NARRATIVE_WORDS, classify_dispute_type, load_cases, parse_dates, scope_for_product,
)
from dispute_rag.viz import CONDITION_COLORS, SERIES, TEXT, TEXT_2, apply_style, finish  # noqa: F401

import matplotlib.pyplot as plt  # noqa: E402

XXXX_RE = re.compile(r"\bX{2,}\b")

p = argparse.ArgumentParser()
p.add_argument("--csv", default="data/raw/cfpb/complaints.csv")
p.add_argument("--cases", default="data/cases/cases.jsonl")
p.add_argument("--selection", default="data/cases/selection_report.json")
p.add_argument("--gold", default="data/gold/gold_cases.jsonl")
p.add_argument("--start", default=DEFAULT_START)
p.add_argument("--end", default=DEFAULT_END)
p.add_argument("--out", default="outputs/eda")
args = p.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
apply_style()
summary: dict = {}


def pct(a, b):
    return round(100 * a / b, 2) if b else None


# ---------------------------------------------------------------------------------------------
# 1. Raw export: one chunked pass for missingness, duplicates and population-level distributions
# ---------------------------------------------------------------------------------------------
csv_path = Path(args.csv)
if not csv_path.exists():
    sys.exit(f"No export at {csv_path}. Run 00_check_export.py first.")
header = pd.read_csv(csv_path, nrows=0).columns.tolist()
col = {k: next((a for a in v if a in header), None) for k, v in COLUMN_ALIASES.items()}

n_rows = 0
missing = Counter()
blank_narr = 0
id_counts = Counter()
products, issues = Counter(), Counter()
t0, t1 = pd.Timestamp(args.start), pd.Timestamp(args.end)
pool_rows = []   # in-scope, narrative, in window: compact per-row features for EDA
for chunk in pd.read_csv(csv_path, dtype=str, chunksize=200_000, low_memory=False):
    n_rows += len(chunk)
    for c in chunk.columns:
        missing[c] += int((chunk[c].isna() | (chunk[c].fillna("").str.strip() == "")).sum())
    id_counts.update(chunk[col["complaint_id"]].tolist())
    products.update(chunk[col["product"]].fillna("(missing)").tolist())
    narr = chunk[col["narrative"]].fillna("") if col["narrative"] else pd.Series("", index=chunk.index)
    has = narr.str.strip() != ""
    d = parse_dates(chunk[col["date_received"]]) if col["date_received"] else pd.Series(pd.NaT, index=chunk.index)
    scope = chunk[col["product"]].map(scope_for_product)
    keep = has & d.between(t0, t1) & (scope == "reg_e")
    sub = chunk[keep]
    if sub.empty:
        continue
    text = narr[keep]
    get = lambda k: sub[col[k]] if col.get(k) else pd.Series([None] * len(sub), index=sub.index)  # noqa: E731
    pool_rows.append(pd.DataFrame({
        "complaint_id": get("complaint_id"),
        "date_received": d[keep],
        "product": get("product"),
        "issue": get("issue"),
        "sub_issue": get("sub_issue"),
        "company": get("company"),
        "state": get("state"),
        "submitted_via": get("submitted_via"),
        "company_response": get("company_response"),
        "timely_response": get("timely_response"),
        "words": text.str.split().str.len(),
        "xxxx_tokens": text.str.count(XXXX_RE.pattern),
        "norm_text_hash": text.str.split().str.join(" ").str.lower().map(hash),
    }))
    issues.update(get("issue").fillna("(missing)").tolist())

pool = pd.concat(pool_rows, ignore_index=True) if pool_rows else pd.DataFrame()
summary["raw"] = {
    "file": str(csv_path), "rows": n_rows, "columns": len(header), "column_names": header,
    "duplicate_complaint_ids": int(sum(v - 1 for v in id_counts.values() if v > 1)),
    "product_counts": dict(products.most_common()),
}

miss = pd.DataFrame({"column": header,
                     "missing": [missing[c] for c in header]})
miss["missing_pct"] = (100 * miss["missing"] / max(n_rows, 1)).round(2)
miss.to_csv(out / "missingness.csv", index=False)
summary["raw"]["missingness"] = miss.set_index("column")["missing_pct"].to_dict()

# ---------------------------------------------------------------------------------------------
# 2. Cleaning log: the funnel recorded by 01_prepare_data.py
# ---------------------------------------------------------------------------------------------
sel = json.load(open(args.selection)) if Path(args.selection).exists() else {}
funnel = sel.get("funnel") or []
if funnel:
    log = pd.DataFrame(funnel)
    log["removed"] = (log["rows"].shift(1) - log["rows"]).fillna(0).astype(int)
    log["removed_pct_of_previous"] = (100 * log["removed"] / log["rows"].shift(1)).round(2)
    log["retained_pct_of_raw"] = (100 * log["rows"] / log["rows"].iloc[0]).round(3)
    log.to_csv(out / "cleaning_log.csv", index=False)
    summary["funnel"] = log.to_dict(orient="records")

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    labels = list(log["description"])
    vals = list(log["rows"].astype(int))
    if sel.get("sampled"):
        labels.append(f"Stratified sample (seed {sel.get('seed', 42)})")
        vals.append(int(sel["sampled"]))
    labels, vals = labels[::-1], np.array(vals[::-1])
    ax.barh(labels, vals, color=SERIES[0], height=0.6)
    for y, v in enumerate(vals):
        ax.text(v + vals.max() * 0.01, y, f"{int(v):,}", va="center", fontsize=8, color=TEXT)
    ax.set_xlabel("Complaints remaining")
    ax.set_title("Complaints remaining after each cleaning step")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, vals.max() * 1.15)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x):,}"))
    finish(fig, out / "fig1_cleaning_funnel.png")

# ---------------------------------------------------------------------------------------------
# 3. Candidate pool (Reg E products, narrative present, inside the window)
# ---------------------------------------------------------------------------------------------
if not pool.empty:
    pool["dispute_type"] = [classify_dispute_type(i, s) for i, s in zip(pool["issue"], pool["sub_issue"])]
    w = pool["words"].astype(float)
    q1, q3 = w.quantile([.25, .75])
    iqr = q3 - q1
    summary["pool"] = {
        "rows": int(len(pool)),
        "dup_ids_in_pool": int(pool["complaint_id"].duplicated().sum()),
        "dup_narratives_in_pool": int(pool["norm_text_hash"].duplicated().sum()),
        "words": {k: float(v) for k, v in w.describe(percentiles=[.1, .25, .5, .75, .9, .99]).items()},
        "words_skew": float(w.skew()),
        "below_min_words": int((w < _MIN_NARRATIVE_WORDS).sum()),
        "above_max_words": int((w > _MAX_NARRATIVE_WORDS).sum()),
        "iqr_upper_fence": float(q3 + 1.5 * iqr),
        "iqr_outliers_high": int((w > q3 + 1.5 * iqr).sum()),
        "share_with_redaction": float((pool["xxxx_tokens"] > 0).mean()),
        "redaction_tokens_median": float(pool["xxxx_tokens"].median()),
        "redaction_token_share_median": float((pool["xxxx_tokens"] / w.clip(lower=1)).median()),
        "unmapped_dispute_type": int(pool["dispute_type"].isna().sum()),
        "dispute_type_counts": pool["dispute_type"].fillna("(unmapped)").value_counts().to_dict(),
        "product_counts": pool["product"].value_counts().to_dict(),
        "top_issues": pool["issue"].value_counts().head(10).to_dict(),
        "company_response_counts": pool["company_response"].fillna("(missing)").value_counts().to_dict(),
        "timely_yes_share": float((pool["timely_response"] == "Yes").mean()),
        "submitted_via_counts": pool["submitted_via"].fillna("(missing)").value_counts().to_dict(),
        "n_companies": int(pool["company"].nunique()),
        "top10_company_share": float(pool["company"].value_counts().head(10).sum() / len(pool)),
        "top_companies": pool["company"].value_counts().head(10).to_dict(),
        "date_min": str(pool["date_received"].min().date()),
        "date_max": str(pool["date_received"].max().date()),
    }
    # Fig 2: narrative length distribution with the 40/600 inclusion limits
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    bins = np.logspace(0, np.log10(max(w.max(), 10)), 50)
    ax.hist(w, bins=bins, color=SERIES[0], edgecolor="white", linewidth=0.5)
    ax.set_xscale("log")
    for x, lab in ((_MIN_NARRATIVE_WORDS, "40"), (_MAX_NARRATIVE_WORDS, "600")):
        ax.axvline(x, color=TEXT, lw=1, ls="--")
        ax.text(x, ax.get_ylim()[1] * 0.95, f" {lab} words", fontsize=8, color=TEXT, va="top")
    ax.set_xlabel("Narrative length in words (log scale)")
    ax.set_ylabel("Complaints")
    ax.set_title("Narrative length in the Regulation E candidate pool")
    finish(fig, out / "fig2_narrative_length.png")

    # Fig 3: monthly volume by product
    m = pool.assign(month=pool["date_received"].dt.to_period("M").dt.to_timestamp())
    monthly = m.groupby(["month", "product"]).size().unstack(fill_value=0)
    monthly.to_csv(out / "monthly_volume_by_product.csv")
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    order = monthly.sum().sort_values(ascending=False).index
    for i, prod in enumerate(order):
        ax.plot(monthly.index, monthly[prod], color=SERIES[i], label=prod)
    ax.set_ylabel("Complaints per month")
    ax.set_title("Monthly complaints with narratives, by product")
    ax.legend(fontsize=8, loc="upper left")
    finish(fig, out / "fig3_monthly_volume.png")

    # Fig 4: redaction density
    share = (pool["xxxx_tokens"] / w.clip(lower=1)) * 100
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ax.hist(share.clip(upper=30), bins=30, color=SERIES[0], edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Redacted tokens (XXXX) as % of narrative words (capped at 30%)")
    ax.set_ylabel("Complaints")
    ax.set_title("How much of each narrative is redacted")
    finish(fig, out / "fig4_redaction_density.png")

# ---------------------------------------------------------------------------------------------
# 4. The study sample
# ---------------------------------------------------------------------------------------------
if Path(args.cases).exists():
    cases = load_cases(Path(args.cases))
    s = pd.DataFrame([{
        "case_id": c.case_id, "dispute_type": c.dispute_type, "product": c.complaint.product,
        "company": c.complaint.company, "company_response": c.complaint.company_response,
        "date_received": c.complaint.date_received, "words": len(c.narrative.split()),
        "xxxx_tokens": len(XXXX_RE.findall(c.narrative)),
    } for c in cases])
    s.to_csv(out / "sample_overview.csv", index=False)
    summary["sample"] = {
        "n": int(len(s)),
        "dispute_type_counts": s["dispute_type"].value_counts().to_dict(),
        "product_counts": s["product"].value_counts().to_dict(),
        "company_response_counts": s["company_response"].fillna("(missing)").value_counts().to_dict(),
        "words": {k: float(v) for k, v in s["words"].describe().items()},
        "share_with_redaction": float((s["xxxx_tokens"] > 0).mean()),
        "n_companies": int(s["company"].nunique()),
        "top_company_share": float(s["company"].value_counts().iloc[0] / len(s)) if len(s) else None,
        "date_min": str(s["date_received"].min()), "date_max": str(s["date_received"].max()),
    }
    pd.crosstab(s["dispute_type"], s["product"], margins=True).to_csv(out / "sample_type_by_product.csv")

    # Fig 5: dispute-type mix, pool vs sample (shares)
    if not pool.empty:
        a = pool["dispute_type"].dropna().value_counts(normalize=True)
        b = s["dispute_type"].value_counts(normalize=True)
        mix = pd.DataFrame({"Candidate pool": a, "Study sample": b}).fillna(0).sort_values("Candidate pool")
        mix.to_csv(out / "dispute_type_mix.csv")
        fig, ax = plt.subplots(figsize=(7.5, 3.8))
        y = np.arange(len(mix))
        ax.barh(y + 0.2, mix["Candidate pool"] * 100, height=0.38, color=SERIES[0], label="Candidate pool")
        ax.barh(y - 0.2, mix["Study sample"] * 100, height=0.38, color=SERIES[1], label="Study sample")
        ax.set_yticks(y, [t.replace("_", " ") for t in mix.index])
        ax.set_xlabel("Share of complaints (%)")
        ax.set_title("Dispute-type mix: candidate pool vs. study sample")
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
        ax.grid(axis="y", visible=False)
        finish(fig, out / "fig5_dispute_type_mix.png")

# ---------------------------------------------------------------------------------------------
# 5. Gold labels (only if labeling has started)
# ---------------------------------------------------------------------------------------------
from dispute_rag.gold import clause_agreement, cohen_kappa, load_gold  # noqa: E402

gold = load_gold(Path(args.gold)) if Path(args.gold).exists() else {}
if Path(args.cases).exists() and not gold:
    print("No gold labels yet; skipping the gold-label tables and figure.")
if Path(args.cases).exists() and gold:
    g = pd.DataFrame([{"case_id": k, "determination": v.determination, "ambiguous": v.ambiguous,
                       "n_clauses": len(v.governing_clauses), "clauses": v.governing_clauses}
                      for k, v in gold.items()])
    g = g.merge(s[["case_id", "dispute_type", "company_response"]], on="case_id", how="left")
    clause_counts = Counter(c for cl in g["clauses"] for c in cl)
    summary["gold"] = {
        "n_labeled": int(len(g)),
        "determination_counts": g["determination"].value_counts().to_dict(),
        "ambiguous_share": float(g["ambiguous"].mean()),
        "clauses_per_case_mean": float(g["n_clauses"].mean()),
        "top_clauses": dict(clause_counts.most_common(10)),
        "kappa": cohen_kappa(list(gold.values())),
        "clause_jaccard": clause_agreement(list(gold.values())),
    }
    ct = pd.crosstab(g["determination"], g["company_response"].fillna("(missing)"))
    ct.to_csv(out / "gold_vs_company_response.csv")
    pd.crosstab(g["dispute_type"], g["determination"]).to_csv(out / "gold_by_dispute_type.csv")

    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    im = ax.imshow(ct.to_numpy(), cmap="Blues", aspect="auto")
    ax.set_xticks(range(ct.shape[1]), [c.replace("Closed with ", "") for c in ct.columns], rotation=30, ha="right")
    ax.set_yticks(range(ct.shape[0]), [r.replace("_", " ") for r in ct.index])
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            v = ct.iat[i, j]
            r, g_, b, _ = im.cmap(im.norm(v))
            dark = (0.2126 * r + 0.7152 * g_ + 0.0722 * b) < 0.5
            ax.text(j, i, str(v), ha="center", va="center", fontsize=8, color="white" if dark else TEXT)
    ax.grid(False)
    ax.set_title("Gold determination vs. what the company did")
    finish(fig, out / "fig6_gold_vs_company.png")

with open(out / "eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"Wrote EDA outputs to {out}/")
for k in ("raw", "pool", "sample", "gold"):
    if k in summary:
        print(f"  {k}: " + ", ".join(f"{kk}={vv}" for kk, vv in summary[k].items()
                                     if isinstance(vv, (int, float, str)) and kk != "file"))
