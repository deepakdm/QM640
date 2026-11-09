"""Step 2: import the human labels and report reliability.

If a blind second-rater sheet exists next to the labeling sheet (second_rater_sheet.csv, made by
scripts/02a_second_rater_sheet.py), its second_rater_determination / second_rater_clauses are merged
in by case_id before kappa is computed. The labeling sheet itself is never modified; the merged copy
is written to labeling_sheet_merged.csv.

Outputs the two reliability figures the report needs: Cohen's kappa on the determination and
Jaccard overlap on the governing clauses. Also cross-tabs the rater's regulation-based
determination against the company's actual disposition — descriptive context for the sample,
never a correctness check on the labels.

    python scripts/02_import_gold.py                # development set (interim pilot)
    python scripts/02_import_gold.py --split test   # locked test set
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dispute_rag.cfpb_cases import load_cases
from dispute_rag.gold import (
    clause_agreement, cohen_kappa, company_response_divergence, import_labels,
)
from dispute_rag.splits import PROTOCOL_LOCK, paths

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["dev", "test"], default="dev")
args = ap.parse_args()
sp = paths(args.split)

if args.split == "test" and PROTOCOL_LOCK.exists():
    sys.exit("The test protocol is locked; the test gold set is frozen. Re-importing would change "
             "a locked file. Remove the lock only if no test output has been produced.")

import pandas as pd  # noqa: E402

src = sp.labeling
second = sp.labeling.parent / "second_rater_sheet.csv"
if second.exists():
    sheet = pd.read_csv(sp.labeling, dtype=str)
    sec = pd.read_csv(second, dtype=str).set_index("case_id")
    filled = sec["second_rater_determination"].fillna("").str.strip().ne("")
    for col in ("second_rater_determination", "second_rater_clauses"):
        if col not in sheet:
            sheet[col] = ""
        from_sec = sheet["case_id"].map(sec.loc[filled, col]) if filled.any() else None
        if from_sec is not None:
            blank = sheet[col].fillna("").str.strip().eq("")
            sheet.loc[blank & from_sec.notna(), col] = from_sec[blank & from_sec.notna()]
    src = sp.labeling.parent / "labeling_sheet_merged.csv"
    sheet.to_csv(src, index=False)
    print(f"Merged {int(filled.sum())} second-rater ratings from {second.name} "
          f"({len(sec)} rows in the sheet).")

labels = import_labels(src, sp.gold)
if not labels:
    sys.exit(f"[{args.split}] No labels found in {sp.labeling}: the 'determination' column is empty. "
             "Label the sheet first, then run this again. (Nothing to lock or run yet.)")
cases = load_cases(sp.cases)

k = cohen_kappa(labels)
j = clause_agreement(labels)
n_ambig = sum(1 for g in labels if g.ambiguous)
n_double = sum(1 for g in labels if g.second_rater_determination)

print(f"[{args.split}] Imported {len(labels)} gold labels ({n_ambig} flagged ambiguous, "
      f"{n_double} double-rated).")
print(f"Cohen's kappa (determination): {k:.3f}" if k is not None
      else "Cohen's kappa: n/a — fewer than 10 double-rated cases; "
           "state single-rater labeling as a limitation.")
print(f"Clause agreement (Jaccard):   {j:.3f}" if j is not None
      else "Clause agreement: n/a — no second-rater clauses recorded.")

sp.gold.parent.mkdir(parents=True, exist_ok=True)
(sp.gold.parent / "reliability.json").write_text(json.dumps(
    {"split": args.split, "n_labels": len(labels), "n_ambiguous": n_ambig, "n_double_rated": n_double,
     "cohen_kappa": k, "clause_jaccard": j}, indent=2))

div = company_response_divergence(labels, cases)
if not div.empty:
    print("\nGold determination vs. company disposition (descriptive only):")
    print(div.to_string())
    div.to_csv(sp.gold.parent / "company_response_divergence.csv")
