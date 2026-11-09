"""Step 2a: make the BLIND second-rater sheet (no first-rater labels in it).

    python scripts/02a_second_rater_sheet.py --split test           # 36 drawn -> about 30 in scope
    python scripts/02a_second_rater_sheet.py --split test --n 40

Writes data/gold/<split>/second_rater_sheet.csv with the complaint fields and three empty columns
for the second rater: second_rater_determination, second_rater_clauses, second_rater_name.
The draw is seeded and stratified by dispute type. Because some complaints turn out to be out of
scope, it draws a few more than 30. Give the sheet to the second rater; when it comes back, run
scripts/02_import_gold.py --split <split>, which merges it and reports Cohen's kappa.
The development set already has one (data/gold/second_rater_sheet.csv); it is never overwritten.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dispute_rag.splits import paths  # noqa: E402

COLS = ["case_id", "dispute_type", "product", "issue", "sub_issue", "date_received", "narrative",
        "company_response_CONTEXT_ONLY"]

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["dev", "test"], default="test")
ap.add_argument("--n", type=int, default=36)
ap.add_argument("--seed", type=int, default=30)
args = ap.parse_args()

sp = paths(args.split)
out = sp.labeling.parent / "second_rater_sheet.csv"
if out.exists():
    sys.exit(f"{out} already exists; it is made once so the second rater's work is never overwritten.")
sheet = pd.read_csv(sp.labeling, dtype=str)
per = -(-args.n // sheet["dispute_type"].nunique())
pick = (sheet.groupby("dispute_type").sample(n=per, random_state=args.seed, replace=False)
             .sample(frac=1, random_state=args.seed).head(args.n))
blind = pick[[c for c in COLS if c in pick]].copy()
for c in ("second_rater_determination", "second_rater_clauses", "second_rater_name"):
    blind[c] = ""
blind.to_csv(out, index=False)
print(f"Wrote {out}: {len(blind)} complaints, {blind['dispute_type'].value_counts().to_dict()}")
print("Allowed determinations: approve_dispute, deny_dispute, partial_credit, escalate, request_more_info.\n"
      "Clauses: pipe-separated, e.g. 1005.6(b)(1)|1005.11(c). Rate without seeing the first labels.")
