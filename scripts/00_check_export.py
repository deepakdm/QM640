"""Step 0: validate a CFPB export before running anything else.

Run this the moment you download an export. It checks the three things that silently produce an
empty case set, and tells you exactly how to fix each one.

Usage:
    python scripts/00_check_export.py
    python scripts/00_check_export.py --csv path/to/export.csv --include-reg-z
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from dispute_rag.cfpb_cases import (
    COLUMN_ALIASES, REG_E_PRODUCTS, REG_Z_PRODUCTS, _MIN_NARRATIVE_WORDS,
    _MAX_NARRATIVE_WORDS, classify_dispute_type, scope_for_product,
)

p = argparse.ArgumentParser()
p.add_argument("--csv", default="data/raw/cfpb/complaints.csv")
p.add_argument("--include-reg-z", action="store_true")
p.add_argument("--rows", type=int, default=50_000, help="rows to inspect for the diagnosis")
args = p.parse_args()

path = Path(args.csv)
if not path.exists():
    sys.exit(f"FAIL  No file at {path}\n      See docs/data_sources.md for the export steps.")

header = pd.read_csv(path, nrows=0).columns.tolist()
print(f"Checking {path}\n{len(header)} columns found\n")

problems = []

# --- Check 1: narrative column -------------------------------------------------------
narrative_col = next((a for a in COLUMN_ALIASES["narrative"] if a in header), None)
if narrative_col:
    print(f"PASS  Narrative column present: '{narrative_col}'")
else:
    print("FAIL  No consumer-narrative column.")
    print("      Found:", ", ".join(header))
    problems.append(
        "Re-export with narratives. On the CFPB search page, filter to complaints that HAVE a\n"
        "  consumer narrative before exporting (the narrative field is only populated when the\n"
        "  consumer consented to publish it). Without it there is no case text to retrieve\n"
        "  against, and a RAG study is not possible on this export."
    )

# --- Check 2: products in scope ------------------------------------------------------
df = pd.read_csv(path, nrows=args.rows, dtype=str, low_memory=False)
prod_col = next((a for a in COLUMN_ALIASES["product"] if a in header), None)
if not prod_col:
    print("FAIL  No Product column — this does not look like a CCDB export.")
    problems.append("Download a fresh export from the CFPB site.")
else:
    counts = df[prod_col].value_counts()
    in_scope = {p for p in counts.index if scope_for_product(p)}
    n_in_scope = int(counts[list(in_scope)].sum()) if in_scope else 0
    print(f"\n{'PASS' if n_in_scope else 'FAIL'}  In-scope rows by product: "
          f"{n_in_scope:,} of {len(df):,} inspected")
    print("      Products present in this export:")
    for prod, n in counts.head(12).items():
        mark = scope_for_product(prod) or "OUT OF SCOPE"
        print(f"        {n:>7,}  {prod[:64]:<64} [{mark}]")
    if not n_in_scope:
        allowed = sorted(REG_E_PRODUCTS | (REG_Z_PRODUCTS if args.include_reg_z else set()))
        problems.append(
            "No rows match an in-scope product. This study covers consumer payment disputes\n"
            "  under Regulation E (and Regulation Z with --include-reg-z). Re-export filtering\n"
            "  Product to one of:\n    - " + "\n    - ".join(allowed) +
            "\n  Mortgage, credit-reporting and debt-collection complaints are governed by other\n"
            "  regulations (RESPA, FCRA, FDCPA) and are out of scope for this corpus."
        )

# --- Check 3: issues classify into dispute types -------------------------------------
issue_col = next((a for a in COLUMN_ALIASES["issue"] if a in header), None)
sub_col = next((a for a in COLUMN_ALIASES["sub_issue"] if a in header), None)
if issue_col:
    pairs = df[[issue_col] + ([sub_col] if sub_col else [])].fillna("")
    types = [classify_dispute_type(r[issue_col], r[sub_col] if sub_col else None)
             for _, r in pairs.iterrows()]
    matched = sum(1 for t in types if t)
    print(f"\n{'PASS' if matched else 'FAIL'}  Issues classified into a dispute type: "
          f"{matched:,} of {len(df):,}")
    if matched:
        print("      Distribution:", pd.Series([t for t in types if t]).value_counts().to_dict())
    else:
        problems.append(
            "No issue text matched a dispute pattern. Either the products are out of scope\n"
            "  (see above), or DISPUTE_TYPE_PATTERNS in src/dispute_rag/cfpb_cases.py needs a\n"
            "  pattern for the issue wording in your export."
        )

# --- Check 4: usable narrative length ------------------------------------------------
if narrative_col:
    wc = df[narrative_col].dropna().str.split().str.len()
    usable = int(((wc >= _MIN_NARRATIVE_WORDS) & (wc <= _MAX_NARRATIVE_WORDS)).sum())
    print(f"\n{'PASS' if usable else 'WARN'}  Narratives of usable length "
          f"({_MIN_NARRATIVE_WORDS}-{_MAX_NARRATIVE_WORDS} words): {usable:,}")
    if len(wc):
        print(f"      Word counts: median {int(wc.median())}, "
              f"10th pct {int(wc.quantile(.1))}, 90th pct {int(wc.quantile(.9))}")

# --- Check 5: study window ---------------------------------------------------------
from dispute_rag.cfpb_cases import DEFAULT_END, DEFAULT_START, parse_dates
date_col = next((a for a in COLUMN_ALIASES["date_received"] if a in header), None)
if date_col:
    d = parse_dates(df[date_col])
    inside = int(d.between(pd.Timestamp(DEFAULT_START), pd.Timestamp(DEFAULT_END)).sum())
    print(f"\n{'PASS' if inside else 'WARN'}  Rows inside the study window "
          f"{DEFAULT_START} to {DEFAULT_END}: {inside:,} of {len(df):,} inspected "
          f"(dates span {d.min():%Y-%m-%d} to {d.max():%Y-%m-%d})")

# --- Verdict --------------------------------------------------------------------------
print("\n" + "=" * 72)
if problems:
    print(f"NOT READY — {len(problems)} problem(s) to fix:\n")
    for i, prob in enumerate(problems, 1):
        print(f"{i}. {prob}\n")
    sys.exit(1)
print("READY — run: python scripts/01_prepare_data.py --n 160 --seed 42")
