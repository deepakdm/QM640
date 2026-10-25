"""Step 1: build the case set from the CFPB Consumer Complaint Database and export the
labeling sheet.

No LLM is called in this step. Cases are real published complaints, selected by filter and
stratified sample only — nothing is generated. That keeps Phase 1 outside the scope of the
course's generative-AI approval and makes the sampling fully reproducible from the seed.

Usage:
    python scripts/01_prepare_data.py --n 160
    python scripts/01_prepare_data.py --n 160 --include-reg-z     # adds credit-card billing disputes
    python scripts/01_prepare_data.py --csv data/raw/cfpb/my_export.csv

Input: a CCDB export at data/raw/cfpb/complaints.csv. Either the full bulk file
(https://files.consumerfinance.gov/ccdb/complaints.csv.zip) or a filtered export from
https://www.consumerfinance.gov/data-research/consumer-complaints/search/ works; the filtered
export is much smaller and is the recommended route.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dispute_rag.cfpb_cases import (
    DEFAULT_END, DEFAULT_START, filter_report, load_candidates, sample_cases, save_cases, to_cases,
)
from dispute_rag.gold import export_labeling_sheet

p = argparse.ArgumentParser()
p.add_argument("--csv", default="data/raw/cfpb/complaints.csv",
               help="CCDB export (bulk or filtered)")
p.add_argument("--n", type=int, default=160, help="target gold-set size (synopsis: 160)")
p.add_argument("--seed", type=int, default=42, help="sampling seed, recorded in the report")
p.add_argument("--include-reg-z", action="store_true",
               help="also include credit-card billing disputes (requires Reg Z in the corpus)")
p.add_argument("--start", default=DEFAULT_START, help="first date received (YYYY-MM-DD)")
p.add_argument("--end", default=DEFAULT_END, help="last date received / download date (YYYY-MM-DD)")
args = p.parse_args()

csv_path = Path(args.csv)
if not csv_path.exists():
    sys.exit(f"CCDB export not found at {csv_path}. See docs/data_sources.md for the download steps.")

print(f"Scanning {csv_path} ...")
funnel: dict = {}
candidates = load_candidates(csv_path, include_reg_z=args.include_reg_z,
                             start=args.start, end=args.end, funnel=funnel)
print(f"  in-scope complaints with usable narratives: {len(candidates):,}")
for k, v in funnel.items():
    print(f"    {k:<22} {v:>10,}")

sampled = sample_cases(candidates, n=args.n, seed=args.seed)
cases = to_cases(sampled)
save_cases(cases, Path("data/cases/cases.jsonl"))

report = filter_report(csv_path, candidates, sampled, funnel, args.start, args.end)
report["seed"] = args.seed
report["include_reg_z"] = args.include_reg_z
Path("data/cases").mkdir(parents=True, exist_ok=True)
with open("data/cases/selection_report.json", "w") as f:
    json.dump(report, f, indent=2, default=str)

# Flat copy of the sample for the repository's processed-data folder and the report screenshot.
sampled.drop(columns=["scope"], errors="ignore").assign(
    narrative_words=sampled["narrative"].str.split().str.len()
).to_csv("data/cases/sample_160.csv", index=False)

export_labeling_sheet(cases, Path("data/gold/labeling_sheet.csv"))

print(f"\nSampled {len(cases)} cases -> data/cases/cases.jsonl")
print("Dispute types:", report["dispute_type_counts"])
print("Provenance    -> data/cases/selection_report.json  (paste into the methods section)")
print("Next: python scripts/01b_eda.py  (cleaning log + EDA figures)")
print("\nThen: label data/gold/labeling_sheet.csv (see docs/gold_labeling_protocol.md),")
print("then run scripts/02_import_gold.py")
