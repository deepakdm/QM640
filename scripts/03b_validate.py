"""Step 3b: preliminary system-level validation (synopsis Table 15 / Appendix D).

Two modes:

  1. Make the sheet, BEFORE running the model:
       python scripts/03b_validate.py --make-sheet --per-rq 8
     Draws 8 gold cases per research question (seed 42) into data/validation/validation_sheet.csv.
     Fill the expected_* columns by hand from the gold label / a manual check, before any run.

  2. Compare, AFTER running the cases through a condition:
       python scripts/03_run_experiment.py configs/hybrid_rerank.yaml --case-ids data/validation/case_ids.txt
       python scripts/03b_validate.py --config configs/hybrid_rerank.yaml

Acceptable-match criteria (Table 15):
  RQ1  predicted determination equals the expected determination (exact)
  RQ2  no fabricated citation AND the expected clause (or a sub-paragraph of it) is cited
  RQ3  computed clause_in_context, dispute_type and narrative length equal the manual values
  RQ4  rule-assigned failure category equals the manual one, and latency/run fields are complete

Outputs: outputs/validation/validation_records.csv (Appendix D) and validation_summary.csv.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from dispute_rag.cfpb_cases import load_cases  # noqa: E402
from dispute_rag.config import ExperimentConfig  # noqa: E402
from dispute_rag.gold import load_gold  # noqa: E402
from dispute_rag.refs import cites_gold  # noqa: E402

SHEET = Path("data/validation/validation_sheet.csv")
COLS = ["case_id", "rq", "input_summary", "expected_determination", "expected_clause",
        "expected_clause_in_context", "expected_dispute_type", "expected_words",
        "expected_failure_category", "comment"]

p = argparse.ArgumentParser()
p.add_argument("--make-sheet", action="store_true")
p.add_argument("--force", action="store_true",
               help="overwrite an existing validation sheet (a timestamped backup is kept)")
p.add_argument("--per-rq", type=int, default=8)
p.add_argument("--config", default="configs/hybrid_rerank.yaml")
p.add_argument("--out", default="outputs/validation")
args = p.parse_args()

if args.make_sheet:
    if SHEET.exists() and not args.force:
        raise SystemExit(f"{SHEET} already exists, so it was NOT overwritten (your expected_* entries are safe).\n"
                         "To create a fresh sheet anyway, add --force; the old one is saved as a timestamped backup.")
    if SHEET.exists():
        import shutil, time
        backup = SHEET.with_name(f"{SHEET.stem}_backup_{time.strftime('%Y%m%d_%H%M%S')}{SHEET.suffix}")
        shutil.copy2(SHEET, backup)
        print(f"Backed up the existing sheet to {backup}")
    gold = load_gold(Path("data/gold/gold_cases.jsonl"))
    cases = {c.case_id: c for c in load_cases(Path("data/cases/cases.jsonl"))}
    ids = sorted(i for i in gold if i in cases)
    rng = random.Random(42)
    rng.shuffle(ids)
    rows = []
    for k, rq in enumerate(["RQ1", "RQ2", "RQ3", "RQ4"]):
        for cid in ids[k * args.per_rq:(k + 1) * args.per_rq]:
            c = cases[cid]
            rows.append({"case_id": cid, "rq": rq,
                         "input_summary": f"{c.dispute_type}; {c.complaint.product}; {len(c.narrative.split())} words",
                         **{col: "" for col in COLS[3:]}})
    SHEET.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=COLS).to_csv(SHEET, index=False)
    (SHEET.parent / "case_ids.txt").write_text("\n".join(sorted({r["case_id"] for r in rows})))
    print(f"Wrote {len(rows)} use cases to {SHEET}. Fill the expected_* columns BEFORE running.")
    sys.exit(0)

cfg = ExperimentConfig.load(args.config)
run_dir = Path("outputs/runs") / cfg.name
metrics = pd.read_csv(run_dir / "per_case_metrics.csv", dtype={"case_id": str}).set_index("case_id")
fails = pd.read_csv(run_dir / "failure_records.csv", dtype={"case_id": str}).set_index("case_id")
cases = {c.case_id: c for c in load_cases(Path("data/cases/cases.jsonl"))}
answers = {}
for line in open(run_dir / "answers.jsonl"):
    if line.strip():
        import json
        a = json.loads(line)
        if a["run_idx"] == 0:
            answers[a["case_id"]] = a

sheet = pd.read_csv(SHEET, dtype=str).fillna("")
records = []
for _, r in sheet.iterrows():
    cid, rq = r.case_id, r.rq
    if cid not in metrics.index:
        records.append({**r, "observed": "not run", "match": None})
        continue
    m, a, c = metrics.loc[cid], answers.get(cid, {}), cases[cid]
    if rq == "RQ1":
        obs = str(m.pred_determination)
        ok = obs == r.expected_determination
    elif rq == "RQ2":
        cited = a.get("cited_section_refs", [])
        cites = cites_gold(cited, [x for x in r.expected_clause.split("|") if x])
        obs = f"cited={'|'.join(cited)}; fabricated={int(m.fabricated_citation)}"
        ok = bool(cites) and int(m.fabricated_citation) == 0
    elif rq == "RQ3":
        words = len(c.narrative.split())
        obs = f"clause_in_context={m.clause_in_context}; type={c.dispute_type}; words={words}"
        ok = (str(int(m.clause_in_context)) == r.expected_clause_in_context
              and c.dispute_type == r.expected_dispute_type and str(words) == r.expected_words)
    elif rq == "RQ4":
        cat = fails.loc[cid, "category"] if cid in fails.index else "missing"
        complete = pd.notna(m.latency_s) and int(m.n_runs) == cfg.evaluation.runs_per_case
        obs = f"category={cat}; runs={int(m.n_runs)}; latency={m.latency_s:.1f}s"
        ok = cat == r.expected_failure_category and complete
    else:
        obs, ok = "unknown RQ", None
    records.append({**r, "observed": obs, "match": ok})

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
rec = pd.DataFrame(records)
rec.to_csv(out / "validation_records.csv", index=False)
summ = (rec.dropna(subset=["match"]).groupby("rq")["match"]
        .agg(tested="size", matched="sum").assign(match_rate=lambda d: d.matched / d.tested))
summ.to_csv(out / "validation_summary.csv")
print(summ.to_string())
print(f"\nMismatches to trace (gold label, parsing, retrieval, scoring or model):")
print(rec[rec.match == False][["case_id", "rq", "observed", "comment"]].to_string(index=False))  # noqa: E712
