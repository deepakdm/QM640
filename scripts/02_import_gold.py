"""Step 2: import the human labels and report reliability.

Outputs the two reliability figures the report needs: Cohen's kappa on the determination and
Jaccard overlap on the governing clauses. Also cross-tabs the rater's regulation-based
determination against the company's actual disposition — descriptive context for the sample,
never a correctness check on the labels.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dispute_rag.cfpb_cases import load_cases
from dispute_rag.gold import (
    clause_agreement, cohen_kappa, company_response_divergence, import_labels,
)

labels = import_labels(Path("data/gold/labeling_sheet.csv"), Path("data/gold/gold_cases.jsonl"))
cases = load_cases(Path("data/cases/cases.jsonl"))

k = cohen_kappa(labels)
j = clause_agreement(labels)
n_ambig = sum(1 for g in labels if g.ambiguous)

print(f"Imported {len(labels)} gold labels ({n_ambig} flagged ambiguous).")
print(f"Cohen's kappa (determination): {k:.3f}" if k is not None
      else "Cohen's kappa: n/a — fewer than 10 double-rated cases; "
           "state single-rater labeling as a limitation.")
print(f"Clause agreement (Jaccard):   {j:.3f}" if j is not None
      else "Clause agreement: n/a — no second-rater clauses recorded.")

div = company_response_divergence(labels, cases)
if not div.empty:
    print("\nGold determination vs. company disposition (descriptive only):")
    print(div.to_string())
    div.to_csv("data/gold/company_response_divergence.csv")
