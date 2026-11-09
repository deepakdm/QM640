# Gold-label protocol

The same protocol applies to the development (pilot) sheet `data/gold/labeling_sheet.csv` and the test
sheet `data/gold/test/labeling_sheet.csv`.

## Roles

- **Primary rater (the author).** Labels every complaint. Drafts may be prepared with AI assistance,
  but every determination, clause and rationale is reviewed and accepted or corrected by the author;
  declare this in Appendix A of the report.
- **Second rater.** Rates about 30 complaints per set **blind** to the primary labels, on the separate
  sheet `second_rater_sheet.csv` (made by `scripts/02a_second_rater_sheet.py`). The import script
  merges it and reports Cohen's kappa (determination) and Jaccard overlap (clauses).

## Per complaint

1. Read the narrative, product, issue and sub-issue.
2. Find the governing clause(s) **in the corpus** (`data/corpus/`). Enter exact references,
   pipe-separated: `1005.6(b)(1)|1005.11(c)`.
3. Choose one determination:
   - `approve_dispute`: the claim succeeds on the stated facts and the rule
   - `deny_dispute`: the claim fails on the facts or the rule
   - `partial_credit`: liability is capped or split by rule
   - `escalate`: facts suffice, but the case involves a legal dispute, suspected institutional
     misconduct or a deadline the institution already missed
   - `request_more_info`: a fact needed to decide is missing or redacted
4. Write a one-sentence rationale citing the clause; fill `labeled_by`.
5. Set `ambiguous` to `yes` when the published narrative lacks a fact the decision needs.
6. Out of scope (not a Regulation E dispute): leave `determination` blank; the import skips it.

## Rules

- `company_response_CONTEXT_ONLY` is what the firm did, not what the regulation requires. Never copy it.
- A redaction (`XXXX`) is an unknown fact, never evidence.
- If two clauses both apply, list both; retrieval recall is computed over all of them.
- Label in sittings of at most about 40 complaints to limit drift.
- Once the test protocol is locked (`scripts/02b_lock_protocol.py`), test labels are frozen.

## Import and report

```bash
python scripts/02_import_gold.py                 # development set
python scripts/02_import_gold.py --split test    # test set
```

Report: number labelled and dropped, number flagged ambiguous, the determination mix, kappa and
Jaccard for the double-rated subset (or a single-rater limitation), and the cross-tab of gold
determinations against company responses (`company_response_divergence.csv`).
