# Data storage and access

## Storage layers

| Layer | Location | Format | Role |
|---|---|---|---|
| Raw | `data/raw/cfpb/complaints.csv.gz` | gzip CSV, 16 columns | Merged CFPB narratives archive (source of record) |
| Corpus | `data/corpus/reg_e.md`, `reg_e_interp.md` | Markdown with section markers | Retrieval knowledge base |
| Processed | `data/cases/` (dev), `data/cases/test/` (test) | JSONL + CSV | Sampled complaints |
| Labels | `data/gold/`, `data/gold/test/` | CSV sheet → JSONL | Gold determinations and clauses |
| Splits | `data/splits/split_manifest.json`, `protocol_lock.json` | JSON | Dev/test membership, SHA-256 lock |
| Outputs | `outputs/runs*/<config>/` | JSONL + CSV | Raw model answers and scored metrics |
| Analysis DB | `data/db/qm640.sqlite` | SQLite 3 | Everything above in one queryable file (derived) |

Flat files are the source of record because each step's output can be inspected and diffed in Git.
The SQLite database is rebuilt from them by `python scripts/00e_build_db.py` and is not committed
(it is about 11 MB and fully reproducible).

## Schema

`complaints` (case_id PK, split, date_received, quarter, product, issue, sub_issue, company,
dispute_type, narrative, narrative_words, redaction_tokens, company_response, ...),
`gold_labels` (case_id PK/FK, determination, governing_clauses, ambiguous, second rater),
`corpus_chunks` (strategy, chunk_id, section_refs, text), `answers` (split, condition, config,
case_id, run_idx, determination, cited and retrieved references, reasoning, latency, raw response),
`per_case_metrics` and `failure_records` (one row per complaint × condition), `build_info`.
Views: `v_outcomes` (metrics joined to complaint and gold fields) and `v_accuracy` (rates by split
and condition).

## Access

Python:

```python
from dispute_rag.db import query
query("SELECT * FROM v_accuracy WHERE split = ?", params=("dev",))
```

SQL shell: `sqlite3 data/db/qm640.sqlite`, or open the file in DB Browser for SQLite.

Example queries:

```sql
-- Accuracy of Config C by calendar quarter
SELECT quarter, COUNT(*) AS n, AVG(determination_correct) AS accuracy
FROM v_outcomes WHERE split = 'test' AND condition = 'C' GROUP BY quarter;

-- How often each condition answered "escalate" when the gold label was "approve"
SELECT condition, SUM(pred_determination = 'escalate') AS escalated, COUNT(*) AS gold_approve
FROM v_outcomes WHERE gold_determination = 'approve_dispute' GROUP BY condition;

-- Every fabricated citation, with the complaint text, for manual review
SELECT a.condition, a.case_id, m.fabricated_refs, c.narrative
FROM per_case_metrics m JOIN answers a USING (case_id, condition)
JOIN complaints c USING (case_id)
WHERE m.fabricated_citation = 1 AND a.run_idx = 0;
```

The CFPB data are public; narratives are already redacted by the CFPB (`XXXX`). No other personal
data are stored.
