"""SQLite analysis database: one file that holds every record the report is computed from.

The pipeline itself reads and writes flat files (CSV, JSONL) so that each step is inspectable and
diff-able in Git. scripts/00e_build_db.py loads those files into data/db/qm640.sqlite after each
stage, so the data can also be queried with SQL (DB Browser for SQLite, the sqlite3 shell, pandas).
The database is derived and can always be rebuilt; the flat files remain the source of record.

Tables (one row per ...)
  complaints        sampled complaint (dev and test), with split, quarter and derived fields
  gold_labels       labelled complaint: determination, governing clauses, ambiguity, second rater
  corpus_chunks     retrievable passage under each chunking strategy, with its clause references
  answers           model response: complaint x condition x run (raw output retained)
  per_case_metrics  scored first-run outcome: complaint x condition (all pillars)
  failure_records   failure category of each first-run determination
  build_info        provenance of the build (time, source files, row counts)
Views
  v_outcomes        per_case_metrics joined to complaint fields and gold labels
  v_accuracy        accuracy, fabricated and correct-clause rates by split and condition
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/db/qm640.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS complaints (
    case_id TEXT PRIMARY KEY, split TEXT NOT NULL, date_received TEXT, quarter TEXT,
    product TEXT, sub_product TEXT, issue TEXT, sub_issue TEXT, company TEXT, state TEXT,
    dispute_type TEXT, narrative TEXT, narrative_words INTEGER, redaction_tokens INTEGER,
    company_response TEXT, timely_response TEXT);
CREATE TABLE IF NOT EXISTS gold_labels (
    case_id TEXT PRIMARY KEY REFERENCES complaints(case_id), split TEXT NOT NULL,
    determination TEXT NOT NULL, governing_clauses TEXT, ambiguous INTEGER, rationale TEXT,
    labeled_by TEXT, second_rater_determination TEXT, second_rater_clauses TEXT);
CREATE TABLE IF NOT EXISTS corpus_chunks (
    strategy TEXT NOT NULL, chunk_id TEXT NOT NULL, doc_id TEXT, section_ref TEXT,
    section_refs TEXT, n_words INTEGER, text TEXT, PRIMARY KEY (strategy, chunk_id));
CREATE TABLE IF NOT EXISTS answers (
    split TEXT NOT NULL, condition TEXT NOT NULL, config TEXT NOT NULL,
    case_id TEXT NOT NULL REFERENCES complaints(case_id), run_idx INTEGER NOT NULL,
    determination TEXT, cited_section_refs TEXT, retrieved_section_refs TEXT,
    retrieved_chunk_ids TEXT, reasoning TEXT, latency_s REAL, model TEXT, parse_ok INTEGER,
    raw_response TEXT, PRIMARY KEY (split, config, case_id, run_idx));
CREATE INDEX IF NOT EXISTS ix_answers_case ON answers(case_id);
CREATE TABLE IF NOT EXISTS build_info (key TEXT PRIMARY KEY, value TEXT);
"""

VIEWS = """
DROP VIEW IF EXISTS v_outcomes;
CREATE VIEW v_outcomes AS
SELECT m.split, m.condition, m.case_id, c.quarter, c.dispute_type, c.product, c.narrative_words,
       g.determination AS gold_determination, g.ambiguous, m.pred_determination,
       m.determination_correct, m.fabricated_citation, m.correct_clause_cited, m.clause_in_context,
       m.consistent, m.review_required, m.latency_s
FROM per_case_metrics m
JOIN complaints c USING (case_id)
JOIN gold_labels g USING (case_id);
DROP VIEW IF EXISTS v_accuracy;
CREATE VIEW v_accuracy AS
SELECT split, condition, COUNT(*) AS n,
       AVG(determination_correct) AS accuracy,
       AVG(fabricated_citation) AS fabricated_rate,
       AVG(correct_clause_cited) AS correct_clause_rate,
       AVG(review_required) AS review_rate
FROM per_case_metrics GROUP BY split, condition;
"""


def connect(path: Path | str = DB_PATH) -> sqlite3.Connection:
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA foreign_keys = ON")
    return con


def query(sql: str, path: Path | str = DB_PATH, params: tuple = ()) -> pd.DataFrame:
    """Run a read-only query and return a DataFrame, e.g.
    query("SELECT * FROM v_accuracy WHERE split = ?", params=("test",))"""
    with connect(path) as con:
        return pd.read_sql_query(sql, con, params=params)
