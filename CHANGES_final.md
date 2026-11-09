# Changes for the final report (v7-final, 7 Nov 2026)

Builds on interim-v1 (v6 code). Nothing that produced the interim numbers was changed: every
interim result reproduces exactly (C0/A/B/C accuracy 29.6 / 22.4 / 24.8 / 24.8%; strict
correct-clause 76.8 / 42.4 / 44.8 / 44.0%). Default behaviour of every existing script is unchanged
(`--split dev` = the interim paths). Tests: 59 offline tests pass (43 existing + 16 new).

## Evaluation design: development / test split
| File | What |
|---|---|
| `src/dispute_rag/splits.py` | New. Dev (pilot) and test path sets; list of locked files; SHA-256 lock check. |
| `scripts/01c_draw_test_sample.py` | New. Draws the test set: quarter × dispute type, quarter weights by days in window (floor 4), pilot IDs and pilot-identical narratives excluded, seed 2026, default 205 drawn. Writes `data/cases/test/`, `data/gold/test/labeling_sheet.csv`, `data/splits/split_manifest.json`. Refuses to redraw. |
| `scripts/02_import_gold.py` | `--split test`; writes `reliability.json` (kappa, Jaccard); refuses once the protocol is locked. |
| `scripts/02b_lock_protocol.py` | New. Hashes configs, prompt, retrieval, scoring and analysis code, corpus, manifest, test gold and the pre-registration; `--check` verifies. |
| `scripts/03_run_experiment.py` | `--split test` → `outputs/runs_test/`; refuses to call the model unless the lock is intact (`--allow-unlocked` to override, then not confirmatory). |
| `scripts/04_compare_and_business.py` | `--split test` → `outputs/reports_test/` with the test cases and gold. |
| `docs/preregistration_final.md` | New. Fill the `[DECIDE]` items, then lock. |

## Analysis
| File | What |
|---|---|
| `src/dispute_rag/refs.py` | Added `cites_gold_mode` (strict = unchanged primary; parent; section) for the clause-rule sensitivity analysis. |
| `src/dispute_rag/final_analysis.py` | New. Confusion matrices, per-class P/R/F1, macro-F1, majority baseline, needs-human merge (escalate + request more info), unambiguous subset, clause sensitivity, condition summary, pre-specified best-model rule with McNemar vs runner-up, dev-vs-test table. |
| `scripts/05_final_analysis.py` | New. `--split dev|test|both` → `outputs/final/<split>/` tables, JSON and three figures; `both` adds `dev_vs_test.csv` and figure. |

## Prompt (optional)
| File | What |
|---|---|
| `src/dispute_rag/config.py` | `generation.prompt_version: v6` (default) or `v7`. |
| `src/dispute_rag/pipeline.py` | `PROMPTS` dict; v7 replaces rule 5 only, with an explicit decision order that reserves `escalate`. v6 text untouched. Choose before the lock and use it in all four configs. |

## Data storage and access
| File | What |
|---|---|
| `src/dispute_rag/db.py` | New. SQLite schema (complaints, gold_labels, corpus_chunks, answers, per_case_metrics, failure_records, build_info; views v_outcomes, v_accuracy) and `query()`. |
| `scripts/00e_build_db.py` | New. Rebuilds `data/db/qm640.sqlite` from the flat files; fails if dev and test share a complaint. `data/db/` is git-ignored. |
| `docs/data_access.md` | New. Storage layers, schema, example SQL. |

## Report
| File | What |
|---|---|
| `scripts/07_outcome_eda.py` | New. Outcome-linked EDA per split: accuracy by length tercile and by ambiguity (Wilson CI, chi-square), clause in context by dispute type, share answered escalate by gold label. |
| `scripts/08_sample_preview.py` | New. Preview image of sampled complaints with narrative excerpts. |
| `scripts/01c_draw_test_sample.py` (EDA) | Selection report uses the pilot's format so `01b_eda.py` runs on the test sample (`--out outputs/eda_test`); the report then uses those figures. |
| `requirements.txt` | Was empty in the repo; now lists every package the code imports. |
| `scripts/06_pipeline_diagram.py` | New. Six-stage pipeline figure (Data, Preprocessing, EDA, Feature Engineering, Model, Deployment) with legend and storage layer. |
| `scripts/env_report.py` | New. CPU, cores, RAM, GPU, package versions and a ready sentence for the abstract. **Run it on your own machine.** |
| `report/build_final.py` | New. Final-draft mode: rubric-ordered report; all numbers, tables and figures filled from outputs (test split if present, else dev); author prose as highlighted placeholders naming the rubric points. |
| `report/check_final.py` | New. Abstract 300–350 words / one paragraph, placeholders left, table/figure numbering and call-outs, abbreviation first use, reference count and coverage, margins and spacing. |
| `tests/test_final.py` | New. 14 tests. |

## Findings surfaced while building (development set)
- Best model under the pre-specified rule: **C0** (29.6%), statistically indistinguishable from B (McNemar p = .180). All conditions are below the 48.0% majority baseline.
- Needs-human merge: 52.0–55.2% accuracy; no difference from C0 (Holm p = 1.00).
- Unambiguous subset (n = 47): accuracy is *lower* (14.9–19.1%) than on all complaints.
- Clause rule: gap to C0 shrinks from about 33 points (strict) to 8–18 points (parent) and 6–17 (section).


## Repository review (8 Nov 2026)

Full review of the `main` branch as of 25 Oct 2026 plus the changes above. Lint (ruff: pyflakes,
bugbear, syntax) is clean; every offline step was re-run in a clean copy; regenerated development
results match the committed ones (RQ1, RQ2, RQ4 identical to 15 decimal places).

### Defects fixed
| Area | Issue | Fix |
|---|---|---|
| Data safety | `01_prepare_data.py` silently overwrote `data/gold/labeling_sheet.csv` (the 125 pilot labels) and `cases.jsonl` if rerun | Refuses when labels exist; `--force` backs up first |
| Data safety | `00b_build_corpus.py` overwrote the committed corpus every result depends on | Refuses without `--force` |
| Data access | `00a_build_from_archive.py` downloaded from the CFPB on every run | Downloads only with `--download`; refuses to overwrite `complaints.csv` |
| Data access | `01c_draw_test_sample.py` crashed on the Git LFS placeholder `complaints.csv.gz` | One resolver (`cfpb_cases.require_raw_csv`) for all raw-data scripts: uses `data/raw/cfpb/complaints.csv`, falls back to a real `.gz`, rejects LFS stubs with instructions |
| Docs | Scripts pointed to `docs/data_sources.md` and `docs/gold_labeling_protocol.md`, which did not exist | Both written |
| Docs | `README.md` (7 bytes), `requirements.txt`, `.env.example` were empty; `PROJECT_BRIEF.md` was a 1-byte stub | README, requirements and .env.example written; empty brief removed |
| Git | `.gitignore` had no final newline, ignored the LFS-tracked `complaints.csv.gz` but not the uncompressed `complaints.csv`, and did not ignore `.env` | Rewritten: ignores `complaints.csv`, `archive/`, `.env`, `.venv/`, Word lock files, derived database |
| Hygiene | `scripts/00b_build_corpus.py_1` (older duplicate) and `report/~$640_Interim_Report.docx` (Word lock file) committed | Removed |
| Kappa | The blind `second_rater_sheet.csv` was never read, so kappa could not be computed | `02_import_gold.py` merges it; `02a_second_rater_sheet.py` makes the test-set sheet |
| Lint | Unused imports, an unused variable, a duplicate set item, placeholder-free f-strings, a loop-variable closure in `01b_eda.py` | Fixed |

### Not changed (decide yourself)
- `report/QM640_Interim_Report.docx` and `QM640_Interim_Report_o.docx` are both builder outputs and
  neither is byte-identical to the interim you submitted; replace them with the submitted file if the
  repository should hold the graded version.
- `data/validation/` holds four sheets; `03b_validate.py` reads only `validation_sheet.csv`
  (`_filled`, `_M`, `_o` are working copies). Complaints 21070226 and 15018298 still lack
  `expected_failure_category` there.
- The RQ3 driver model needs `statsmodels`; without it the code falls back to a cluster-robust logit and
  records `engine` in `rq3_drivers.json`. The committed results used the GEE engine.
- `01c_draw_test_sample.py` keeps `--end 2026-08-11` (the pilot used 2026-08-14); no complaint falls in
  between, so the candidate pool is identical, and an existing test draw still reproduces exactly.
