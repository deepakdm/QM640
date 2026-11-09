# QM640: Evaluating RAG as decision support for Regulation E payment disputes

Walsh College, QM 640 Data Analytics Capstone, Deepak Dharani.

The study tests whether retrieval-augmented generation (RAG) improves the accuracy and groundedness
of dispute determinations on real Consumer Financial Protection Bureau (CFPB) complaint narratives.
Four conditions are compared on the same labelled complaints: no retrieval (C0), dense retrieval with
512-token chunks (A), dense with 256-token chunks (B) and hybrid BM25 + dense with cross-encoder
re-ranking (C). Nothing is trained; the language model is used through its API with fixed settings.

| Document | Purpose |
|---|---|
| `docs/data_sources.md` | Data sources, the expected local raw file, which steps need it |
| `docs/gold_labeling_protocol.md` | How complaints are labelled and double-rated |
| `docs/preregistration_final.md` | Decisions fixed before the test set is run |
| `docs/data_access.md` | Storage layers, SQLite schema, example queries |
| `CHANGES_final.md` | What changed for the final report, and the repository review |

## Requirements

- Python 3.10 or later (tested on 3.12 and 3.13); `pip install -r requirements.txt`
- For new model runs only: an Anthropic API key in `.env` (copy `.env.example`); the embedding and
  re-ranking models download from Hugging Face on first use
- For sampling and EDA only: the raw file at **`data/raw/cfpb/complaints.csv`** (not downloaded by
  any script; see `docs/data_sources.md`)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then add the API key
pytest tests -q               # offline; no data, key or model downloads needed
```

## Run order

Steps are grouped by what they need. **Data** = raw CSV; **API** = key and network; everything
else runs offline from committed files.

| # | Command | Needs | Output |
|---|---|---|---|
| 0 | `python scripts/00_check_export.py` | Data | READY check |
| 0a | `python scripts/00a_build_from_archive.py` | archive ZIPs (optional) | builds `complaints.csv`; `--download` to fetch |
| 0b | `python scripts/00b_build_corpus.py` | eCFR (done) | corpus is committed; refuses to overwrite |
| 1 | `python scripts/01_prepare_data.py --n 160 --seed 42` | Data (done) | development sample; refuses to redraw once labelled |
| 1b | `python scripts/01b_eda.py` | Data | `outputs/eda/` |
| 2 | `python scripts/02_import_gold.py` | | development gold labels, kappa |
| 3 | `for c in configs/*.yaml; do python scripts/03_run_experiment.py $c; done` | API (done) | `outputs/runs/` |
| 3b | `python scripts/03b_validate.py --config configs/hybrid_rerank.yaml` | | `outputs/validation/` |
| 4 | `python scripts/04_compare_and_business.py` | | `outputs/reports/` (RQ1 to RQ4) |
| 5 | `python scripts/05_final_analysis.py --split dev` | | `outputs/final/dev/` |
| 7 | `python scripts/07_outcome_eda.py --split dev` | | outcome-linked EDA |

Locked test set:

| # | Command | Needs | Output |
|---|---|---|---|
| T1 | `python scripts/01c_draw_test_sample.py --n-draw 290` | Data | `data/cases/test/`, split manifest (drawn once) |
| T2 | label `data/gold/test/labeling_sheet.csv`; `python scripts/02a_second_rater_sheet.py --split test` | | blind second-rater sheet |
| T3 | `python scripts/02_import_gold.py --split test` | | test gold labels, kappa |
| T4 | fill and date `docs/preregistration_final.md`; commit; `python scripts/02b_lock_protocol.py`; `git tag test-protocol-lock` | | protocol lock |
| T5 | `for c in configs/*.yaml; do python scripts/03_run_experiment.py $c --split test; done` | API | `outputs/runs_test/` (refuses if the lock is broken) |
| T6 | `python scripts/04_compare_and_business.py --split test` | | `outputs/reports_test/` |
| T7 | `python scripts/05_final_analysis.py --split both` and `python scripts/07_outcome_eda.py --split both` | | `outputs/final/`, `dev_vs_test.csv` |
| T8 | `python scripts/01b_eda.py --cases data/cases/test/cases.jsonl --selection data/cases/test/selection_report.json --gold data/gold/test/gold_cases.jsonl --out outputs/eda_test` | Data | EDA on the test sample |

Figures, environment, database and report:

```bash
python scripts/06_pipeline_diagram.py      # six-stage pipeline figure
python scripts/08_sample_preview.py        # preview of sampled complaints
python scripts/env_report.py               # CPU / RAM / GPU sentence for the abstract (run on your machine)
python scripts/00e_build_db.py             # SQLite analysis database (git-ignored, rebuildable)
cd report && python build_final.py && python check_final.py QM640_Final_Report_DRAFT.docx
```

Re-running a model step resumes where it stopped. `--score-only` re-scores saved answers without
calling the API.

## Repository layout

| Path | Contents |
|---|---|
| `configs/` | The four conditions (one YAML each) |
| `src/dispute_rag/` | Library: data loading, corpus, retrieval, pipeline, scoring, statistics, splits, database |
| `scripts/` | Numbered pipeline steps |
| `data/` | Corpus, samples, labels, splits, validation sheet (raw file local only) |
| `outputs/` | `eda/`, `runs*/`, `reports*/`, `validation/`, `final/` |
| `report/` | APA report builders (`build_interim.py`, `build_final.py`), checker and built documents |
| `tests/` | Offline tests |

## Reproducibility notes

- Seeds: development sample 42, test sample 2026, second-rater sheet 30, validation sheet 42.
- Generation uses temperature 0 and three runs per complaint; the model string returned by the API is
  stored with every answer.
- The test protocol is fixed by SHA-256 hashes of every file that decides the test numbers
  (`data/splits/protocol_lock.json`).
- All reported development-set numbers can be regenerated offline from the committed `outputs/runs/`.
