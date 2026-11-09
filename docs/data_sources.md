# Data sources and local data requirements

No script downloads data by default. Every step that needs the raw file reads it from one fixed
local path, and every other step runs without it.

## 1. Raw complaint data (required only for sampling and EDA)

| | |
|---|---|
| Expected path | `data/raw/cfpb/complaints.csv` (git-ignored; place it yourself) |
| Fallback | `data/raw/cfpb/complaints.csv.gz`, if it is the real gzip file (the repository's copy is a Git LFS object; `git lfs pull` fetches it) |
| Source | CFPB Consumer Complaint Database narratives archive, FOIA electronic reading room, files `CCDB_Export_7` to `CCDB_Export_21` (August 2024 to August 2026): https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/ |
| Why the archive | The CFPB stopped publishing complaint narratives on 14 August 2026, so the live search export no longer carries them |
| Content | One row per complaint, the standard CCDB columns (Date received, Product, Sub-product, Issue, Sub-issue, Consumer complaint narrative, Company public response, Company, State, ZIP code, Tags, Consumer consent provided?, Submitted via, Date sent to company, Company response to consumer, Timely response?, Consumer disputed?, Complaint ID) |
| Pre-filter (if built with `00a`) | Received 2024-09-18 to 2026-08-14; product in Checking or savings account, Prepaid card, or Money transfer, virtual currency, or money service; non-empty narrative; unique Complaint ID |
| Size used in the study | 1,610,649 rows read by the cleaning funnel (see `data/cases/selection_report.json`) |

Check the file before using it:

```bash
python scripts/00_check_export.py      # prints READY when the three requirements hold
```

To build the file from archive ZIPs you already have, copy them into `data/raw/cfpb/archive/`
and run `python scripts/00a_build_from_archive.py`. Add `--download` only if you want the script to
fetch missing ZIPs from the CFPB.

### Steps that need the raw file

| Step | Script | Without the file |
|---|---|---|
| Validate the export | `00_check_export.py` | exits with instructions |
| Draw the development (pilot) sample | `01_prepare_data.py` | not needed: the pilot is drawn and labelled; the script refuses to redraw it |
| Cleaning log and EDA | `01b_eda.py` | committed outputs in `outputs/eda/` remain valid |
| Draw the test sample | `01c_draw_test_sample.py` | not needed once `data/cases/test/` exists |

Everything else (labels, kappa, model runs, scoring, statistics, figures, database, report) reads
the processed files under `data/cases/`, `data/gold/` and `outputs/`.

## 2. Regulatory corpus (committed)

`data/corpus/reg_e.md` and `data/corpus/reg_e_interp.md`: Regulation E (12 C.F.R. Part 1005) and its
Supplement I, converted from the eCFR XML by `scripts/00b_build_corpus.py` with section markers kept.
The committed files are the ones every result used; the script refuses to overwrite them without
`--force`. Source: https://www.ecfr.gov/current/title-12/chapter-X/part-1005

## 3. Processed data (committed)

| Path | Contents |
|---|---|
| `data/cases/cases.jsonl`, `sample_160.csv`, `selection_report.json` | Development sample (seed 42) and its selection funnel |
| `data/cases/test/` | Test sample (seed 2026), once drawn |
| `data/gold/labeling_sheet.csv`, `gold_cases.jsonl` | Development labels (sheet = source; JSONL = imported) |
| `data/gold/second_rater_sheet.csv` | Blind second-rater sheet for kappa |
| `data/gold/test/` | Test labels and second-rater sheet |
| `data/splits/` | Split manifest and protocol lock |
| `data/validation/validation_sheet.csv` | Week-5 system validation sheet read by `03b_validate.py` |

## 4. Models and services (needed only for new model runs)

| Item | Setting |
|---|---|
| Generator and judge | Anthropic API, `GENERATION_MODEL` (default `claude-sonnet-4-6`), `JUDGE_MODEL`; key in `.env` |
| Dense embeddings | `BAAI/bge-base-en-v1.5` (Hugging Face, downloaded by sentence-transformers on first use; override with `EMBEDDING_MODEL`) |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

All four conditions have already been run on the development set; their outputs are committed under
`outputs/runs/`, so every analysis of them is reproducible offline.

## 5. Provenance facts for the Limitations section

Complaints are published after the company responds or after 15 days; complaints referred to other
regulators are not published; narratives are one-sided and redacted by the CFPB (`XXXX`). The
published set is therefore not a census of disputes.
