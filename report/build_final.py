"""Build the QM 640 Final Report DRAFT (.docx) in the final-report rubric order.

    cd report
    python build_final.py                       # reads ../outputs, ../data; writes QM640_Final_Report_DRAFT.docx
    python build_final.py --primary dev         # before the test set exists: dev results fill the result slots
    python check_final.py QM640_Final_Report_DRAFT.docx

Final-draft mode:
  * every number, table and figure that the analysis produces is filled from the output files
    (outputs/final, outputs/reports, outputs/reports_test, outputs/eda, data/splits), so re-running
    the pipeline and this script refreshes the whole evidence base of the report;
  * design facts that do not depend on results (RQs, hypotheses, variables, data sources, split,
    database, sample-size mathematics) are written out;
  * interpretation and narrative that must be written by the author are highlighted
    [placeholders] that name the rubric criterion and its points, so nothing unverified appears
    as a finding and the remaining work is visible at a glance.

Results slots use the TEST split when outputs exist for it, otherwise the DEV split, and every
table says which one it shows.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from report_lib import APADoc

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=str(HERE.parent))
ap.add_argument("--base", default=str(HERE / "base_style.docx"))
ap.add_argument("--primary", choices=["auto", "dev", "test"], default="auto")
ap.add_argument("--date", default="[[Submission date, e.g., November 24, 2026]]")
ap.add_argument("--toc", default="", help="json list of [level, text, page] from a first render")
ap.add_argument("--out", default=str(HERE / "QM640_Final_Report_DRAFT.docx"))
args = ap.parse_args()
R = Path(args.repo)


# ----------------------------------------------------------------------------------- inputs
def jload(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def csv(p: Path, **kw):
    return pd.read_csv(p, **kw) if p.exists() else None


SPLIT_DIRS = {"dev": ("outputs/reports", "outputs/final/dev", "data/gold", "data/cases"),
              "test": ("outputs/reports_test", "outputs/final/test", "data/gold/test", "data/cases/test")}


def load_split(s: str) -> dict:
    rep, fin, gold, cases = (R / d for d in SPLIT_DIRS[s])
    return {"name": s, "summary": csv(fin / "condition_summary.csv", index_col=0),
            "best": jload(fin / "best_model.json"), "base": jload(fin / "majority_baseline.json"),
            "bestm": csv(fin / "best_by_metric.csv"),
            "collapse": jload(fin / "secondary_needs_human.json"),
            "unamb": jload(fin / "secondary_unambiguous.json"),
            "clause": csv(fin / "sensitivity_clause_match.csv", index_col=0),
            "rq1": jload(rep / "rq1_accuracy.json"), "rq2": jload(rep / "rq2_grounding.json"),
            "rq3": csv(rep / "rq3_drivers_odds_ratios.csv", index_col=0),
            "rq4": jload(rep / "rq4_dominant_failure.json"), "bench": csv(rep / "rq4_benchmarks.csv"),
            "rel": jload(gold / "reliability.json"), "sel": jload(cases / "selection_report.json"),
            "fig_cm": fin / "fig_confusion_matrices.png", "fig_acc": fin / "fig_accuracy_macro_f1.png",
            "fig_clause": fin / "fig_clause_sensitivity.png", "fig_fail": rep / "fig10_failure_modes.png",
            "fig_cost": rep / "fig11_cost_sensitivity.png", "fig_forest": rep / "fig9_driver_forest.png",
            "outcome": jload(fin / "outcome_eda.json"),
            "fig_out_len": fin / "fig_outcome_accuracy_by_length.png",
            "fig_out_amb": fin / "fig_outcome_accuracy_by_ambiguity.png",
            "fig_out_clause": fin / "fig_outcome_clause_by_type.png",
            "fig_out_esc": fin / "fig_outcome_escalation_by_gold.png"}


DEV, TEST = load_split("dev"), load_split("test")
HAS_TEST = TEST["summary"] is not None
P = TEST if (args.primary == "test" or (args.primary == "auto" and HAS_TEST)) else DEV
PLABEL = "locked test set" if P is TEST else "development set (pilot; test set not yet run)"
ENV = jload(R / "outputs/final/environment.json")
MAN = jload(R / "data/splits/split_manifest.json")
DVT = csv(R / "outputs/final/dev_vs_test.csv", index_col=0)
EDA_DIR = R / ("outputs/eda_test" if (R / "outputs/eda_test/eda_summary.json").exists() else "outputs/eda")
EDA = jload(EDA_DIR / "eda_summary.json") or {}


def ph(text: str) -> str:
    return f"[[{text}]]"


def pct(x, d=1) -> str:
    return ph("value") if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{d}f}%"


def num(x, fmt="{:.3f}") -> str:
    return ph("value") if x is None or (isinstance(x, float) and math.isnan(x)) else fmt.format(x)


def pval(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ph("p")
    return "< .001" if p < .001 else f"{p:.3f}".lstrip("0")


def n_of(s: dict):
    return None if s["summary"] is None else int(s["summary"]["n"].iloc[0])


# ----------------------------------------------------------------------------------- numbering
T = {k: i + 1 for i, k in enumerate([
    "vars", "rq", "lit", "incl", "dict", "db", "split", "n", "feat", "cond", "perf", "bestm", "dvt",
    "hyp", "second", "bench"])}
F = {k: i + 1 for i, k in enumerate([
    "pipeline", "preview", "funnel", "length", "mix", "goldco", "out_len", "out_amb", "out_clause", "out_esc",
    "acc", "cm", "clause", "forest", "fail", "cost"])}

d = APADoc(args.base)
TITLE = ("Developing an Evaluation Methodology to Assess Accuracy, Groundedness, and Operational Value "
         "of Retrieval-Augmented Generation (RAG) as a Decision-Support System for Consumer Payment Disputes")

# ================================================================================ title page
d._counting = False
d.blank(2)
d.p("**Data Analytics Capstone**", indent=False, align="center")
d.blank(1)
d.p(f"**{TITLE}**", indent=False, align="center")
d.blank(1)
d.p("**Final Report**", indent=False, align="center")
d.blank(1)
for line in ["Deepak Dharani", "Walsh College", "QM640: Data Analytics Capstone", "Instructor: Javad Katibai",
             "Mentor: Jainesh Garg", "Term 3, Group 3", args.date]:
    d.p(line, indent=False, align="center")
d.page_break()

# ================================================================================ abstract
d.h1("Abstract", toc=False)
nP = n_of(P)
d.p(ph(
    "ABSTRACT, write last, ONE paragraph of 300-350 words (rubric 37 pts). Must state: the problem (5); "
    "the approach (5); data context with N = "
    f"{nP or '?'} complaints, unit = the complaint, window September 18, 2024 to August 11, 2026 (8); "
    "technology incl. CPU/GPU, e.g. the sentence in outputs/final/environment.json (4); concrete results "
    "with numbers from Table " + str(T['perf']) + " and the best model (8); implementation / deployment (3). "
    "Run check_final.py to verify word count and single paragraph."), indent=False)
if ENV:
    d.note("Technology sentence for the abstract: " + ENV.get("sentence_for_report", ""))
d.page_break()

d.p("**GitHub Repository**", indent=False, align="center")
d.p("Data files, code, configurations and outputs: https://github.com/deepakdm/QM640 "
    f"({ph('release tag, e.g., final-v1')}). The storage layout and access methods are described under "
    "Database Setup and Data Access.", indent=False)
d.blank(1)
toc_entries = json.load(open(args.toc)) if args.toc and Path(args.toc).exists() else None
d.toc(toc_entries)
d.page_break()
d._counting = True

# ================================================================================ introduction
d.p(f"**{TITLE}**", indent=False, align="center")
d.h1("Introduction")
d.h2("Background and Context")
d.p(ph("Carry over the three Background paragraphs from the interim report (domain context in 2-4 "
       "paragraphs, 6 pts). Keep the Regulation E deadlines, the 2025 CFPB complaint volumes and the "
       "legal-hallucination evidence; define every abbreviation at first use (RAG, LLM, CFPB, ATM)."))
d.h2("Problem Statement")
d.p(ph("Carry over the interim Problem Statement (one focused paragraph); add one sentence that the "
       "final report tests the method on a locked, quarter-stratified test set."))
d.h2("Variables, Unit of Analysis and Success Metrics")
d.p("The **dependent variables (Y)** are the outcomes of each complaint under each condition: whether "
    "the determination equals the gold determination (primary, binary), whether a cited section is "
    "absent from the regulation (fabricated citation, binary), whether a governing clause is cited "
    "(binary), run-to-run consistency (binary), response time (seconds) and the failure category of an "
    "incorrect determination (nominal). The **independent variable (X)** is the retrieval condition, a "
    "four-level experimental factor: no retrieval (C0), dense retrieval with 512-token passages (A), "
    "dense retrieval with 256-token passages (B) and hybrid lexical-plus-dense retrieval with "
    "re-ranking (C). Case factors (dispute type, narrative length, ambiguity) and the mechanism variable "
    "clause in context enter the driver model for RQ3. The **unit of analysis** is the complaint; the "
    "**timeframe** is complaints received by the Consumer Financial Protection Bureau (CFPB) from September 18, 2024 to August 11, 2026. "
    f"Table {T['vars']} lists the variables and the **success metrics** fixed before scoring.")
d.table(T["vars"], "Variables and Success Metrics",
        ["Role", "Variable", "Type", "Success metric (benchmark)"],
        [["Y, primary", "Determination correct", "Binary", "Gain over C0 of at least 15 points, Holm p < .05"],
         ["Y", "Fabricated citation", "Binary", "Upper 95% Wilson bound at most 5%"],
         ["Y", "Correct clause cited", "Binary", "Higher than C0, Holm p < .05"],
         ["Y", "Consistent across 3 runs", "Binary", "At least 90% of complaints"],
         ["Y", "Response time", "Continuous (s)", "95th percentile at most 30 s"],
         ["Y", "Accuracy vs break-even", "Proportion", "Above modeled break-even accuracy"],
         ["X", "Retrieval condition", "Categorical (4)", "Experimental factor"],
         ["Covariates", "Dispute type, log length, ambiguity", "Categorical / continuous / binary", "RQ3 drivers"],
         ["Mediator", "Clause in context", "Binary", "RQ3 mechanism"]],
        [1500, 2600, 2000, 3260], note="Benchmarks were fixed before any output was scored (interim Table 23).")
d.h2("Stakeholder Value")
d.p(ph("Explain stakeholder value (3 pts): consumers (faster, correct provisional credit), investigators "
       "(drafted, cited determinations against the statutory clock), compliance and model-risk teams "
       "(an evidence-based adoption rule and audit trail), examiners (traceable citations)."))

d.h2("Study Pipeline")
d.p(f"Figure {F['pipeline']} shows the end-to-end pipeline in six stages: data, preprocessing, exploratory "
    "data analysis, feature engineering, model and deployment, with the storage layer beneath them. Box "
    "colour marks who performs each step, so the figure also shows where people review the work.")
d.figure(F["pipeline"], "End-to-End Pipeline From Complaint Archive to Deployment", R / "outputs/final/fig_pipeline.png",
         placeholder="run scripts/06_pipeline_diagram.py", width_in=6.5,
         note="EDA = exploratory data analysis; BM25 = Best Match 25 lexical ranking; T = temperature.")

# ================================================================================ RQs + hypotheses
d.h1("Research Questions and Hypotheses")
d.p("Four research questions organize the study. Each is tested at the 5% significance level "
    f"(α = .05); comparisons with the baseline are Holm-corrected over the three retrieval conditions. "
    f"Table {T['rq']} states each question with its null (H₀) and alternative (Hₐ) hypotheses and test.")
d.table(T["rq"], "Research Questions, Hypotheses and Tests",
        ["RQ", "Question", "H₀ and Hₐ", "Test"],
        [["RQ1", "To what extent does retrieval improve determination accuracy over the same LLM without retrieval?",
          "H1₀: p(correct | retrieval k) = p(correct | C0); H1ₐ: they differ", "Exact McNemar × 3, Holm; Cochran's Q"],
         ["RQ2", "Does retrieval reduce fabricated citations and increase citation of the governing clause?",
          "H2₀: fabricated and correct-clause rates equal C0's; H2ₐ: they differ", "Exact McNemar × 3 per outcome, Holm"],
         ["RQ3", "Which system and case factors explain a correct determination, and does the clause in context mediate the effect?",
          "H3₀: OR(clause in context) = 1; H3ₐ: OR ≠ 1", "GEE logistic regression, clustered by complaint"],
         ["RQ4", "Does Config C meet the adoption benchmarks, and which failure mode dominates?",
          "H4₀: dominant failure share ≤ .20; H4ₐ: share > .20", "One-sided exact binomial; interval benchmarks"]],
        [600, 3500, 3300, 1960], note="OR = odds ratio; GEE = generalized estimating equations.")

# ================================================================================ literature
d.h1("Literature Review")
d.h2("Search Approach")
d.p(ph("Carry over the interim Search Approach; add the search for deep-learning retrieval and "
       "agentic/LLM work and the date of the final search."))
d.h2("Literature Matrix")
d.p(f"Table {T['lit']} compares the sources by author, domain, dataset, method, findings and the research "
    "question each one informs, and the decision it supports in this study.")
LIT = [
    ["Lewis et al. (2020)", "Open-domain NLP", "NQ, TriviaQA, FEVER; Wikipedia", "Dense retriever + seq2seq generator (DL)", "Retrieval beat parametric-only models", "RQ1: defines RAG and the no-retrieval baseline C0"],
    ["Karpukhin et al. (2020)", "Open-domain QA", "Five QA benchmarks", "Dual-encoder dense retrieval (DL)", "Beat BM25 by 9-19 points top-20", "RQ3: dense retriever of A and B"],
    ["Robertson & Zaragoza (2009)", "Information retrieval", "Theory and review", "BM25 probabilistic relevance", "Robust lexical matching", "RQ3: lexical half of C; section numbers"],
    ["Nogueira & Cho (2019)", "Passage ranking", "MS MARCO, TREC-CAR", "BERT cross-encoder re-ranking (DL)", "27% relative MRR@10 gain", "RQ3: re-ranking step of C"],
    ["Shuster et al. (2021)", "Dialogue", "Wizard of Wikipedia, CMU_DoG", "Retrieval-augmented dialogue (DL)", "Retrieval reduced hallucination", "RQ2: direction of hypothesis"],
    ["Y. Gao et al. (2023)", "RAG survey", "Literature to 2023", "Taxonomy of RAG", "Chunking, hybrid, re-ranking are main levers", "RQ3: choice of configuration factors"],
    ["Liu et al. (2024)", "Long-context LLMs", "Multi-document QA", "Vary context position and size", "Mid-context evidence is lost", "RQ1, RQ3: context budget of B; two-sided tests"],
    ["Barnett et al. (2024)", "RAG engineering", "Three deployed systems", "Case studies", "Seven failure points", "RQ4: failure taxonomy"],
    ["T. Gao et al. (2023)", "Cited generation", "ALCE benchmark", "Citation recall and precision", "Half of citations unsupported on ELI5", "RQ2: citation-level measures"],
    ["Es et al. (2024)", "RAG evaluation", "WikiEval", "LLM-prompted metrics (RAGAS)", "Tracked human judgments", "RQ2: secondary judged support"],
    ["Saad-Falcon et al. (2024)", "RAG evaluation", "KILT, SuperGLUE, AIS", "Fine-tuned LM judges + PPI", "Few hundred labels calibrate a judge", "RQ2: judge validation design"],
    ["Zheng et al. (2023)", "LLM-as-a-judge", "MT-Bench, Chatbot Arena", "GPT-4 judge vs humans", ">80% agreement, known biases", "RQ2: judge kept secondary"],
    ["Dahl et al. (2024)", "Legal LLMs", "U.S. federal case questions", "Hallucination profiling", "58-88% hallucination", "RQ2: motivation"],
    ["Magesh et al. (2025)", "Legal RAG tools", "200+ pre-registered queries", "Commercial RAG vs GPT-4", "17-33% hallucination; ~65% accuracy", "RQ1, RQ2: planning values for sample size"],
    ["Islam et al. (2023)", "Financial QA", "FinanceBench", "16 configurations, manual review", "81% failed or refused", "RQ1, RQ3: retrieval design decides finance outcomes"],
    ["Yao et al. (2023)", "Agentic LLMs", "HotpotQA, FEVER, ALFWorld, WebShop", "Interleaved reasoning and tool actions (ReAct prompting)", "Wikipedia look-ups reduced hallucination vs chain-of-thought; +34 and +10 points success on ALFWorld, WebShop", "RQ4, future scope: agentic retrieval vs this study's single-pass drafting"],
    ["Asai et al. (2024)", "RAG with self-reflection", "Open-domain QA, reasoning, fact verification, long-form generation", "LM trained to retrieve on demand and critique its output with reflection tokens (DL, 7B/13B)", "Beat ChatGPT and retrieval-augmented Llama2-chat; better factuality and citation accuracy", "RQ2, RQ3: adaptive retrieval vs fixed top-k; citation quality"],
]
d.table(T["lit"], "Literature Matrix", ["Author", "Domain", "Dataset", "Method", "Findings", "RQ Linkage"],
        LIT, [1500, 1250, 1600, 1700, 1650, 1660], size=8,
        note="DL = deep learning; QA = question answering; MRR = mean reciprocal rank; PPI = prediction-powered inference.")
d.h2("Synthesis of Prior Work")
d.p(ph("Carry over the four interim synthesis paragraphs (3-6 paragraphs, 6 pts). Add one sentence each on "
       "the deep-learning basis of dense retrieval and cross-encoders (3 pts) and on agentic/LLM work (3 pts). "
       "End each paragraph with how the sources shaped a project decision (5 pts)."))
d.h2("Research Gap")
d.p(ph("Carry over the interim Research Gap and Table 5 (coverage of pillars)."))

# ================================================================================ data
d.h1("Data")
d.h2("Data Source and Acquisition")
d.p("The cases are complaints from the CFPB Consumer Complaint "
    "Database (https://www.consumerfinance.gov/data-research/consumer-complaints/). Because the CFPB stopped "
    "publishing narratives on August 14, 2026, the narratives were obtained from the CFPB narratives archive "
    "in its Freedom of Information Act reading room (files CCDB_Export_7 to CCDB_Export_21), merged by "
    "scripts/00a_build_from_archive.py into data/raw/cfpb/complaints.csv.gz. The regulatory corpus is "
    "Regulation E (12 C.F.R. Part 1005) and its Supplement I, downloaded as XML from the Electronic Code of "
    "Federal Regulations (https://www.ecfr.gov/current/title-12/chapter-X/part-1005) and converted by "
    "scripts/00b_build_corpus.py with section markers preserved. No scraping was required. "
    f"Figure {F['preview']} shows sampled complaints as they appear in the data.")
d.figure(F["preview"], "Preview of Sampled Complaints", R / "outputs/final/fig_sample_preview.png",
         placeholder="run scripts/08_sample_preview.py", width_in=6.5,
         note="Eight complaints from the development sample, at least two per dispute type, chosen with a fixed seed.")
d.h2("Inclusion and Exclusion Criteria")
d.table(T["incl"], "Inclusion and Exclusion Criteria",
        ["Criterion", "Included", "Excluded", "Reason"],
        [["Narrative", "Published consumer narrative", "No narrative", "Narrative is the model input"],
         ["Window", "Received Sep 18, 2024 to Aug 11, 2026", "Outside window", "Archive coverage"],
         ["Product", "Checking or savings; prepaid card; money transfer, virtual currency or money service", "All other products (mortgage, credit reporting, credit cards)", "Governed by Regulation E"],
         ["Dispute type", "Unauthorized transaction; incorrect amount; error resolution", "Issues mapping to no dispute type", "Must be a payment dispute"],
         ["Length", "40 to 600 words", "Shorter or longer", "Decidable facts; predictable prompt size"],
         ["Duplicates", "First occurrence", "Repeated IDs; identical narratives", "Each case counts once"],
         ["Rater scope check", "Dispute under Regulation E", "Found out of scope while labelling", "Valid ground truth"],
         ["Test set only", "Not in the pilot", "Pilot IDs; narratives identical to a pilot narrative", "No leakage from development"]],
        [1500, 2900, 2700, 2260])
d.h2("Data Dictionary")
d.p(f"Table {T['dict']} defines every variable used in the analysis.")
d.table(T["dict"], "Data Dictionary", ["Variable", "Definition", "Data type", "Allowed values", "Missing values", "Notes"],
        [[ph("Carry over interim Tables 7 and 8 (8 pts); add split and quarter"), "", "", "", "", ""]],
        [1500, 2200, 1100, 1600, 1600, 1360], size=8, group_rows=(0,))
d.h2("Database Setup")
d.p("The pipeline stores each stage as flat files (CSV and JSON Lines) under version control, so every "
    "intermediate result can be inspected and reproduced. After each stage, scripts/00e_build_db.py loads "
    "those files into a single SQLite 3 database, data/db/qm640.sqlite, with the tables in Table "
    f"{T['db']}, foreign keys from labels and answers to complaints, and two analysis views. The database is "
    "derived: it is rebuilt from the flat files in under a minute and is never edited by hand.")
d.table(T["db"], "Analysis Database Tables",
        ["Table / view", "One row per", "Key fields"],
        [["complaints", "Sampled complaint (dev and test)", "case_id (PK), split, quarter, product, dispute_type, narrative"],
         ["gold_labels", "Labelled complaint", "case_id (PK, FK), determination, governing_clauses, ambiguous"],
         ["corpus_chunks", "Passage per chunking strategy", "strategy, chunk_id (PK), section_refs, text"],
         ["answers", "Model response", "split, config, case_id (FK), run_idx (PK), determination, citations, latency"],
         ["per_case_metrics", "Complaint × condition", "determination_correct, fabricated_citation, correct_clause_cited"],
         ["failure_records", "Complaint × condition", "failure category"],
         ["v_outcomes, v_accuracy", "View", "Metrics joined to complaint and gold fields; rates by split and condition"]],
        [2000, 2600, 4760], note="PK = primary key; FK = foreign key. Schema in src/dispute_rag/db.py.")
d.h2("Data Access")
d.p("Data are accessed in three ways: the numbered scripts read the flat files directly; analysts query "
    "the SQLite database from Python (dispute_rag.db.query) or any SQLite client; and the public GitHub "
    "repository provides the raw archive, corpus, samples, labels and outputs. docs/data_access.md gives "
    "example queries. The model is accessed through the provider's application programming interface (API) "
    "with the key held in a local environment file that is not committed.")
d.h2("Data Splitting")
dev_n = len(MAN["dev"]["case_ids"]) if MAN else 160
test_n = len(MAN["test"]["case_ids"]) if MAN else None
d.p("No model is trained, so the split does not separate training from testing data. It separates the "
    "complaints used to make method decisions from those used to report results. The 160 sampled pilot "
    f"complaints (125 labelled in scope) form the **development set**: the prompt, the clause-matching rule, "
    "the secondary analyses and the best-model rule were decided while viewing their results. A new "
    f"**test set** of {test_n or ph('n drawn')} complaints was then drawn by scripts/01c_draw_test_sample.py with seed "
    "2026, stratified by calendar quarter and dispute type, excluding every pilot complaint and every "
    "narrative identical to a pilot narrative. Its labels were frozen and the protocol locked by SHA-256 hashes "
    "(scripts/02b_lock_protocol.py, docs/preregistration_final.md) before any model output on it existed. "
    f"Table {T['split']} summarises the two sets; hypothesis decisions use the test set only.")
rows = [["Development (pilot)", "Seed 42, dispute type", str(dev_n), str(n_of(DEV) or ph("n")),
         "Method decisions; development results"],
        ["Test (locked)", "Seed 2026, quarter × dispute type", str(test_n or ph("n")), str(n_of(TEST) or ph("n")),
         "Confirmatory results"]]
d.table(T["split"], "Development and Test Sets", ["Set", "Sampling", "Drawn", "Labelled in scope", "Use"],
        rows, [2000, 2500, 1000, 1500, 2360],
        note="Membership and hashes: data/splits/split_manifest.json and protocol_lock.json.")
d.h2("Exploratory Data Analysis")
d.p(ph("Carry over the interim EDA text (Tables 9-13); recompute on the test sample and add one sentence "
       "per figure on what it shows and which RQ it informs (5 pts understanding, 5 pts visuals)."))
d.p(f"Figures {F['funnel']} to {F['goldco']} summarise the cleaning funnel (Figure {F['funnel']}), narrative "
    f"length (Figure {F['length']}), dispute-type mix (Figure {F['mix']}) and gold labels against the company's "
    f"response (Figure {F['goldco']}).")
for key, name, title in (("funnel", "fig1_cleaning_funnel.png", "Complaints Remaining After Each Cleaning Step"),
                         ("length", "fig2_narrative_length.png", "Distribution of Narrative Length (Log Scale)"),
                         ("mix", "fig5_dispute_type_mix.png", "Dispute-Type Mix in the Pool and the Samples"),
                         ("goldco", "fig6_gold_vs_company.png", "Gold Determination Against the Company's Response")):
    d.figure(F[key], title, EDA_DIR / name, placeholder=f"run scripts/01b_eda.py ({name})")

d.h3("Outcome-Linked Exploratory Analysis")
OE = P["outcome"]
def _p(key, c):
    v = (OE or {}).get(key, {}).get(c)
    return pval(v["p_value"]) if v else ph("p")
if OE:
    cuts = OE["length_tercile_cutpoints_words"]
    d.p(f"Figures {F['out_len']} to {F['out_esc']} link case characteristics to model outcomes on the {PLABEL}. "
        f"Accuracy by narrative-length tercile (cut points {cuts[0]:.0f} and {cuts[1]:.0f} words; Figure "
        f"{F['out_len']}) differs by length in C0 (χ² p = {_p('chi2_accuracy_by_length', 'C0')}), A "
        f"(p = {_p('chi2_accuracy_by_length', 'A')}) and B (p = {_p('chi2_accuracy_by_length', 'B')}) but not C "
        f"(p = {_p('chi2_accuracy_by_length', 'C')}). Figure {F['out_amb']} compares complaints the rater flagged as "
        f"ambiguous with the rest, Figure {F['out_clause']} shows how often the governing clause reached the model "
        f"by dispute type, and Figure {F['out_esc']} shows how often each condition answered escalate for each gold "
        f"determination. " + ph("Interpret (EDA insights): what each pattern means for RQ1-RQ4; descriptive only, "
                                "not causal; the GEE model is the confirmatory test."))
else:
    d.p(ph("run scripts/07_outcome_eda.py"))
for key, fk, title in (("out_len", "fig_out_len", "Accuracy by Narrative-Length Tercile"),
                       ("out_amb", "fig_out_amb", "Accuracy by Rater-Flagged Ambiguity"),
                       ("out_clause", "fig_out_clause", "Governing Clause in Retrieved Context by Dispute Type"),
                       ("out_esc", "fig_out_esc", "Share Answered Escalate by Gold Determination")):
    d.figure(F[key], title, P[fk], placeholder="run scripts/07_outcome_eda.py",
             note="First of three runs; bars show 95% Wilson intervals.")

# ================================================================================ sample size
d.h1("Sample Size")
za, zb, psi, delta = 2.394, 0.842, 0.30, 0.15
n_req = (za * math.sqrt(psi) + zb * math.sqrt(psi - delta**2)) ** 2 / delta**2
def power(n):
    from statistics import NormalDist
    nd = NormalDist()
    return nd.cdf((delta * math.sqrt(n) - za * math.sqrt(psi)) / math.sqrt(psi - delta**2))
nt = n_of(TEST)
d.p("RQ1 and RQ2 compare paired proportions, so the number of complaints follows Connor (1987): "
    "n = [z₁₋α/2·√ψ + z₁₋β·√(ψ − δ²)]² / δ², where ψ is the probability that a pair is discordant and δ the "
    f"smallest important difference. With α = .05/3 for three Holm-corrected comparisons (z = {za}), power "
    f".80 (z = {zb}), ψ = {psi} and δ = {delta}: n = ({za} × {math.sqrt(psi):.3f} + {zb} × "
    f"{math.sqrt(psi - delta**2):.3f})² / {delta**2:.4f} = {n_req:.1f}, rounded up to {math.ceil(n_req)}. "
    "RQ4 needs 43 errors to estimate a failure share within ±.15 (n = 1.96² × .25 / .15² = 42.7), which at the "
    "planning error rate of .30 is 144 complaints, and RQ3 needs 152 after the design effect for three "
    "correlated observations per complaint (1 + 2 × .40 = 1.8; Hsieh et al., 1998). The study therefore "
    f"needed at least 152 labelled complaints. The development set reached 125 (power {power(125):.2f}); "
    f"the test set {'reached ' + str(nt) + f' (power {power(nt):.2f})' if nt else 'targets 160 labelled from 205 drawn (power ' + f'{power(160):.2f})'}"
    ", which justifies the volume of data collected relative to the research questions. "
    f"Table {T['n']} summarises the calculation.")
d.table(T["n"], "Minimum Sample Size by Research Question", ["RQ", "Method", "Key parameters", "Minimum N"],
        [["RQ1", "Power, paired proportions (Connor, 1987)", "α = .05/3, power = .80, δ = .15, ψ = .30", str(math.ceil(n_req))],
         ["RQ2", "Same as RQ1", "Same", str(math.ceil(n_req))],
         ["RQ3", "Logistic regression (Hsieh et al., 1998) × design effect", "P₁ = .55, P₂ = .75, B = .75, ICC = .40, m = 3", "152"],
         ["RQ4", "CI for a proportion", "z = 1.96, e = .15, p = .50, error rate = .30", "144"],
         ["Required", "Maximum of the four", "", "152"],
         ["Achieved", "Development / test", "", f"125 / {nt or ph('n')}"]],
        [800, 3200, 3800, 1560])

# ================================================================================ methodology
d.h1("Methodology")
d.h2("Data Ingestion")
d.p(f"The methodology follows the six stages of Figure {F['pipeline']}. Ingestion reads the merged archive in chunks of 200,000 rows (src/dispute_rag/cfpb_cases.py), "
    "resolves column names against the CFPB codebook, parses dates and applies the scope filters while "
    "streaming, so the 1.6-million-row file never has to fit in memory. Every count is written to "
    "selection_report.json. The corpus is ingested from eCFR XML into Markdown with its section markers.")
d.h2("Preprocessing")
d.p("Preprocessing applies the filters in Table " + str(T['incl']) + " in a fixed order, maps issue and "
    "sub-issue to a dispute type by ordered text rules (categorical encoding into three levels), removes "
    "duplicate IDs and identical narratives, and draws the stratified samples. Narratives are whitespace-"
    "normalized; redaction tokens (XXXX) are kept and the model is told to treat them as unknown facts. "
    "**Scaling:** no numeric scaling is applied to model inputs, because the language model reads text and "
    "embeddings are unit-normalized by construction; the only scaled variable is narrative length, which "
    "is log-transformed and centred for the driver model. **Encoding:** dispute type and condition are "
    "dummy-coded in the driver model (references: incorrect amount and Config A).")
d.h2("Missing Values")
d.p(ph("Carry over the missing-value row of interim Table 10 (6 pts): sub-issue 6.3% blank mapped from issue "
       "alone; company public response, tags and state not used; no imputation; unparseable model output "
       "scored as incorrect; failed API calls re-run, never scored."))
d.h2("Outliers")
d.p(ph("Carry over the outlier row of interim Table 10 (6 pts): pre-set 40/600-word limits versus the IQR "
       "fence of 401 words; why limits were fixed before seeing the data; latency outliers kept and "
       "summarised by the 95th percentile."))
d.h2("Feature Engineering")
d.p(f"Table {T['feat']} lists the inputs the system receives and the predictors of the driver model.")
d.table(T["feat"], "Feature Set and Usage", ["Feature", "Original / engineered", "Type", "Reason", "Used in"],
        [[ph("Carry over interim Table 15 and its leakage paragraph"), "", "", "", ""]], [1800, 1800, 1500, 2600, 1660], size=8, group_rows=(0,))
d.h2("Model Selection and Justification")
d.p(ph("Carry over interim 'Choice of Models with Justification' and 'Generator model selection' (7 pts); "
       "keep the alternatives considered and why ROC-AUC is replaced by confusion matrices."))
d.h2("Model Development")
d.p(f"Table {T['cond']} gives the settings of the four conditions. " + ph(
    "State the prompt version used on the test set (v6 or v7, from docs/preregistration_final.md) and the pre-run fixes."))
d.table(T["cond"], "Experimental Conditions and Settings", ["Setting", "C0", "A", "B", "C"],
        [[ph("Carry over interim Table 14"), "", "", "", ""]], [2400, 1700, 1700, 1700, 1860], size=8, group_rows=(0,))

# ================================================================================ results
d.h1("Results")
d.p(f"Results tables show the {PLABEL} unless they say otherwise. Development and test results are shown "
    f"side by side in Table {T['dvt']}.")
d.h2("Performance Comparison")
d.p(f"Table {T['perf']} compares the four conditions on every pillar, Figure {F['acc']} shows accuracy and "
    f"macro-F1 against the majority-class baseline, and Figure {F['cm']} gives the confusion matrices.")
S = P["summary"]
if S is not None:
    rows = [[c, str(int(r.n)), f"{pct(r.accuracy)} [{100*r.accuracy_lo:.1f}, {100*r.accuracy_hi:.1f}]",
             num(r.macro_f1), pct(r.fabricated_rate), pct(r.correct_clause_rate),
             "—" if c == "C0" else pct(r.clause_in_context_rate), pct(r.consistent_rate), f"{r.latency_p95_s:.1f}"]
            for c, r in S.iterrows()]
    base = P["base"]
    rows.append([f"Majority class ({base['label'].replace('_', ' ')})", str(base["n"]), pct(base["accuracy"]),
                 "—", "—", "—", "—", "—", "—"])
else:
    rows = [[ph("run 05_final_analysis.py")] + [""] * 8]
d.table(T["perf"], f"Model Performance Comparison ({PLABEL.split(' (')[0]})",
        ["Condition", "n", "Accuracy [95% CI]", "Macro-F1", "Fabricated", "Correct clause", "Clause in context",
         "Consistent", "p95 s"], rows, [1500, 500, 1700, 900, 1000, 1000, 1000, 1000, 760], size=8,
        note="Accuracy and citation rates use the first of three runs; consistency uses all three. CI = Wilson "
             "interval; p95 = 95th-percentile response time. Source: outputs/final/<split>/condition_summary.csv.")
d.figure(F["acc"], "Accuracy and Macro-F1 by Condition", P["fig_acc"], placeholder="run 05_final_analysis.py",
         note="The dashed line is the accuracy of always predicting the most common gold label.")
d.figure(F["cm"], "Confusion Matrices by Condition", P["fig_cm"], placeholder="run 05_final_analysis.py",
         note="Rows are gold determinations, columns the system's first-run determination.")
d.h2("Best Model")
best = P["best"]
d.p((best.get("statement", "") + " " if best else ph("run 05_final_analysis.py")) +
    "The ranking rule, fixed on the development set, orders conditions by accuracy, then macro-F1, then the "
    "fabricated-citation rate, then the 95th-percentile response time. " +
    ph("Interpret (7 pts): name the best model clearly and say whether it is good enough to deploy, "
       "comparing it with the majority-class baseline and the break-even accuracy."))
BM = P["bestm"]
d.p(f"Table {T['bestm']} names the best condition on each metric.")
d.table(T["bestm"], "Best Condition by Metric",
        ["Metric", "Better when", "Best condition", "Best value", "Runner-up value"],
        ([[r.metric, r.direction, r.best,
           (pct(r.best_value) if "rate" in r.column or r.column == "accuracy" else
            f"{r.best_value:.1f}" if "latency" in r.column else f"{r.best_value:.3f}"),
           (pct(r.runner_up_value) if "rate" in r.column or r.column == "accuracy" else
            f"{r.runner_up_value:.1f}" if "latency" in r.column else f"{r.runner_up_value:.3f}")]
          for r in BM.itertuples()] if BM is not None else [[ph("run 05_final_analysis.py"), "", "", "", ""]]),
        [3000, 1300, 1700, 1600, 1760], size=8,
        note="Ties are listed with an equals sign. The overall best model follows the ranking rule above.")
d.h2("Development Versus Test Results")
if DVT is not None:
    rows = [[c, f"{int(r.n_dev)} / {int(r.n_test)}", f"{pct(r.accuracy_dev)} / {pct(r.accuracy_test)}",
             f"{100 * r.accuracy_diff:+.1f}", f"{r.macro_f1_dev:.3f} / {r.macro_f1_test:.3f}",
             f"{pct(r.fabricated_rate_dev)} / {pct(r.fabricated_rate_test)}",
             f"{pct(r.correct_clause_rate_dev)} / {pct(r.correct_clause_rate_test)}"] for c, r in DVT.iterrows()]
else:
    rows = [[ph("available after the test runs: python scripts/05_final_analysis.py --split both")] + [""] * 6]
d.table(T["dvt"], "Separated Evaluation Results: Development and Test Sets",
        ["Condition", "n dev / test", "Accuracy dev / test", "Diff (points)", "Macro-F1 dev / test",
         "Fabricated dev / test", "Correct clause dev / test"], rows, [1100, 1200, 1700, 900, 1500, 1500, 1460], size=8,
        group_rows=() if DVT is not None else (0,),
        note="Equivalent of training and test accuracy for a system that is not trained. Source: outputs/final/dev_vs_test.csv.")
d.h2("Hypothesis Tests")
d.p(f"Table {T['hyp']} gives the decision on each null hypothesis at the 5% level.")
hy = []
for c in ("A", "B", "C"):
    a = (P["rq1"] or {}).get("vs_control", {}).get(c)
    hy.append([f"H1: {c} vs C0 accuracy", f"{100*a['difference']:+.1f} pts" if a else ph("diff"),
               pval(a["p_holm"]) if a else ph("p"),
               ("Reject H₀" if a and a["p_holm"] < .05 else "Fail to reject H₀") if a else ph("decision")])
for c in ("A", "B", "C"):
    for key, lab in (("fabricated_citation", "fabricated"), ("correct_clause_cited", "correct clause")):
        a = (P["rq2"] or {}).get(key, {}).get("vs_control", {}).get(c)
        hy.append([f"H2: {c} vs C0 {lab}", f"{100*a['difference']:+.1f} pts" if a else ph("diff"),
                   pval(a["p_holm"]) if a else ph("p"),
                   ("Reject H₀" if a and a["p_holm"] < .05 else "Fail to reject H₀") if a else ph("decision")])
r3 = P["rq3"]
if r3 is not None and "clause_in_context" in r3.index:
    x = r3.loc["clause_in_context"]
    hy.append(["H3: clause in context", f"OR = {x.odds_ratio:.2f} [{x.or_ci_low:.2f}, {x.or_ci_high:.2f}]",
               pval(x.p_value), "Reject H₀" if x.p_value < .05 else "Fail to reject H₀"])
else:
    hy.append(["H3: clause in context", ph("OR"), ph("p"), ph("decision")])
r4 = P["rq4"]
hy.append(["H4: dominant failure share > .20",
           f"{r4['category'].replace('_', ' ')} {pct(r4['share'])}" if r4 else ph("share"),
           pval(r4["p_one_sided_vs_0.20"]) if r4 else ph("p"),
           ("Reject H₀" if r4 and r4["p_one_sided_vs_0.20"] < .05 else "Fail to reject H₀") if r4 else ph("decision")])
q = (P["rq1"] or {}).get("cochran_q")
d.table(T["hyp"], "Hypothesis Tests at the 5% Significance Level", ["Hypothesis", "Estimate", "*p*", "Decision (α = .05)"],
        hy, [3400, 2600, 1100, 2260], size=8,
        note="H1 and H2: exact McNemar tests, Holm-adjusted over three comparisons. H3: GEE logistic regression. "
             "H4: one-sided exact binomial test. " + (f"Cochran's Q across four conditions: Q({q['df']}) = {q['Q']:.2f}, p = {pval(q['p_value'])}." if q else ""))
d.p(ph("Interpret each decision in words (2 pts for notation and logic): what rejecting or failing to reject "
       "means for each RQ, with the effect size and interval, not only the p value."))
d.h2("Secondary and Sensitivity Analyses")
d.p(f"Table {T['second']} and Figure {F['clause']} report the pre-specified secondary analyses; Figure "
    f"{F['forest']} and Figure {F['fail']} show the drivers of a correct determination and the failure modes.")
sec = []
if P["collapse"]:
    for c, v in P["collapse"]["per_condition"].items():
        u = (P["unamb"] or {}).get("per_condition", {}).get(c, {})
        cl = P["clause"].loc[c] if P["clause"] is not None and c in P["clause"].index else None
        sec.append([c, pct(v["rate"]), pct(u.get("rate")),
                    *(pct(cl[m]) if cl is not None else ph("v") for m in ("strict", "parent", "section"))])
else:
    sec = [[ph("run 05_final_analysis.py")] + [""] * 5]
d.table(T["second"], "Secondary and Sensitivity Analyses",
        ["Condition", "Accuracy, needs-human merged", "Accuracy, unambiguous only", "Clause: strict", "Clause: parent", "Clause: section"],
        sec, [1300, 1900, 1900, 1400, 1400, 1460], size=8,
        note="Needs-human merges escalate and request more information. Clause rules: strict (primary) credits the "
             "gold clause or a more specific one; parent credits either direction; section ignores paragraphs.")
d.figure(F["clause"], "Correct-Clause Rate Under Three Matching Rules", P["fig_clause"], placeholder="run 05_final_analysis.py")
d.figure(F["forest"], "Drivers of a Correct Determination: GEE Odds Ratios", P["fig_forest"], placeholder="run 04_compare_and_business.py")
d.figure(F["fail"], "Failure Modes of Incorrect Determinations by Condition", P["fig_fail"], placeholder="run 04_compare_and_business.py")
d.h2("Interpretation")
d.p(ph("Interpret the metrics and visuals in writing (8 pts): accuracy against the majority baseline; the "
       "escalate column of the confusion matrices; what the clause sensitivity shows; whether test results "
       "replicate development results; what the GEE drivers say. No causal claims for case factors."))

# ================================================================================ deployment
d.h1("Deployment and Application")
d.p(f"Table {T['bench']} checks Config C against the adoption benchmarks, and Figure {F['cost']} shows the "
    "modeled handling-time saving at each accuracy level.")
B = P["bench"]
if B is not None:
    d.table(T["bench"], "Adoption Benchmarks for Config C", ["Pillar", "Indicator", "Estimate", "Benchmark", "Verdict"],
            B.astype(str).values.tolist(), [1300, 2300, 2300, 2200, 1260], size=8,
            note="Verdicts follow the interval rules fixed before scoring.")
d.figure(F["cost"], "Modeled Handling-Time Saving by Assistant Accuracy", P["fig_cost"], placeholder="run 04_compare_and_business.py",
         note="Model under stated assumptions, not observation.")
d.h2("Deployment Approach")
d.p(ph("Describe the deployment approach (8 pts): decision support only; batch or on-demand drafting per "
       "new dispute; pinned model version and corpus snapshot; shadow mode first (drafts not shown to the "
       "customer) with acceptance rate tracked; go/no-go tied to the benchmarks in the table above."))
d.h2("System Integration")
d.p(ph("Explain system integration (7 pts): case-management system sends the complaint text and issue "
       "fields; the service returns JSON (determination, reasoning, cited sections); citations are checked "
       "against the corpus index before display; every draft and reviewer action is logged to the audit store "
       "(the answers table shows the record kept per draft); corpus refreshed when eCFR amends Part 1005."))
d.h2("Benefits to Users")
d.p(ph("Explain user benefits (7 pts), tied to measured results only: consistency, latency, citation "
       "traceability; state plainly where the results do NOT yet support a benefit (accuracy)."))
d.h2("Example Use Case")
d.p(ph("Example use case (8 pts): walk through one real complaint from the test set (query the database: "
       "SELECT * FROM v_outcomes WHERE condition='C' AND determination_correct=1) showing the redacted "
       "narrative excerpt, retrieved clauses, the drafted determination with citations, the review routing "
       "and what the investigator does. Use one correct case and, briefly, one over-escalated case."))

# ================================================================================ limitations etc.
d.h1("Limitations")
d.p(ph("Data limitations (3 pts), model limitations (3 pts), impact on findings and generalizability (2 pts). "
       "Carry over interim Table 24; update: second-rater kappa (" +
       (f"κ = {P['rel']['cohen_kappa']:.2f}" if P["rel"] and P["rel"].get("cohen_kappa") is not None else "pending") +
       "), single generator model, AI-assisted gold drafts, one-sided redacted narratives."))
d.h1("Future Scope")
d.p(ph("Future scope (2 pts): practitioner study with investigators; newer-model replication; confidence "
       "scores to allow ROC analysis; agentic retrieval; organizational case files under ethics approval."))
d.h1("Conclusion")
d.p(ph("Answer each RQ in one or two sentences and give the adoption recommendation (go / no-go / conditions)."))

# ================================================================================ references
d.page_break()
d.h1("References")
REFS = [
    "Asai, A., Wu, Z., Wang, Y., Sil, A., & Hajishirzi, H. (2024). Self-RAG: Learning to retrieve, generate, and critique through self-reflection. In *The Twelfth International Conference on Learning Representations*. https://openreview.net/forum?id=hSyW5go0v8",
    "Barnett, S., Kurniawan, S., Thudumu, S., Brannelly, Z., & Abdelrazek, M. (2024). *Seven failure points when engineering a retrieval augmented generation system* (arXiv:2401.05856). arXiv. https://doi.org/10.48550/arXiv.2401.05856",
    "Board of Governors of the Federal Reserve System. (2026). *Revised guidance on model risk management* (SR Letter 26-2). https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm",
    "Cochran, W. G. (1950). The comparison of percentages in matched samples. *Biometrika, 37*(3/4), 256–266. https://doi.org/10.2307/2332378",
    "Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement, 20*(1), 37–46. https://doi.org/10.1177/001316446002000104",
    "Connor, R. J. (1987). Sample size for testing differences in proportions for the paired-sample design. *Biometrics, 43*(1), 207–211. https://doi.org/10.2307/2531961",
    "Consumer Financial Protection Bureau. (n.d.). *Consumer complaint database* [Data set]. https://www.consumerfinance.gov/data-research/consumer-complaints/",
    "Consumer Financial Protection Bureau. (2015). *Disclosure of consumer complaint narrative data* [Final policy statement]. https://files.consumerfinance.gov/f/201503_cfpb_disclosure-of-consumer-complaint-narrative-data.pdf",
    "Consumer Financial Protection Bureau. (2021). *Supervisory highlights: Issue 25, fall 2021*. https://www.consumerfinance.gov/data-research/research-reports/supervisory-highlights-issue-25-fall-2021/",
    "Consumer Financial Protection Bureau. (2023). *Chatbots in consumer finance* [Issue spotlight]. https://files.consumerfinance.gov/f/documents/cfpb_chatbot-issue-spotlight_2023-06.pdf",
    "Consumer Financial Protection Bureau. (2026a). *Consumer complaint database narratives archive* [Data set]. https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/",
    "Consumer Financial Protection Bureau. (2026b). *Consumer Response annual report: January 1–December 31, 2025*. https://files.consumerfinance.gov/f/documents/cfpb_2025-cr-annual-report_2026-03.pdf",
    "Dahl, M., Magesh, V., Suzgun, M., & Ho, D. E. (2024). Large legal fictions: Profiling legal hallucinations in large language models. *Journal of Legal Analysis, 16*(1), 64–93. https://doi.org/10.1093/jla/laae003",
    "Electronic Fund Transfers (Regulation E), 12 C.F.R. pt. 1005 (2026). https://www.ecfr.gov/current/title-12/chapter-X/part-1005",
    "Es, S., James, J., Espinosa Anke, L., & Schockaert, S. (2024). RAGAs: Automated evaluation of retrieval augmented generation. In *Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations* (pp. 150–158). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.eacl-demo.16",
    "Gao, T., Yen, H., Yu, J., & Chen, D. (2023). Enabling large language models to generate text with citations. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing* (pp. 6465–6488). Association for Computational Linguistics. https://doi.org/10.18653/v1/2023.emnlp-main.398",
    "Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., Wang, M., & Wang, H. (2023). *Retrieval-augmented generation for large language models: A survey* (arXiv:2312.10997). arXiv. https://doi.org/10.48550/arXiv.2312.10997",
    "Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics, 6*(2), 65–70.",
    "Hsieh, F. Y., Bloch, D. A., & Larsen, M. D. (1998). A simple method of sample size calculation for linear and logistic regression. *Statistics in Medicine, 17*(14), 1623–1634.",
    "Islam, P., Kannappan, A., Kiela, D., Qian, R., Scherrer, N., & Vidgen, B. (2023). *FinanceBench: A new benchmark for financial question answering* (arXiv:2311.11944). arXiv. https://doi.org/10.48550/arXiv.2311.11944",
    "Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W. (2020). Dense passage retrieval for open-domain question answering. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing* (pp. 6769–6781). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.emnlp-main.550",
    "Kish, L. (1965). *Survey sampling*. Wiley.",
    "Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459–9474.",
    "Liang, K.-Y., & Zeger, S. L. (1986). Longitudinal data analysis using generalized linear models. *Biometrika, 73*(1), 13–22. https://doi.org/10.1093/biomet/73.1.13",
    "Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2024). Lost in the middle: How language models use long contexts. *Transactions of the Association for Computational Linguistics, 12*, 157–173. https://doi.org/10.1162/tacl_a_00638",
    "Magesh, V., Surani, F., Dahl, M., Suzgun, M., Manning, C. D., & Ho, D. E. (2025). Hallucination-free? Assessing the reliability of leading AI legal research tools. *Journal of Empirical Legal Studies, 22*(2), 216–242. https://doi.org/10.1111/jels.12413",
    "McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika, 12*(2), 153–157. https://doi.org/10.1007/BF02295996",
    "National Institute of Standards and Technology. (2023). *Artificial intelligence risk management framework (AI RMF 1.0)* (NIST AI 100-1). U.S. Department of Commerce. https://doi.org/10.6028/NIST.AI.100-1",
    "Nogueira, R., & Cho, K. (2019). *Passage re-ranking with BERT* (arXiv:1901.04085). arXiv. https://doi.org/10.48550/arXiv.1901.04085",
    "Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval, 3*(4), 333–389. https://doi.org/10.1561/1500000019",
    "Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2024). ARES: An automated evaluation framework for retrieval-augmented generation systems. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics* (pp. 338–354). https://doi.org/10.18653/v1/2024.naacl-long.20",
    "Shuster, K., Poff, S., Chen, M., Kiela, D., & Weston, J. (2021). Retrieval augmentation reduces hallucination in conversation. In *Findings of the Association for Computational Linguistics: EMNLP 2021* (pp. 3784–3803). https://doi.org/10.18653/v1/2021.findings-emnlp.320",
    "Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association, 22*(158), 209–212. https://doi.org/10.1080/01621459.1927.10502953",
    "Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2023). ReAct: Synergizing reasoning and acting in language models. In *The Eleventh International Conference on Learning Representations*. https://doi.org/10.48550/arXiv.2210.03629",
    "Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems, 36*, 46595–46623.",
]
for r in REFS:
    d.reference(r)

# ================================================================================ appendices
d.page_break()
d.h1("Appendix A: Declaration of Generative AI Use")
d.table("A1", "Generative AI Use by Project Component", ["Component", "AI used?", "Model", "What it did", "Approval"],
        [["Case selection, sampling, split", "No", "—", "Deterministic filters, seeded stratified draws", "Not applicable"],
         ["Gold labels", "Yes", "Claude Sonnet 4.6", "Drafted clauses and determinations; every draft reviewed by the author; blind second rating", ph("date and approver ONLY if the written approval exists; otherwise state none")],
         ["System under study", "Yes", "Claude Sonnet 4.6", "Generates the determinations evaluated", ph("approval ref")],
         ["Groundedness judge", "Yes", "Claude Sonnet 4.6", "Rates claim support on a sample", ph("approval ref")],
         ["Writing and code assistance", "Yes", ph("model(s) used"), "Code review, analysis and report-builder scripts, drafting and formatting, reference checking", ph("approval ref")]],
        [1900, 800, 1500, 3400, 1760], size=8)
d.h1("Appendix B: Reproducibility")
d.table("B1", "Run Order for the Final Report", ["Step", "Command", "Output"],
        [["0", "python scripts/00a_build_from_archive.py; 00b_build_corpus.py", "Merged archive; corpus"],
         ["1", "python scripts/01_prepare_data.py --n 160 --seed 42; 01b_eda.py", "Development sample; EDA"],
         ["1c", "python scripts/01c_draw_test_sample.py --n-draw 205 --seed 2026", "Test sample; split manifest"],
         ["2", "python scripts/02_import_gold.py [--split test]", "Gold labels; kappa"],
         ["2b", "python scripts/02b_lock_protocol.py", "Protocol lock (SHA-256)"],
         ["3", "python scripts/03_run_experiment.py configs/<c>.yaml --split test", "outputs/runs_test/"],
         ["4", "python scripts/04_compare_and_business.py --split test", "outputs/reports_test/"],
         ["5", "python scripts/05_final_analysis.py --split both", "outputs/final/"],
         ["6", "python scripts/07_outcome_eda.py --split both; 08_sample_preview.py", "Outcome-linked EDA; preview"],
         ["6b", "python scripts/06_pipeline_diagram.py; env_report.py; 00e_build_db.py", "Figure; environment; database"],
         ["7", "cd report; python build_final.py; python check_final.py QM640_Final_Report_DRAFT.docx", "This report; checks"]],
        [600, 5600, 3160], size=8)

d.save(args.out)
print(f"Wrote {args.out}  (results from the {PLABEL}; {d.placeholders} placeholders; ~{d.body_words} body words)")
