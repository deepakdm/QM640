"""Build the QM 640 Interim Report (.docx).

    python build_interim.py --eda <repo>/outputs/eda --reports <repo>/outputs/reports --out Interim.docx

Numbers and figures come from the analysis outputs when they exist; anything not yet available is
written as a highlighted [placeholder] so nothing unverified appears as a finding.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from report_lib import APADoc

HERE = Path(__file__).parent
ap = argparse.ArgumentParser()
ap.add_argument("--base", default=str(HERE / "base_style.docx"))
ap.add_argument("--eda", default="")
ap.add_argument("--reports", default="")
ap.add_argument("--toc", default="", help="json list of [level, text, page] from a first render")
ap.add_argument("--out", default=str(HERE / "QM640_Interim_Report.docx"))
args = ap.parse_args()

EDA = Path(args.eda) if args.eda else None
REP = Path(args.reports) if args.reports else None
S = json.load(open(EDA / "eda_summary.json")) if EDA and (EDA / "eda_summary.json").exists() else {}


def v(path: str, fmt: str = "{}", ph: str | None = None) -> str:
    """Value from eda_summary.json by dotted path, or a highlighted placeholder."""
    cur = S
    for k in path.split("."):
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        elif isinstance(cur, list) and k.isdigit() and int(k) < len(cur):
            cur = cur[int(k)]
        else:
            return f"[[{ph or path}]]"
    try:
        return fmt.format(cur)
    except Exception:
        return str(cur)


def fig(name: str):
    for d in (EDA, REP):
        if d and (d / name).exists():
            return d / name
    return None


T = {k: i + 1 for i, k in enumerate([
    "rq", "vars", "n", "lit", "corpus", "dictA", "dictB", "funnel", "clean", "desc", "insight",
    "cond", "feat", "metrics", "valid", "perf", "risk"])}
F = {k: i + 1 for i, k in enumerate([
    "framework", "pipeline", "funnel", "length", "monthly", "redact", "mix", "goldco", "acc", "fab"])}

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
d.p("**Interim Report**", indent=False, align="center")
d.blank(1)
for line in ["Deepak Dharani", "Walsh College", "QM640: Data Analytics Capstone", "Instructor: Javad Katibai",
             "Mentor: Jainesh Garg", "Term 3, Group 3", "[[Submission date, e.g., November 3, 2026]]"]:
    d.p(line, indent=False, align="center")
d.page_break()

# ================================================================================ GitHub + TOC
d.p("**GitHub Repository**", indent=False, align="center")
d.p("Data files, code, configurations and outputs: https://github.com/deepakdm/QM640 "
    "([[release tag for this submission, e.g., interim-v1]]). The repository layout and the location of "
    "each dataset are given in the GitHub Data Availability Statement and Appendix B.", indent=False)
d.blank(1)
toc_entries = json.load(open(args.toc)) if args.toc and Path(args.toc).exists() else None
d.toc(toc_entries)
d.page_break()
d._counting = True

# ================================================================================ INTRODUCTION
d.p(f"**{TITLE}**", indent=False, align="center")
d.h1("Introduction")
d.h2("Background and Context")
d.p("Electronic payments that consumers dispute, such as a debit card charge they do not recognize, an "
    "automated teller machine (ATM) withdrawal they did not make or a peer-to-peer transfer sent to the wrong "
    "person, are governed in the United States by Regulation E, which implements the Electronic Fund Transfer "
    "Act. Once a consumer reports an error, the institution generally has 10 business days to investigate. It "
    "may take up to 45 days, or 90 days for some transfers, only if it provisionally credits the account, and "
    "it must cap the consumer's liability for unauthorized transfers according to how quickly the loss was "
    "reported (Electronic Fund Transfers [Regulation E], 2026, §§ 1005.6, 1005.11). Every case therefore asks "
    "an investigator to read the consumer's account, find the provision that controls it and reach a "
    "determination that will withstand review by compliance staff and bank examiners.")
d.p("The workload is large and the work is error-prone. The Consumer Financial Protection Bureau (CFPB) "
    "received about 104,200 complaints about checking and savings accounts and 102,900 about money transfers "
    "and virtual currency in 2025 (CFPB, 2026), and these are only the disputes consumers escalated to a "
    "federal regulator. The Bureau's examiners have found institutions that refused to treat misdirected "
    "person-to-person payments as errors, judged investigations only by whether a payment followed the "
    "sender's instructions, and missed the 90-day limit for prepaid-card investigations (CFPB, 2021). Four "
    "groups carry the consequences: consumers waiting for provisional credit, investigators working against a "
    "statutory clock, compliance teams who must evidence each decision, and examiners who audit the process. "
    "A wrongful denial breaches the regulation and harms a consumer who may have been defrauded; a wrongful "
    "approval is a direct loss to the institution.")
d.p("Large language models (LLMs) can read a complaint and draft a reasoned determination in seconds, which "
    "makes them an obvious candidate to support this work. Their weakness is that they invent authority. "
    "General-purpose models gave false answers to between 58% and 88% of verifiable questions about federal "
    "court cases (Dahl et al., 2024), and commercial legal research tools built on retrieval still produced "
    "hallucinated answers 17% to 33% of the time (Magesh et al., 2025). Retrieval-augmented generation (RAG) "
    "gives the model passages from a trusted corpus before it answers (Lewis et al., 2020) and is the standard "
    "mitigation (Shuster et al., 2021), but most of the evidence for it comes from curated benchmark questions. "
    "Real complaints are written by non-specialists, tell one side of the story and arrive with identifying "
    "details redacted. That is the setting in which a dispute team would actually use such a tool, and it is "
    "the setting this study evaluates.")

d.h2("Problem Statement")
d.p("Financial institutions that resolve Regulation E disputes have no evidence-based way to decide whether an "
    "LLM can safely support their investigators. The models draft quickly but are known to fabricate the "
    "regulatory authority behind their conclusions, and the usual safeguard, RAG, has been tested mainly on "
    "clean benchmark questions rather than on the incomplete, one-sided and redacted narratives consumers "
    "actually write. General RAG evaluation frameworks score faithfulness and citation support (Es et al., 2024; "
    "T. Gao et al., 2023), and legal studies report hallucination rates for commercial tools (Magesh et al., "
    "2025), but none combines outcome accuracy, grounding in the governing clause and operational cost into a "
    "single adoption decision tested against an unaugmented baseline on real complaint text. This study "
    "develops that evaluation methodology and applies it to complaints published by the CFPB.")

d.h2("Purpose of the Study")
d.p("The study is evaluative and explanatory rather than predictive. In analytics terms, the LLM acts as a "
    "five-class classifier that assigns each complaint the determination an investigator would make (approve, "
    "deny, partial credit, escalate or request more information). The study measures how that classifier's "
    "accuracy and grounding change when retrieval is added, explains why they change, and judges whether the "
    "result is good enough to use. No model is trained; the deliverable is the methodology and the evidence it "
    "produces. Five research objectives (ROs) organize the work:")
d.bullets([
    "**Identify performance drivers (RO1):** determine which system and case factors influence accuracy, "
    "groundedness and operational value (RQ3).",
    "**Define the evaluation pillars (RO2):** fix measurable indicators and adoption benchmarks for accuracy, "
    "groundedness and operational value before any output is scored.",
    "**Explain relationships between variables (RO3):** estimate how much of the retrieval effect works by "
    "getting the governing clause to the model (RQ3).",
    "**Evaluate performance against the baseline (RO4):** test whether retrieval improves accuracy and "
    "grounding over the same model without retrieval (RQ1, RQ2).",
    "**Determine adoption suitability (RO5):** establish whether, and under what conditions, the deployment "
    "candidate is reliable and economical enough to adopt as decision support (RQ4).",
], numbered=True)
d.p("Throughout, the baseline is the same model with no retrieval. Because the study has no data on human "
    "investigators, comparisons with today's manual process are made only through a cost model whose inputs "
    "are sourced or labeled as assumptions.")
d.figure(F["framework"], "Conceptual Framework: Drivers, Mechanism, Evaluation Pillars and Adoption Decision",
         HERE / "assets" / "fig_framework.png",
         note="System factors are expected to act mainly through whether the governing clause reaches the model; "
              "case factors can lower performance whatever is retrieved. RQ = research question.")
d.p(f"Figure {F['framework']} summarizes the logic. A determination can reach the right outcome while citing "
    "an invented section, or cite the right clause and still reach the wrong outcome, so accuracy and "
    "groundedness are measured separately before they are combined with operational measures into one "
    "adoption decision.")

d.h2("Interim Project Status (Progress Snapshot)")
d.p("The status below is as of [[status date]]. Items marked for confirmation must be checked against the "
    "repository before submission.", indent=False)
d.p("**Completed**", indent=False, keep_next=True)
d.bullets([
    "Topic approved; written approval for the three uses of generative AI received from the instructor on "
    "October 6, 2026 (Appendix A).",
    "Synopsis submitted on October 13, 2026, with the adoption benchmarks and sample size fixed in advance.",
    "CFPB export downloaded with the study filters on [[September 18, 2026; confirm the file includes the "
    "narrative column and passes scripts/00_check_export.py]].",
    "Regulatory corpus assembled (Regulation E, its official interpretations, the CFPB electronic fund "
    "transfer FAQs and the Visa merchant dispute guidelines) [[confirm]].",
    f"Data cleaning and exploratory data analysis (EDA) completed and documented (Tables {T['funnel']} to "
    f"{T['insight']}, Figures {F['funnel']} to {F['mix']}).",
    "Stratified sample of 160 complaints drawn with seed 42 and the labeling sheet generated.",
    "Evaluation code upgraded to version 6: two defects found in a pre-run review were corrected, the "
    "groundedness, driver and adoption measures promised in the synopsis were implemented, and 43 offline "
    "tests pass (Appendix D).",
])
d.p("**In progress**", indent=False, keep_next=True)
d.bullets([
    "Gold labeling: [[n]] of 160 complaints labeled; blind second rating of 30 complaints [[n done]].",
    f"System-level validation on eight use cases per research question (Table {T['valid']}) [[status]].",
    "Runs of the no-retrieval baseline (C0) and Configuration A [[status]].",
])
d.p("**Pending**", indent=False, keep_next=True)
d.bullets([
    "Runs of Configurations B and C, and validation of the LLM judge against manual review on 40 complaints.",
    "Confirmatory tests for RQ1 and RQ2, the RQ3 driver model and the RQ4 benchmark and failure analysis.",
    "Sourcing of cost-model inputs, results freeze (Week 8), presentation (Week 9) and final report (Week 11).",
])
d.h3("Changes Since the Synopsis")
d.p("The research questions, sample size and adoption benchmarks are unchanged. A review of the code before "
    "any model call found two problems that would have biased the comparisons, and both were corrected before "
    "data collection. First, fixed-size passages, used by Configurations A and B, were not being labeled with "
    "the regulation section they contain, so the governing clause could almost never be detected in their "
    "retrieved context; every passage now carries all section references it contains. Second, the shared "
    "instruction told the model to answer only from the supplied excerpts and to escalate when they were "
    "insufficient, which would have pushed the no-retrieval baseline toward escalation and mixed an "
    "instruction effect into the retrieval effect. The instruction is now identical and neutral across "
    "conditions: escalation and requests for information depend on the complaint's facts, not on whether "
    "excerpts are present. Both changes are disclosed here because they were made after the synopsis but "
    "before any result was seen.")

# ================================================================================ SCOPE
d.h1("Scope and Objectives")
d.p(f"The research problem is broken into four questions, each covering one part of the evaluation framework. "
    f"Every complaint is processed under four conditions: no retrieval (C0), dense retrieval with 512-token "
    f"passages (A), dense retrieval with 256-token passages (B), and hybrid lexical-plus-dense retrieval with "
    f"re-ranking (C). Table {T['rq']} traces each question to its data, method and output.")
d.h2("Research Question 1 (RQ1)")
d.p("**Accuracy.** To what extent does retrieval augmentation improve the accuracy of dispute determinations on "
    "real consumer complaints, compared with the same LLM without retrieval? *H1₀:* the accuracy of each "
    "retrieval condition equals that of C0; *H1ₐ:* it differs. Each retrieval condition is compared with C0 by "
    "the exact McNemar (1947) test with Holm's (1979) correction, and a gain of at least 15 percentage points "
    "is the practical threshold.")
d.h2("Research Question 2 (RQ2)")
d.p("**Groundedness.** Does retrieval make determinations better grounded in the regulation, with fewer "
    "citations to provisions that do not exist and more citations of the clause that actually governs the "
    "case? *H2₀:* the fabricated-citation rate, and separately the correct-clause rate, of each retrieval "
    "condition equal those of C0; *H2ₐ:* they differ. Both indicators are deterministic string checks, compared "
    "with C0 by the exact McNemar test with Holm's correction.")
d.h2("Research Question 3 (RQ3)")
d.p("**Performance drivers.** Which system and case factors explain whether a retrieval-augmented determination "
    "is correct, and how much of the difference between configurations works through getting the governing "
    "clause to the model? *H3₀:* the presence of the governing clause in the model's context does not change "
    "the odds of a correct determination (odds ratio [OR] = 1); *H3ₐ:* OR ≠ 1. A logistic regression estimated "
    "by generalized estimating equations (GEE) and clustered by complaint answers it (Liang & Zeger, 1986).")
d.h2("Research Question 4 (RQ4)")
d.p("**Operational value and adoption.** Does the configuration intended for deployment (C, fixed in advance) "
    "meet the operational requirements for adoption, namely consistency, response time, review burden and cost "
    "per case, and which failure modes determine where human review must remain? *H4₀:* the most common "
    "failure category accounts for no more than an even share of errors (.20); *H4ₐ:* its share exceeds .20.")
d.table(T["rq"], "Traceability of Research Questions to Data, Methods and Outputs",
        ["RQ", "Pillar and objective", "Data", "Method", "Key metrics", "Expected output"],
        [["RQ1", "Accuracy; RO4", "Gold determinations; first-run outputs of C0, A, B, C (160 × 4)",
          "Exact McNemar × 3 vs C0, Holm; bootstrap CI of the gain; Cochran's Q",
          "Accuracy, gain over C0, macro-F1", "Accuracy table; gain judged against 15 points"],
         ["RQ2", "Groundedness; RO4", "Cited sections; corpus section index; gold clauses (160 × 4)",
          "Exact McNemar × 3 per indicator, Holm; Wilson CIs",
          "Fabricated-citation rate; correct-clause rate", "Grounding table; benchmark check"],
         ["RQ3", "Drivers; RO1, RO3", "Retrieval observations (160 × 3) with system and case factors",
          "GEE logistic regression, exchangeable correlation, clustered by complaint",
          "Odds ratios; average marginal effects; share via clause", "Driver table and forest plot"],
         ["RQ4", "Adoption; RO4, RO5", "Config C outputs (160 × 3 runs); response times; cost inputs",
          "Benchmark checks on intervals; failure classification; cost sensitivity",
          "Consistency, p95 latency, review burden, break-even accuracy, failure shares",
          "Benchmark table: met, not met or inconclusive"]],
        [600, 1300, 2000, 2100, 1760, 1600],
        note="CI = confidence interval; RO = research objective; p95 = 95th percentile.")

d.h2("Scope, Variables and Feasibility")
d.p("The scope is narrow in subject and deep in measurement. **In scope** are complaints in the three CFPB "
    "products governed by Regulation E (checking or savings account; prepaid card; money transfer, virtual "
    "currency or money service) with a published narrative of 40 to 600 words, received between September 18, "
    "2024 and September 18, 2026; a public regulatory corpus; one pinned LLM (Claude Sonnet 4.6, temperature 0, "
    "three runs per complaint); the four retrieval conditions; and an analytical cost model. **Out of scope** "
    "are Regulation Z credit card disputes, mortgage and credit reporting complaints, licensed card-network "
    "rulebooks and any model training. The study makes **no claim** about human investigator performance or "
    "about any institution's case mix, because published complaints are not a statistical sample of disputes "
    "(CFPB, n.d.).")
d.p(f"Table {T['vars']} lists the variables. The unit of analysis is the complaint, and because every complaint "
    "is processed under all four conditions with the model, instruction, temperature and case set held "
    "constant, differences in outcome can be attributed to the retrieval configuration.")
d.table(T["vars"], "Study Variables, Types and Roles",
        ["Variable", "Type", "Values / unit", "Role", "RQ"],
        [["Drivers and mechanism", "", "", "", ""],
         ["Retrieval presence", "Binary", "C0 vs A, B, C", "Independent (system)", "1, 2"],
         ["Chunk size", "Categorical", "512 (A) vs 256 tokens (B)", "Independent (system)", "3"],
         ["Retrieval strategy", "Categorical", "Dense (A) vs hybrid + re-rank (C)", "Independent (system)", "3"],
         ["Dispute type", "Categorical (7)", "e.g., unauthorized transaction", "Independent (case); strata", "3"],
         ["Narrative length", "Count", "Words, 40–600; log-transformed", "Independent (case)", "3"],
         ["Ambiguity", "Binary", "Rater flags missing decisive facts", "Independent (case)", "1, 3"],
         ["Clause in context", "Binary", "A gold clause is among retrieved passages", "Mediator", "3"],
         ["Outcomes", "", "", "", ""],
         ["Determination correct", "Binary", "Prediction equals gold label", "Dependent (accuracy)", "1, 3"],
         ["Fabricated citation", "Binary", "A cited section is absent from the corpus", "Dependent (groundedness)", "2"],
         ["Correct clause cited", "Binary", "A citation matches a gold clause", "Dependent (groundedness)", "2"],
         ["Consistent", "Binary", "Same determination in all three runs", "Dependent (operational)", "4"],
         ["Response time", "Continuous", "Seconds per complaint", "Dependent (operational)", "4"],
         ["Review required", "Binary", "Draft routed to mandatory review", "Dependent (operational)", "4"],
         ["Failure category", "Nominal (5)", "Unparseable; fabricated citation; retrieval miss; over-escalation; right clause, wrong reasoning", "Dependent (operational)", "4"]],
        [2300, 1400, 2860, 2100, 700], group_rows=(0, 8),
        note="Gold values come from human labeling; all other outcomes are computed by the evaluation scripts.")
d.p(f"**Sample size.** Each question was sized with the method that suits it (Table {T['n']}). RQ1 and RQ2 "
    "compare paired proportions, so Connor's (1987) formula for McNemar's test applies; RQ3 estimates a "
    "logistic-regression coefficient, so the formula of Hsieh et al. (1998) is inflated for correlated "
    "observations with the design effect (Kish, 1965); RQ4 estimates a proportion, so it is sized by the "
    "precision of a confidence interval. All use α = .05 and power = .80, with the planning values anchored on "
    "the 65% accuracy of the best commercial legal RAG tool (Magesh et al., 2025). The final sample is the "
    "largest of the four minimums, 152 complaints, and 160 are labeled to allow for cases found unusable.")
d.table(T["n"], "Minimum Sample Size by Research Question",
        ["Research question", "Method", "Key parameters", "Minimum N"],
        [["RQ1 accuracy vs C0", "Power, paired proportions (McNemar; Connor, 1987)",
          "α = .05/3, power = .80, δ = .15, ψ = .30", "137"],
         ["RQ2 fabricated citation vs C0", "Power, paired proportions (McNemar; Connor, 1987)",
          "α = .05/3, power = .80, δ = .15, ψ = .30", "137"],
         ["RQ3 clause-in-context driver", "Power, logistic regression (Hsieh et al., 1998) × design effect",
          "P₁ = .55, P₂ = .75, B = .75, ρ² = .10, ICC = .40, m = 3", "152"],
         ["RQ4 dominant failure share", "Confidence interval for a proportion",
          "z = 1.96, e = .15, p = .50, error rate = .30", "144"],
         ["Final", "Maximum of the four", "max(137, 137, 152, 144)", "152 (160 labeled)"]],
        [2300, 2900, 3060, 1100],
        note="δ = smallest important difference; ψ = discordant-pair probability; B = share of observations with the "
             "clause in context; ICC = intraclass correlation within complaints; e = margin of error.")
d.p("**Feasibility.** The design is feasible for one researcher in the term. The full experiment needs about "
    "1,920 generation calls (160 complaints × 4 conditions × 3 runs) and 160 judge calls, which run in a few days; "
    "retrieval runs on a laptop CPU; and the code is complete and tested, so the remaining effort is labeling "
    "(about 20 to 30 hours) and analysis. The main schedule risk is labeling, mitigated by labeling in "
    "sittings of about 40 cases and by the reduced-scope fallback set out under Interim Limitations and Risks.")

# ================================================================================ LITERATURE
d.h1("Literature Survey")
d.h2("Literature Review Approach")
d.p("Sources were searched in the ACL Anthology, arXiv, Google Scholar and the CFPB and Federal Register "
    "publication libraries using combinations of *retrieval-augmented generation*, *hallucination*, "
    "*citation*, *faithfulness*, *evaluation*, *LLM-as-a-judge*, *dense retrieval*, *re-ranking*, *legal* and "
    "*financial*. A source was retained if it (a) was peer reviewed or a widely cited preprint, (b) reported an "
    "empirical evaluation or a method this study uses, and (c) bore directly on at least one research question. "
    "Vendor material without a stated method was excluded. Fifteen sources met the criteria and are compared in "
    f"Table {T['lit']}; the statistical references (for example, McNemar, 1947; Liang & Zeger, 1986) are cited "
    "where they are used rather than reviewed.")
d.p("Summary of key literature (minimum 10 relevant sources). For each article the matrix records its purpose "
    "and context, method, findings and relevance to this project.", indent=False, italic=True)
d.table(T["lit"], "Literature Relevance Matrix",
        ["Author (year)", "Domain / context", "Dataset / setting", "Method(s)", "Key findings", "How it supports RQs / decisions"],
        [["Lewis et al. (2020)", "Open-domain NLP", "Natural Questions, TriviaQA, FEVER and others; Wikipedia index",
          "Dense retriever + seq2seq generator trained end to end", "Retrieval beat parametric-only models and gave more specific, factual text",
          "Defines RAG and the parametric-only baseline; basis for C0 vs retrieval (RQ1)"],
         ["Karpukhin et al. (2020)", "Open-domain QA", "Five QA benchmarks", "Dual-encoder dense passage retrieval",
          "Beat BM25 by 9–19 points in top-20 retrieval accuracy", "Dense retriever of Configs A and B (RQ3)"],
         ["Robertson & Zaragoza (2009)", "Information retrieval", "Theory and review", "Probabilistic relevance framework, BM25",
          "Term-frequency saturation and length normalization explain BM25's robustness", "Lexical half of Config C; exact terms and section numbers matter in regulation (RQ3)"],
         ["Nogueira & Cho (2019)", "Passage ranking", "MS MARCO, TREC-CAR", "BERT cross-encoder re-ranking",
          "27% relative gain in MRR@10 over the prior state of the art", "Re-ranking step of Config C (RQ3)"],
         ["Shuster et al. (2021)", "Knowledge-grounded dialogue", "Wizard of Wikipedia, CMU_DoG", "Retrieval-augmented dialogue models, human evaluation",
          "Retrieval substantially reduced knowledge hallucination", "Direction of the RQ2 hypothesis"],
         ["Y. Gao et al. (2023)", "RAG survey", "Literature to 2023", "Taxonomy: naive, advanced, modular RAG",
          "Chunking, hybrid search and re-ranking are the main levers; evaluation is fragmented", "Choice of configuration factors (RQ3)"],
         ["Liu et al. (2024)", "Long-context use", "Multi-document QA, key-value retrieval", "Vary position and amount of context",
          "Accuracy drops when relevant text sits mid-context; more context can hurt", "Two-sided tests; context budget of Config B (RQ1, RQ3)"],
         ["Barnett et al. (2024)", "RAG engineering", "Three deployed RAG systems", "Case studies",
          "Seven failure points, e.g., missing content, not in context, wrong specificity", "Failure taxonomy for RQ4"],
         ["T. Gao et al. (2023)", "Cited generation", "ALCE: ASQA, QAMPARI, ELI5", "Automatic citation recall and precision",
          "Best models lacked full citation support about half the time on ELI5", "Citation-level groundedness measures (RQ2)"],
         ["Es et al. (2024)", "RAG evaluation", "WikiEval", "Reference-free LLM-prompted metrics (RAGAS)",
          "Faithfulness and relevance scores tracked human judgments", "Secondary judged-support measure (RQ2)"],
         ["Saad-Falcon et al. (2024)", "RAG evaluation", "Eight KILT, SuperGLUE and AIS tasks", "Fine-tuned LM judges + prediction-powered inference",
          "A few hundred human labels suffice to calibrate automated judges", "Validating the judge on 40 manually reviewed cases (RQ2)"],
         ["Zheng et al. (2023)", "LLM-as-a-judge", "MT-Bench, Chatbot Arena", "Agreement of GPT-4 judge with humans",
          "Over 80% agreement, but position, verbosity and self-enhancement biases", "Judge kept secondary to deterministic checks (RQ2)"],
         ["Dahl et al. (2024)", "Legal LLMs", "Questions on U.S. federal court cases", "Profiling hallucination across models",
          "Hallucination in 58%–88% of verifiable answers", "Motivation; legal authority is where LLMs fail (RQ2)"],
         ["Magesh et al. (2025)", "Legal RAG tools", "Pre-registered set of 200+ legal queries", "Comparison of commercial RAG tools and GPT-4",
          "RAG tools still hallucinated 17%–33%; best accuracy about 65%", "Planning accuracy; RAG reduces but does not remove errors (RQ1, RQ2)"],
         ["Islam et al. (2023)", "Financial QA", "FinanceBench, 150-case review sample", "16 model and retrieval configurations, manual review",
          "GPT-4-Turbo with retrieval failed or refused 81% of questions", "Retrieval design decides outcomes in finance; similar sample size (RQ1, RQ3)"]],
        [1300, 1150, 1500, 1650, 1880, 1880], size=8,
        note="QA = question answering; MRR = mean reciprocal rank; LM = language model.")
d.p("**Retrieval reduces hallucination but does not remove it (RQ1, RQ2).** The foundational evidence is "
    "consistent: adding retrieved passages makes generated text more factual than relying on the model's "
    "parameters alone (Lewis et al., 2020; Shuster et al., 2021). The domain studies temper that conclusion. "
    "Commercial legal tools built on retrieval still hallucinated in one response in six to one in three "
    "(Magesh et al., 2025), and in financial question answering a strong model with a retrieval system failed "
    "or declined 81% of questions (Islam et al., 2023). Read together, these studies suggest that the benefit "
    "of retrieval depends on what is retrieved and on the domain, and that it should be measured against the "
    "same model without retrieval rather than assumed. That is why this study pairs every complaint across all "
    "four conditions and sets the practical threshold at 15 points before any result is seen.")
d.p("**Retrieval design is the lever (RQ3).** Dense retrieval outperforms lexical matching on open-domain "
    "questions (Karpukhin et al., 2020), yet regulatory text depends on exact terms and section numbers, which "
    "is where BM25 is strong (Robertson & Zaragoza, 2009); a cross-encoder can then re-order the candidates "
    "(Nogueira & Cho, 2019). Surveys identify chunk size, hybrid search and re-ranking as the main design "
    "choices (Y. Gao et al., 2023), and long-context experiments show that adding more text can lower accuracy "
    "when the relevant passage is buried (Liu et al., 2024). What these studies rarely do is measure the "
    "mechanism: whether the passage that decides the case actually reached the model. This study records that "
    "mediator for every observation and estimates how much of each configuration's effect runs through it.")
d.p("**Measuring groundedness (RQ2).** Automatic citation metrics (T. Gao et al., 2023) and LLM-prompted "
    "faithfulness scores (Es et al., 2024) make groundedness measurable at scale, but both depend on another "
    "model's judgment. LLM judges agree with people about as often as people agree with each other while "
    "showing systematic biases (Zheng et al., 2023), and a small set of human labels can correct an automated "
    "judge (Saad-Falcon et al., 2024). In a regulated decision a citation either exists in the regulation or it "
    "does not, so this study makes the deterministic check against the corpus its primary measure and keeps the "
    "LLM judge secondary, validated on 40 manually reviewed cases.")
d.p("**From accuracy to an adoption decision (RQ4).** Engineering case studies catalogue how RAG systems fail "
    "(Barnett et al., 2024), and domain studies report error rates, but neither links failure modes to where "
    "human review must sit or to the cost of reviewing drafts. Three gaps follow from the review: evidence on "
    "real, redacted consumer narratives rather than curated questions; measurement of the retrieval mechanism "
    "rather than only its outcome; and a single decision rule that weighs accuracy, grounding and operational "
    "cost together. The research questions are built to close these gaps.")

# ================================================================================ DATA
d.h1("Data Description")
d.p("The cases are real complaints from the CFPB Consumer Complaint Database, a public, daily-updated record of "
    "complaints about consumer financial products. Each record carries the product, issue, company and company "
    "response and, where the consumer consented, the consumer's own narrative with personal information removed "
    "(CFPB, 2015). The narratives meet the course's real-data requirement, are written in exactly the "
    "unstructured style the research gap concerns, and as federal open data can be shared without license "
    "restrictions. The second data source is the public regulatory corpus that the retrieval conditions search.")
d.h2("Data Source(s) and Access")
d.bullets([
    "Consumer Complaint Database, landing page (CFPB, n.d.): https://www.consumerfinance.gov/data-research/consumer-complaints/",
    "Filtered search and CSV export used for this study: https://www.consumerfinance.gov/data-research/consumer-complaints/search/",
    "Bulk download: https://files.consumerfinance.gov/ccdb/complaints.csv.zip",
    "Field definitions: https://cfpb.github.io/api/ccdb/fields.html",
    "Regulation E, 12 C.F.R. Part 1005 (Electronic Fund Transfers [Regulation E], 2026): https://www.ecfr.gov/current/title-12/chapter-X/part-1005",
])
d.p(f"The export was downloaded on [[September 18, 2026]] with the filter set recorded in "
    f"data/cases/selection_report.json, because the database changes daily and its product and issue labels "
    f"were revised in 2017 and 2023. Table {T['corpus']} lists the corpus documents; section markers such as "
    f"§ 1005.11(c) are preserved because retrieval and citation checks are scored against them.")
d.table(T["corpus"], "Regulatory Corpus Documents",
        ["doc_id", "Source", "Role in the study", "Sections indexed"],
        [["reg_e", "Regulation E, 12 C.F.R. Part 1005 (eCFR)", "Core rules: liability (§ 1005.6), error resolution (§ 1005.11)", "[[n]]"],
         ["reg_e_interp", "Supplement I, official interpretations (CFPB)", "Timing and what counts as unauthorized", "[[n]]"],
         ["cfpb_eft_faq", "CFPB Electronic Fund Transfers FAQs", "Bureau expectations for investigations", "[[n]]"],
         ["network_rules", "Visa Dispute Management Guidelines for Visa Merchants (public PDF)", "Reason codes and conditions; licensed Core Rules excluded", "[[n]]"]],
        [1400, 3200, 3460, 1300],
        note="Section counts are written by scripts/03_run_experiment.py to outputs/runs/<condition>/corpus_index.txt.")

d.h2("Dataset Overview")
d.p("The data has three layers: the raw export, the cleaned candidate pool and the 160-complaint study sample, "
    "to which human gold labels and model outputs are joined by complaint ID.", indent=False)
d.bullets([
    f"**Number of records (rows):** raw export {v('raw.rows', '{:,}', 'N raw rows')}; candidate pool after cleaning "
    f"{v('funnel.7.rows', '{:,}', 'N after cleaning')}; study sample 160; model outputs 160 × 4 conditions × 3 runs = 1,920.",
    f"**Number of variables (columns):** {v('raw.columns', '{}', '18')} in the CFPB export (Table {T['dictA']}); "
    f"the study adds derived, gold-label and outcome fields (Table {T['dictB']}).",
    f"**Time period:** complaints received from September 18, 2024 to September 18, 2026 (observed range "
    f"{v('pool.date_min', '{}', 'first date')} to {v('pool.date_max', '{}', 'last date')}).",
    "**Unit of analysis:** the complaint; model outputs are complaint × condition × run.",
    "**Target variables:** the gold determination (five classes: approve, deny, partial credit, escalate, "
    "request more information) and the gold governing clauses (section references such as 1005.6(b)(1)), both "
    "assigned by a human rater who reads the regulation. The company's response is never used as the label.",
])
d.h2("Data Dictionary (Mandatory)")
d.p(f"Table {T['dictA']} covers every field in the CFPB export and Table {T['dictB']} the fields the study creates.")
A = [
    ["Complaint ID", "Unique CFPB complaint number", "String (ID)", "Integer as text", "Never missing; duplicates removed", "Join key to all outputs"],
    ["Date received", "Date the CFPB received the complaint", "Date", "2024-09-18 to 2026-09-18", "Unparseable dates excluded", "Defines the window"],
    ["Product", "Product type chosen by the consumer", "Categorical", "3 in-scope values", "Out-of-scope values excluded", "Scope filter"],
    ["Sub-product", "Finer product type", "Categorical", "Text label", "Kept blank", "Descriptive only"],
    ["Issue", "Problem chosen by the consumer", "Categorical", "Text label", "Not missing in scope", "Mapped to dispute type"],
    ["Sub-issue", "Finer problem; often blank", "Categorical", "Text label", "Blank: map from Issue alone", "Joined to Issue for mapping"],
    ["Consumer complaint narrative", "Consumer's account, PII redacted as XXXX", "Free text", "40–600 words", "Missing or empty excluded; XXXX kept", "Core model input"],
    ["Company public response", "Optional company statement", "Categorical", "Fixed list", "Mostly blank; not used", "Not used"],
    ["Company", "Company named in the complaint", "String", "Company name", "Kept", "Robustness clustering; not reported by firm"],
    ["State", "Consumer's state", "Categorical", "Two-letter code", "Kept blank", "Descriptive only"],
    ["ZIP code", "Partly masked ZIP code", "String", "Five characters", "Not used", "Not used"],
    ["Tags", "Older American / Servicemember flags", "Categorical", "Label or blank", "Blank = no tag", "Error rates by group"],
    ["Consumer consent provided?", "Consent to publish narrative", "Categorical", "Consent provided, etc.", "Kept", "Confirms narrative availability"],
    ["Submitted via", "Submission channel", "Categorical", "Web, phone, referral, etc.", "Kept", "Descriptive only"],
    ["Date sent to company", "Date forwarded to the company", "Date", "YYYY-MM-DD", "Not used", "Not used"],
    ["Company response to consumer", "How the company closed the complaint", "Categorical", "e.g., Closed with explanation", "Kept", "Rater context only; withheld from the model"],
    ["Timely response?", "Company responded on time", "Binary", "Yes / No", "Kept", "Descriptive only"],
    ["Consumer disputed?", "Legacy field, discontinued 2017", "Categorical", "Yes / No / N/A", "Not used", "Not used"],
]
d.table(T["dictA"], "Data Dictionary, Part A: CFPB Consumer Complaint Database Fields",
        ["Variable name", "Definition", "Datatype", "Allowed values / range", "Missing values handling", "Notes"],
        A, [1650, 1950, 1000, 1500, 1660, 1600], size=8,
        note="Field names follow the CFPB codebook. PII = personally identifiable information. Missingness counts for "
             f"every field are reported in outputs/eda/missingness.csv and summarized under Data Cleaning.")
B = [
    ["Derived case fields", "", "", "", "", ""],
    ["dispute_type", "Normalized category from Issue + Sub-issue", "Categorical", "7 labels", "Unmapped complaints excluded", "Strata; RQ3 factor"],
    ["narrative_words", "Narrative length", "Integer", "40–600", "None (computed)", "RQ3 factor (log, centred)"],
    ["Gold-label fields (human)", "", "", "", "", ""],
    ["governing_clauses", "Section references that decide the case", "List of strings", "e.g., 1005.6(b)(1)", "Required; blank row dropped", "Ground truth, RQ2–RQ3"],
    ["determination", "Correct outcome under the regulation", "Categorical", "5 classes", "Required; blank row dropped", "Ground truth, RQ1, RQ4"],
    ["ambiguous", "Narrative lacks decisive facts", "Binary", "Yes / No", "Blank = No", "RQ3 factor; sensitivity"],
    ["second_rater_determination", "Blind second label (30 cases)", "Categorical", "5 classes", "Blank if not double-rated", "Cohen's κ"],
    ["Outcome fields (per complaint × condition)", "", "", "", "", ""],
    ["pred_determination", "Model's determination from JSON output", "Categorical", "5 classes or none", "Unparseable = incorrect", "RQ1"],
    ["determination_correct", "Prediction equals gold", "Binary", "0 / 1", "None", "DV, RQ1 and RQ3"],
    ["fabricated_citation", "A cited section is absent from the corpus index", "Binary", "0 / 1", "None", "DV, RQ2"],
    ["correct_clause_cited", "A citation matches a gold clause", "Binary", "0 / 1", "None", "DV, RQ2"],
    ["clause_in_context", "A gold clause is among retrieved passages", "Binary", "0 / 1", "Undefined for C0", "Mediator, RQ3"],
    ["consistent", "All three runs agree", "Binary", "0 / 1", "None", "DV, RQ4"],
    ["review_required", "Draft routed to mandatory review", "Binary", "0 / 1", "None", "DV, RQ4"],
    ["latency_s", "End-to-end response time", "Continuous", "Seconds > 0", "Failed calls re-run, not scored", "DV, RQ4"],
    ["failure_category", "Category of an incorrect determination", "Categorical", "5 categories", "Only for errors", "DV, RQ4"],
]
d.table(T["dictB"], "Data Dictionary, Part B: Derived, Gold-Label and Outcome Fields",
        ["Variable name", "Definition", "Datatype", "Allowed values / range", "Missing values handling", "Notes"],
        B, [1900, 2100, 1050, 1300, 1560, 1450], size=8, group_rows=(0, 3, 8),
        note="DV = dependent variable. Outcome fields are produced by scripts/03_run_experiment.py.")
d.h2("GitHub Data Availability Statement")
d.p("The raw and processed datasets, the code and all outputs are available in the GitHub repository "
    "(https://github.com/deepakdm/QM640) under the following paths:", indent=False)
d.bullets([
    "**Raw data:** data/raw/cfpb/complaints.csv.gz (filtered CFPB export, compressed) [[confirm the file is committed; "
    "use Git LFS if it exceeds 100 MB]]; data/corpus/ (regulatory corpus).",
    "**Processed/clean data:** data/cases/sample_160.csv and cases.jsonl (study sample); data/cases/selection_report.json "
    "(filters and counts); data/gold/ (labeling sheet and gold labels); outputs/eda/ (cleaning log, EDA tables and figures).",
    "**Code/notebooks:** scripts/ (numbered pipeline steps), src/dispute_rag/ (library), configs/ (four conditions), "
    "tests/ (offline tests); outputs/runs/ and outputs/reports/ hold model outputs and analysis results.",
])
d.h2("Screenshots / Evidence")
d.p("Appendix B shows a preview of the study export (Figure B1) and the repository tree (Figure B2).")

# ================================================================================ ANALYSIS
d.h1("Analysis")
d.p("This section reports the data cleaning and the exploratory analysis of the complaint data. Model results "
    "are reported under Preliminary Results. Every count below is produced by scripts/01_prepare_data.py and "
    "scripts/01b_eda.py, so the section can be reproduced from the export.")
d.h2("Data Cleaning")
d.p(f"Cleaning is deliberately light because nothing is trained, but each step removes records that would "
    f"either fall outside the regulation being studied or make the comparison unfair. Table {T['funnel']} and "
    f"Figure {F['funnel']} give the number of complaints surviving each step, and Table {T['clean']} summarizes "
    f"how missing values, outliers and duplicates were detected and treated.")
d.bullets([
    "**Narrative present.** Complaints without a published narrative were removed, because there is no case "
    "for the model or the rater to read. This is the largest source of missingness and is structural: the CFPB "
    "publishes a narrative only with the consumer's consent (CFPB, 2015). No imputation is possible or appropriate.",
    "**Study window.** Complaints outside September 18, 2024 to September 18, 2026 were removed, which fixes the "
    "population and avoids the product and issue label changes of 2017 and 2023.",
    "**Regulation E scope.** Only the three products governed by Regulation E were kept, so every case can be "
    "decided from the corpus.",
    "**Dispute-type mapping.** Issue and sub-issue were mapped to seven dispute types by ordered text rules. "
    "Where the sub-issue was blank the issue alone was used; complaints that matched no rule were removed because "
    "they are not payment disputes (for example, account-opening problems).",
    "**Length limits (outliers).** Narratives shorter than 40 words rarely contain the facts needed to label a "
    "clause, and those longer than 600 words are extreme in length and cost. The limits were fixed in the "
    "synopsis; the interquartile-range fence is reported alongside them as a check.",
    "**Duplicates.** Repeated complaint IDs were removed, then exact duplicate narratives after whitespace and case "
    "normalization, because the same story filed twice would otherwise count twice.",
    "**Redactions and leakage.** The CFPB's XXXX redactions were kept and the model is told to treat them as "
    "unknown facts; they are not imputed. The company's response is shown to the rater as context but withheld "
    "from the model so the outcome cannot leak into retrieval.",
    "**Sampling.** A stratified sample of 160 complaints was drawn across the seven dispute types with seed 42, so "
    "the sample does not simply mirror the dominant types.",
])
funnel = S.get("funnel")
rows = []
labels = ["Rows in the export", "Consumer narrative present and non-empty", "Received within the study window",
          "Product governed by Regulation E", "Issue / sub-issue mapped to one of seven dispute types",
          "Narrative length 40–600 words", "Duplicate complaint IDs removed", "Exact duplicate narratives removed"]
for i, lab in enumerate(labels):
    if funnel and i < len(funnel):
        r = funnel[i]
        rows.append([lab, f"{int(r['rows']):,}", f"{int(r['removed']):,}" if i else "—",
                     f"{r['removed_pct_of_previous']:.1f}%" if i and r['removed_pct_of_previous'] == r['removed_pct_of_previous'] else "—",
                     f"{r['retained_pct_of_raw']:.2f}%"])
    else:
        rows.append([lab, "[[n]]", "[[n]]" if i else "—", "[[%]]" if i else "—", "[[%]]"])
rows.append(["Stratified study sample (seed 42)", "160", "—", "—", "—"])
d.table(T["funnel"], "Complaints Remaining After Each Cleaning Step",
        ["Step", "Remaining", "Removed", "% of previous", "% of raw"], rows, [4100, 1300, 1300, 1330, 1330],
        note="Counts from data/cases/selection_report.json (outputs/eda/cleaning_log.csv).")
d.figure(F["funnel"], "Complaints Remaining After Each Cleaning Step", fig("fig1_cleaning_funnel.png"),
         "Figure generated by scripts/01b_eda.py: outputs/eda/fig1_cleaning_funnel.png")
d.p(f"[[Interpretation of Figure {F['funnel']} once run: name the step that removes the most records and its "
    f"share; state how large the final candidate pool is relative to the sample of 160 (the sampling fraction) and "
    f"whether any dispute type had fewer complaints than its stratum quota.]] The size of the pool matters for "
    f"RQ1 to RQ4 because a large pool means the sample of 160 can be stratified without exhausting any dispute "
    f"type, and replacements can be drawn from the same frame if labeled cases prove unusable.")
d.table(T["clean"], "Data Cleaning Log",
        ["Issue", "Variables affected", "Detection method", "Treatment applied", "Rationale"],
        [["Missing values", "Consumer complaint narrative; Sub-issue; Date received",
          f"Null or empty-string test per column over the full export (narrative missing: "
          f"{v('raw.missingness.Consumer complaint narrative', '{:.1f}%', 'n %')})",
          "Rows without narrative removed; blank sub-issue mapped from Issue; unparseable dates excluded; no imputation",
          "Narrative is the case itself; consent-based missingness cannot be imputed; other fields are not model inputs"],
         ["Outliers", "narrative_words",
          f"Pre-set 40/600-word limits; IQR fence Q3 + 1.5 × IQR = {v('pool.iqr_upper_fence', '{:.0f}', 'n')} words "
          f"({v('pool.iqr_outliers_high', '{:,}', 'n')} above fence)",
          f"{v('pool.below_min_words', '{:,}', 'n')} below 40 and {v('pool.above_max_words', '{:,}', 'n')} above 600 words removed",
          "Too short to decide; too long for a predictable prompt budget; limits fixed before the data were seen"],
         ["Duplicates", "Complaint ID; narrative text",
          f"Repeated IDs ({v('raw.duplicate_complaint_ids', '{:,}', 'n')} in raw export); exact match of normalized narrative",
          "Keep first occurrence", "Each case must count once in paired tests"],
         ["Redactions", "Consumer complaint narrative",
          f"XXXX token count ({v('pool.share_with_redaction', '{:.0%}', 'n%')} of narratives contain at least one)",
          "Kept in place; model told to treat as unknown", "Imputing would invent facts"],
         ["Leakage risk", "Company response to consumer", "Design review", "Withheld from model and retrieval query",
          "It reveals the firm's outcome"]],
        [1200, 1700, 2350, 2160, 1950], size=8)
d.p("The workflow is reproducible: the filters, window, seed and download date are recorded in "
    "selection_report.json, and running scripts/00_check_export.py, 01_prepare_data.py --n 160 --seed 42 and "
    "01b_eda.py on the same export regenerates every count in this section (Appendix D).")

d.h2("EDA Results (Interpretation Emphasis)")
d.p(f"Table {T['desc']} describes the candidate pool and the study sample; the figures that follow each "
    f"highlight one property of the data that bears on the research questions, and Table {T['insight']} "
    f"summarizes the insights and the decisions they led to.")
d.table(T["desc"], "Descriptive Statistics of the Candidate Pool and the Study Sample",
        ["Measure", "Candidate pool", "Study sample"],
        [["Complaints", v("pool.rows", "{:,}", "n"), v("sample.n", "{:,}", "160")],
         ["Narrative words, median (IQR)", f"{v('pool.words.50%', '{:.0f}', 'n')} ({v('pool.words.25%', '{:.0f}', 'q1')}–{v('pool.words.75%', '{:.0f}', 'q3')})",
          v("sample.words.50%", "{:.0f}", "n")],
         ["Narrative words, mean (SD)", f"{v('pool.words.mean', '{:.0f}', 'n')} ({v('pool.words.std', '{:.0f}', 'sd')})",
          f"{v('sample.words.mean', '{:.0f}', 'n')} ({v('sample.words.std', '{:.0f}', 'sd')})"],
         ["Skewness of narrative length", v("pool.words_skew", "{:.2f}", "n"), "—"],
         ["Narratives with at least one redaction", v("pool.share_with_redaction", "{:.1%}", "n%"), v("sample.share_with_redaction", "{:.1%}", "n%")],
         ["Median share of words redacted", v("pool.redaction_token_share_median", "{:.1%}", "n%"), "—"],
         ["Distinct companies", v("pool.n_companies", "{:,}", "n"), v("sample.n_companies", "{:,}", "n")],
         ["Share of complaints at the top 10 companies", v("pool.top10_company_share", "{:.1%}", "n%"), "—"],
         ["Company responded on time", v("pool.timely_yes_share", "{:.1%}", "n%"), "—"]],
        [4400, 2480, 2480],
        note="Candidate pool = Regulation E products with a narrative inside the study window, before the length filter. "
             "IQR = interquartile range; SD = standard deviation.")
d.p("[[Interpretation of the table once run: compare pool and sample on length and redaction; note how "
    "concentrated complaints are among the largest companies and what that implies for the company-level "
    "robustness check.]]")
d.figure(F["length"], "Distribution of Narrative Length in the Candidate Pool (Log Scale)", fig("fig2_narrative_length.png"),
         "Figure generated by scripts/01b_eda.py: outputs/eda/fig2_narrative_length.png",
         note="Dashed lines mark the 40- and 600-word inclusion limits.")
d.p(f"[[Interpretation of Figure {F['length']}: shape of the distribution (skewness value), the median and the share "
    f"of complaints outside 40–600 words.]] Narrative length matters for RQ3 because longer narratives may carry "
    f"more decisive facts but also more noise for retrieval; its skew is why the driver model uses the logarithm "
    f"of word count rather than the raw count.")
d.figure(F["monthly"], "Monthly Complaints With Narratives by Regulation E Product", fig("fig3_monthly_volume.png"),
         "Figure generated by scripts/01b_eda.py: outputs/eda/fig3_monthly_volume.png")
d.p(f"[[Interpretation of Figure {F['monthly']}: trend and any spikes or dips by product, and whether any month or "
    f"product is unusual enough to check for a reporting change.]] This matters because a sudden change in volume "
    f"or mix could signal a CFPB labeling change inside the window, which would affect the dispute-type mapping.")
d.figure(F["redact"], "Share of Each Narrative Redacted by the CFPB", fig("fig4_redaction_density.png"),
         "Figure generated by scripts/01b_eda.py: outputs/eda/fig4_redaction_density.png")
d.p(f"[[Interpretation of Figure {F['redact']}: how common redaction is and how much of a typical narrative it "
    f"removes.]] Redactions hide dates, amounts and names that an investigator would normally have, so heavy "
    f"redaction is expected to raise the share of request-more-information determinations; this is checked in "
    f"the gold labels and is one reason the ambiguity flag enters the RQ3 model.")
d.figure(F["mix"], "Dispute-Type Mix in the Candidate Pool and the Stratified Sample", fig("fig5_dispute_type_mix.png"),
         "Figure generated by scripts/01b_eda.py: outputs/eda/fig5_dispute_type_mix.png")
d.p(f"[[Interpretation of Figure {F['mix']}: which dispute types dominate the pool and how stratification changes "
    f"their weight in the sample.]] Because the sample over-represents rare types relative to the pool, overall "
    f"accuracy describes the stratified mix, not the population; results by dispute type are therefore reported "
    f"descriptively, and dispute type enters RQ3 as a control.")
d.figure(F["goldco"], "Gold Determination Against the Company's Response", fig("fig6_gold_vs_company.png"),
         "Figure generated by scripts/01b_eda.py after gold labels are imported: outputs/eda/fig6_gold_vs_company.png")
d.p(f"[[Interpretation of Figure {F['goldco']} once labeling is complete: the distribution of gold determinations "
    f"(class balance), the share flagged ambiguous, Cohen's κ on the double-rated subset, and how often the "
    f"regulation-based label diverges from what the company did.]] Class balance decides whether macro-F1 must "
    f"accompany accuracy in RQ1, and divergence from the company's response is reported as a finding about the "
    f"marketplace rather than an error in the labels.")
d.table(T["insight"], "EDA Insight Summary",
        ["Figure / table", "What it shows", "Key insight", "Why it matters (RQ / objective)", "Decision / next step"],
        [[f"Table {T['funnel']}, Figure {F['funnel']}", "Records surviving each cleaning step", "[[largest loss; pool size]]",
          "Sample must be drawn from an adequate, in-scope frame (all RQs)", "Keep 40–600 limits; replacements from same frame"],
         [f"Figure {F['length']}", "Narrative length distribution", "[[skew; % outside limits]]",
          "Length is an RQ3 case factor", "Log-transform and centre length in the GEE model"],
         [f"Figure {F['monthly']}", "Monthly volume by product", "[[trend; anomalies]]",
          "Label changes would distort dispute-type mapping", "Check mapping coverage by month"],
         [f"Figure {F['redact']}", "Redaction density", "[[% redacted; typical share]]",
          "Missing facts lower achievable accuracy (RQ1, RQ3)", "Keep redactions; ambiguity flag in RQ3; sensitivity without ambiguous cases"],
         [f"Figure {F['mix']}", "Dispute-type mix, pool vs sample", "[[dominant types]]",
          "Stratification changes weights (RQ1 generalization)", "Report by-type results descriptively; dispute type as control"],
         [f"Figure {F['goldco']}", "Gold labels vs company response", "[[class balance; κ; divergence]]",
          "Label validity and class imbalance (RQ1, RQ4)", "Report macro-F1; discuss divergence"]],
        [1300, 1700, 1800, 2280, 2280], size=8)

# ================================================================================ MODELLING
d.h1("Modelling")
d.p("Nothing is trained in this study: the LLM is hosted with fixed weights and the retrieval index is built from "
    "the regulation, not from labeled cases. The \"models\" compared are therefore four system configurations "
    "that share one generator, and the statistical models are the tests used to compare and explain them. The "
    "problem type is classification, since each determination is one of five classes, and correctness is a "
    "binary outcome for every complaint and condition.")
d.h2("Choice of Models with Justification")
d.p("**Model 1: No-retrieval baseline (C0).** The same LLM, instruction and settings with no excerpts. It is the "
    "baseline the rubric and the research design require: without it there is no test of whether retrieval helps "
    "at all, only comparisons between retrieval variants (Lewis et al., 2020).")
d.p("**Model 2: Dense retrieval, 512-token passages (Config A).** A dual-encoder retriever (bge-base-en-v1.5) over "
    "fixed passages, the most common RAG design (Karpukhin et al., 2020; Y. Gao et al., 2023). It is the reference "
    "level for the RQ3 configuration effects.")
d.p("**Model 3: Dense retrieval, 256-token passages (Config B).** Identical to A except passage size, with eight "
    "passages instead of five so the volume of retrieved text stays comparable (about 2,000 against 2,500 tokens). "
    "It isolates the effect of finer passages at a similar context budget (Liu et al., 2024).")
d.p("**Model 4: Hybrid retrieval with re-ranking (Config C).** BM25 and dense scores combined with equal weight, "
    "section-aware passages and a cross-encoder re-ranker (ms-marco-MiniLM-L-6-v2). Regulatory text depends on "
    "exact terms and section numbers, which favors lexical matching (Robertson & Zaragoza, 2009), and re-ranking "
    "improves precision at the top of the list (Nogueira & Cho, 2019). C changes several components together, so "
    "it is interpreted as a structure-aware retrieval package and was fixed in advance as the deployment candidate.")
d.p("**Statistical models.** RQ1 and RQ2 use the exact McNemar test because every condition sees the same "
    "complaints, so an independent-samples test would ignore the pairing and misstate the evidence; Cochran's "
    "(1950) Q is the omnibus check. RQ3 uses a GEE logistic regression because each complaint contributes three "
    "correlated retrieval observations and the institutional question is a population-averaged one (Liang & "
    "Zeger, 1986). Its assumptions are a linear logit in log length, no severe collinearity (variance inflation "
    "below 5), at least 10 events per parameter and enough clusters (160); all are checked and reported. "
    "Configuration effects are experimental and can be read causally; case-factor effects are observational "
    "associations.")
d.p("**Alternatives considered.** Fine-tuning a classifier was rejected because 160 labels are too few and the aim "
    "is to evaluate a deployable assistant, not to build one; long-context prompting with the whole regulation was "
    "rejected because it is impractical for latency and cost (Islam et al., 2023); an LLM judge as the primary "
    "groundedness measure was rejected because of known judge biases (Zheng et al., 2023); and the area under the "
    "receiver operating characteristic curve is not reported because the system returns a class, not a calibrated "
    "score.")
d.table(T["cond"], "Experimental Conditions and Settings",
        ["Setting", "C0 baseline", "A dense 512", "B dense 256", "C hybrid + re-rank"],
        [["Retrieval", "None", "Dense", "Dense", "BM25 + dense (α = .5), cross-encoder re-rank"],
         ["Passages (chunking)", "—", "Fixed, 512 tokens, 64 overlap", "Fixed, 256 tokens, 32 overlap", "Section-aware, ≤ 512 tokens"],
         ["Passages passed to the model (k)", "0", "5", "8", "5 (from 20 candidates)"],
         ["Generation (all)", "Claude Sonnet 4.6, temperature 0, 800 output tokens, 3 runs, JSON output", "Same", "Same", "Same"]],
        [2200, 1800, 1700, 1700, 1960],
        note="Everything not listed is identical across conditions, including the instruction text.")
d.h2("Features Included and Feature Engineering")
d.p(f"Because no model is fitted to the labels, \"features\" here are the inputs the system receives and the "
    f"predictors in the RQ3 driver model. Table {T['feat']} lists them with the basis for each decision.")
d.table(T["feat"], "Feature Set and Usage",
        ["Feature", "Original / engineered", "Type", "Reason for inclusion", "Used in"],
        [["Consumer narrative", "Original", "Text", "The case facts", "Query and prompt, all conditions"],
         ["Product, issue, sub-issue", "Original", "Categorical text", "Consumer's own classification of the problem", "Query and prompt"],
         ["Passage embeddings", "Engineered (bge-base, unit-normalized)", "768-d vector", "Semantic matching of complaint to regulation", "A, B, C retrieval"],
         ["BM25 term weights", "Engineered", "Sparse vector", "Exact terms and section numbers", "C retrieval"],
         ["dispute_type", "Engineered from issue + sub-issue", "Categorical (7)", "Stratification; type-specific difficulty", "Sampling; RQ3"],
         ["log narrative length", "Engineered (log, centred)", "Continuous", "Skewed length; information vs noise", "RQ3"],
         ["ambiguous", "Human flag", "Binary", "Missing decisive facts lower achievable accuracy", "RQ3; RQ1 sensitivity"],
         ["clause_in_context", "Engineered from retrieved section references", "Binary", "Mechanism of the retrieval effect", "RQ3 mediator"],
         ["Configuration (B, C vs A)", "Design", "Dummy variables", "Experimental factor", "RQ3"],
         ["Company response", "Original", "Categorical", "Excluded: would leak the outcome", "Rater context only"]],
        [1900, 2100, 1300, 2400, 1660], size=8)
d.p("**Avoiding leakage.** The company's response is excluded from the query and prompt; gold labels never enter "
    "the corpus or the index; the judge sees only the excerpts and the reasoning, never the gold label; the rater "
    "labels without seeing model outputs; and the benchmarks were fixed before any output was scored. The "
    "ambiguity flag is assigned by the rater from the narrative alone, so it is valid as an explanatory factor but "
    "would not be available to a deployed system, which is stated as a limitation of the driver analysis.")
d.h2("Evaluation Metrics (Include Formulae and Calculations)")
d.p(f"Table {T['metrics']} gives the formula, interpretation and use of each metric with a worked calculation. "
    "Rates are reported with Wilson (1927) intervals and label reliability with Cohen's (1960) κ. "
    "The numbers in the calculation column are illustrative and are not results. Regression metrics such as root "
    "mean square error do not apply to a classification problem and are omitted.")
d.table(T["metrics"], "Evaluation Metrics and Formulae",
        ["Metric", "Formula", "Interpretation", "Used for", "Worked calculation (illustrative)"],
        [["Accuracy", "Acc = (1/N) Σ 1[ŷᵢ = yᵢ]", "Share of correct determinations", "RQ1, RQ3", "105 of 150 correct: 105/150 = .700"],
         ["Precision, recall, F1 (per class)", "P = TP/(TP+FP); R = TP/(TP+FN); F1 = 2PR/(P+R)", "Errors by class", "RQ1", "TP 30, FP 8, FN 10: P = .789, R = .750, F1 = .769"],
         ["Macro-F1", "(1/K) Σ F1ₖ, K = 5", "Accuracy that does not hide weak classes", "RQ1", "F1 = .77, .70, .55, .60, .68: 3.30/5 = .660"],
         ["Wilson 95% CI", "[p̂ + z²/2n ± z√(p̂(1−p̂)/n + z²/4n²)] / (1 + z²/n)", "Interval for a rate", "All rates", "p̂ = .70, n = 150: [.622, .768]"],
         ["Exact McNemar test", "p = 2 × P(X ≤ min(b, c)), X ~ Bin(b + c, .5)", "Paired difference vs C0", "RQ1, RQ2", "b = 34, c = 11: p = .0008"],
         ["Holm correction", "Reject p₍ⱼ₎ while p₍ⱼ₎ ≤ α/(m − j + 1)", "Family-wise error over m = 3", "RQ1, RQ2", ".0008, .012, .030 vs .0167, .025, .05: all rejected"],
         ["Cochran's Q", "Q = (k−1)[kΣCⱼ² − T²]/(kT − ΣRᵢ²)", "Omnibus test across 4 conditions", "RQ1", "χ² with k − 1 = 3 df"],
         ["Fabricated-citation rate", "(1/N) Σ 1[∃ c ∈ Cᵢ: c ∉ S]", "Invented authority", "RQ2", "4 of 160: .025 [.010, .063]: inconclusive vs ≤ .05"],
         ["Correct clause cited", "(1/N) Σ 1[Cᵢ ∩ Gᵢ ≠ ∅]", "Rests on the deciding rule", "RQ2", "Cites 1005.6(b)(1); gold 1005.6(b): match"],
         ["Clause in context", "1[Rᵢ ∩ Gᵢ ≠ ∅]", "Clause reached the model", "RQ3", "Retrieved 1005.6(b) contains gold 1005.6(b)(1): 1"],
         ["Odds ratio", "OR = exp(β); CI = exp(β ± 1.96 SE)", "Effect of a driver", "RQ3", "β = 1.10, SE = .25: OR = 3.00 [1.84, 4.90]"],
         ["Share via clause", "(AME_total − AME_direct)/AME_total", "Mechanism of a configuration effect", "RQ3", "(.12 − .06)/.12 = .50"],
         ["Consistency", "Share with identical determination in 3 runs", "Auditability", "RQ4", "146 of 160: .913 [.858, .947]: met"],
         ["p95 response time", "95th percentile of seconds per case", "Usability", "RQ4", "Bootstrap CI; ≤ 30 s benchmark"],
         ["Dominant failure share", "x/nₑ; one-sided binomial vs .20", "Where review must focus", "RQ4", "20 of 43: .465 [.325, .611], p < .001"],
         ["Break-even accuracy", "1 − (M − A − sV)/(dW)", "Accuracy where assisted = manual time", "RQ4", "25, 12, 1.0 × 4, .85 × 20: 1 − 9/17 = .471"],
         ["Cohen's κ", "(pₒ − pₑ)/(1 − pₑ)", "Label reliability", "Data quality", "(.80 − .30)/(1 − .30) = .714"]],
        [1400, 2300, 1700, 900, 3060], size=8,
        note="ŷ = predicted and y = gold determination; Cᵢ = cited, Rᵢ = retrieved and Gᵢ = gold sections; S = corpus "
             "section index; b, c = discordant pairs; AME = average marginal effect; M = manual minutes, A = assisted "
             "minutes, s = review share, V = review minutes, d = share of errors caught, W = rework minutes.")
d.p("**Validation approach.** There is no train-test split because nothing is fitted; all 160 complaints are "
    "evaluation data and every condition is scored on the same complaints. Uncertainty is handled by Wilson and "
    "bootstrap intervals (10,000 resamples of complaints), a robustness bootstrap that resamples companies, and a "
    "five-partition stability check that reports the spread of accuracy across folds. That check serves the intent "
    "of cross-validation, not over-reading one estimate, without implying a training step. The LLM judge is "
    "validated against manual review on 40 complaints before its scores are used.")

# ================================================================================ PRELIMINARY RESULTS
d.h1("Preliminary Results")
d.p("At the interim stage the pipeline has been validated end to end, and [[state which conditions have been run, "
    "e.g., C0 and A on n complaints]]. The results below are preliminary: they are reported to show the analysis "
    "works and to flag problems early, not to answer the research questions.")
d.h2("Preliminary Findings by Research Question")
d.p(f"**System-level validation.** Before the full runs, eight use cases per question were checked against "
    f"outputs written down in advance (Table {T['valid']}; case records in Appendix C). With eight cases a single "
    f"mismatch moves the match rate by 12.5 points, so the check detects defects in parsing, retrieval and scoring "
    f"but does not estimate reliability.")
d.table(T["valid"], "Preliminary System Validation by Research Question",
        ["RQ", "What is compared", "Acceptable match", "Tested", "Matched", "Match rate", "Discrepancies and cause"],
        [["RQ1", "Predicted vs expected determination", "Same class", "[[8]]", "[[n]]", "[[%]]", "[[ ]]"],
         ["RQ2", "Citations vs corpus index and expected clause", "No invented section; expected clause cited", "[[8]]", "[[n]]", "[[%]]", "[[ ]]"],
         ["RQ3", "Computed drivers vs manual check", "All values agree", "[[8]]", "[[n]]", "[[%]]", "[[ ]]"],
         ["RQ4", "Rule vs manual failure category; operational fields", "Same category; no missing fields", "[[8]]", "[[n]]", "[[%]]", "[[ ]]"]],
        [600, 2000, 1900, 800, 900, 1000, 2160], size=8,
        note="Produced by scripts/03b_validate.py (outputs/validation/validation_summary.csv).")
d.p("**RQ1 (accuracy).** *Completed:* gold labels for [[n]] complaints; run of C0 and Config A [[status]]. "
    f"*Preliminary output:* [[accuracy of C0 and A with Wilson intervals, the paired difference and its bootstrap "
    f"interval, Figure {F['acc']}, Table {T['perf']}]]. *Remaining:* Configs B and C, the Holm-adjusted tests on all "
    "160 complaints and the macro-F1 view.")
d.p("**RQ2 (groundedness).** *Completed:* the corpus section index and the deterministic citation checks. "
    f"*Preliminary output:* [[fabricated-citation and correct-clause rates for C0 and A, Figure {F['fab']}; list of "
    "any flagged citations reviewed by hand to confirm they are truly absent from the regulation]]. *Remaining:* B "
    "and C, the judge validation on 40 complaints and the benchmark verdicts.")
d.p("**RQ3 (drivers).** *Completed:* the driver model is implemented and was checked on simulated data, where it "
    "recovered a known clause effect. *Preliminary output:* [[clause-in-context rate for Config A]]; the model needs "
    "at least two retrieval conditions, so estimates follow the B and C runs. *Remaining:* GEE odds ratios, average "
    "marginal effects and the mediation share with its bootstrap interval.")
d.p("**RQ4 (operational value).** *Completed:* review routing, failure classification and the cost model. "
    "*Preliminary output:* [[consistency, p95 response time and review burden for the conditions run so far]]; under "
    "the stated assumptions the break-even accuracy is .47 when every draft is reviewed, falling as the review share "
    "falls. *Remaining:* the Config C benchmark table, failure shares and sourced cost inputs.")
d.figure(F["acc"], "Preliminary Determination Accuracy by Condition With 95% Wilson Intervals",
         fig("fig7_accuracy_by_condition.png"),
         "Figure generated by scripts/04_compare_and_business.py: outputs/reports/fig7_accuracy_by_condition.png")
d.figure(F["fab"], "Preliminary Share of Determinations Citing a Non-Existent Section by Condition",
         fig("fig8_fabricated_by_condition.png"),
         "Figure generated by scripts/04_compare_and_business.py: outputs/reports/fig8_fabricated_by_condition.png")
d.h2("Preliminary Model Performance")
d.table(T["perf"], "Preliminary Model Performance Comparison",
        ["Model", "Feature set", "Validation approach", "Accuracy [95% CI]", "Fabricated citation [95% CI]", "Key takeaway"],
        [["C0 baseline", "Narrative + product/issue; no excerpts", "Paired, n = [[n]], run 1", "[[ ]]", "[[ ]]", "[[ ]]"],
         ["A dense 512", "+ 5 dense passages", "Paired, n = [[n]], run 1", "[[ ]]", "[[ ]]", "[[ ]]"],
         ["B dense 256", "+ 8 dense passages", "Paired", "Not yet run", "Not yet run", "Week 6–7"],
         ["C hybrid + re-rank", "+ 5 re-ranked hybrid passages", "Paired", "Not yet run", "Not yet run", "Week 6–7"]],
        [1500, 2100, 1500, 1350, 1350, 1560], size=8,
        note="Accuracy and fabricated-citation rates use the first of three runs; intervals are Wilson intervals.")
d.p("[[Interpretation once C0 and A are run: direction and size of the difference, whether the interval excludes "
    "zero, what the error mix looks like, and any surprise worth checking before B and C are run.]]")

# ================================================================================ LIMITATIONS / NEXT
d.h1("Interim Limitations and Risks")
d.table(T["risk"], "Interim Limitations, Risks and Mitigation",
        ["Limitation or risk", "Effect on findings", "Mitigation planned"],
        [["Gold labeling is the critical path; one primary rater", "Delays every RQ; label errors bias accuracy both ways",
          "Label in sittings of about 40; blind second rater on 30 cases with κ; if late, label at least 137 (the RQ1–RQ2 minimum) and report the RQ3 power shortfall; dropping Config B (m = 2) lowers the RQ1 minimum to 125"],
         ["Published narratives are one-sided, redacted and not a random sample", "Accuracy may understate performance on full case files",
          "Ambiguity flag; sensitivity analysis without ambiguous cases; no population claims"],
         ["Real provisions outside the corpus are scored as fabricated", "Could overstate fabrication",
          "Every flagged citation is listed (fabricated_refs) and reviewed by hand before reporting"],
         ["Hosted model may change during the study", "Runs not comparable over time", "Version string recorded per response; all conditions run in one window; raw responses saved"],
         ["Amount-based review rule cannot be applied", "Review burden slightly understated", "Stated as a limitation of the RQ4 routing"],
         ["Ambiguity flag is assigned by the rater", "Explanatory only, not available in deployment", "Interpreted as a case characteristic, not a deployable feature"],
         ["Raw export size for GitHub", "Evaluator access", "Commit compressed export or use Git LFS; document re-export steps"],
         ["Small validation sample", "Detects defects, not reliability", "Every mismatch traced and fixed or recorded"]],
        [2700, 2900, 3760], size=8)
d.h1("Next Steps (for Final Report)")
d.bullets([
    "Week 6: finish gold labeling and the blind second rating; report κ and clause agreement.",
    "Week 6: complete the system validation (Appendix C) and fix any discrepancies.",
    "Weeks 6–7: run C0, A, B and C on all 160 complaints (three runs each); validate the LLM judge on 40 complaints.",
    "Week 7: RQ1 and RQ2 confirmatory tests with Holm correction; manual review of all flagged citations.",
    "Weeks 7–8: RQ3 GEE driver model with diagnostics; RQ4 benchmark table, failure analysis and sourced cost model.",
    "Week 8: freeze outputs and tag the repository; Week 9 presentation; Weeks 10–11 final report.",
])

# ================================================================================ REFERENCES
d.page_break()
d.h1("References")
REFS = [
    "Barnett, S., Kurniawan, S., Thudumu, S., Brannelly, Z., & Abdelrazek, M. (2024). *Seven failure points when engineering a retrieval augmented generation system* (arXiv:2401.05856). arXiv. https://doi.org/10.48550/arXiv.2401.05856",
    "Cochran, W. G. (1950). The comparison of percentages in matched samples. *Biometrika, 37*(3/4), 256–266. https://doi.org/10.2307/2332378",
    "Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement, 20*(1), 37–46. https://doi.org/10.1177/001316446002000104",
    "Connor, R. J. (1987). Sample size for testing differences in proportions for the paired-sample design. *Biometrics, 43*(1), 207–211. https://doi.org/10.2307/2531961",
    "Consumer Financial Protection Bureau. (n.d.). *Consumer complaint database* [Data set]. Retrieved September 18, 2026, from https://www.consumerfinance.gov/data-research/consumer-complaints/",
    "Consumer Financial Protection Bureau. (2015). *Disclosure of consumer complaint narrative data* [Final policy statement]. https://files.consumerfinance.gov/f/201503_cfpb_disclosure-of-consumer-complaint-narrative-data.pdf",
    "Consumer Financial Protection Bureau. (2021). *Supervisory highlights: Issue 25, fall 2021*. https://www.consumerfinance.gov/data-research/research-reports/supervisory-highlights-issue-25-fall-2021/",
    "Consumer Financial Protection Bureau. (2026). *Consumer Response annual report: January 1–December 31, 2025*. https://files.consumerfinance.gov/f/documents/cfpb_2025-cr-annual-report_2026-03.pdf",
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
    "Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in Neural Information Processing Systems, 33*, 9459–9474. https://papers.nips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html",
    "Liang, K.-Y., & Zeger, S. L. (1986). Longitudinal data analysis using generalized linear models. *Biometrika, 73*(1), 13–22. https://doi.org/10.1093/biomet/73.1.13",
    "Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2024). Lost in the middle: How language models use long contexts. *Transactions of the Association for Computational Linguistics, 12*, 157–173. https://doi.org/10.1162/tacl_a_00638",
    "Magesh, V., Surani, F., Dahl, M., Suzgun, M., Manning, C. D., & Ho, D. E. (2025). Hallucination-free? Assessing the reliability of leading AI legal research tools. *Journal of Empirical Legal Studies, 22*(2), 216–242. https://doi.org/10.1111/jels.12413",
    "McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika, 12*(2), 153–157. https://doi.org/10.1007/BF02295996",
    "Nogueira, R., & Cho, K. (2019). *Passage re-ranking with BERT* (arXiv:1901.04085). arXiv. https://doi.org/10.48550/arXiv.1901.04085",
    "Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval, 3*(4), 333–389. https://doi.org/10.1561/1500000019",
    "Saad-Falcon, J., Khattab, O., Potts, C., & Zaharia, M. (2024). ARES: An automated evaluation framework for retrieval-augmented generation systems. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)* (pp. 338–354). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.naacl-long.20",
    "Shuster, K., Poff, S., Chen, M., Kiela, D., & Weston, J. (2021). Retrieval augmentation reduces hallucination in conversation. In *Findings of the Association for Computational Linguistics: EMNLP 2021* (pp. 3784–3803). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.findings-emnlp.320",
    "Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association, 22*(158), 209–212. https://doi.org/10.1080/01621459.1927.10502953",
    "Zheng, L., Chiang, W.-L., Sheng, Y., Zhuang, S., Wu, Z., Zhuang, Y., Lin, Z., Li, Z., Li, D., Xing, E. P., Zhang, H., Gonzalez, J. E., & Stoica, I. (2023). Judging LLM-as-a-judge with MT-Bench and Chatbot Arena. *Advances in Neural Information Processing Systems, 36*, 46595–46623. https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract.html",
]
for r in REFS:
    d.reference(r)

# ================================================================================ APPENDICES
d._counting = False
d.page_break()
d.h1("Appendix A: Declaration of Generative AI Use")
d.p("The syllabus treats generative AI as an unauthorized aid unless the instructor approves its use in advance and in "
    "writing. Table A1 lists each component of the project and the approval reference.", indent=False)
d.table("A1", "Generative AI Use by Project Component",
        ["Component", "AI used?", "Model", "What it does", "Approval reference"],
        [["Case selection and sampling", "No", "—", "Deterministic filters; stratified sample with recorded seed", "Not applicable"],
         ["Gold labels", "No", "—", "Human rater(s) reading the regulatory corpus", "Not applicable"],
         ["System under study", "Yes", "Claude Sonnet 4.6", "Generates the determinations being evaluated", "October 6, 2026, Javad Katibai"],
         ["Groundedness judge", "Yes", "Claude Sonnet 4.6", "Rates claim support on 40 complaints; validated by manual review", "Same as above"],
         ["Writing and code assistance", "Yes", "[[model used]]", "Code review and fixes, analysis scripts, drafting and formatting of this report, reference checking", "Same as above"]],
        [1900, 900, 1500, 3160, 1900], size=8)
d.h1("Appendix B: Dataset Screenshots")
d.p("Figure B1 shows the first rows of the filtered study export and Figure B2 the repository structure, as referenced in "
    "the Data Description.", indent=False)
d.figure("B1", "Preview of the Filtered Study Export", None,
         "Insert screenshot: first rows of data/raw/cfpb/complaints.csv (or data/cases/sample_160.csv) showing the column headers, "
         "including Consumer complaint narrative",
         note="Consumer Financial Protection Bureau (n.d.). In the public domain.")
d.figure("B2", "Repository Structure", None,
         "Insert screenshot: tree of the repository (e.g., output of tree /F or tree -L 3) showing configs, data/raw, data/cases, "
         "data/gold, data/corpus, data/validation, outputs/eda, outputs/runs, outputs/reports, scripts, src and tests")
d.h1("Appendix C: Preliminary Validation Case Records")
d.p("Table C1 records each validation use case; expected outputs were written down before the system was run.", indent=False)
d.table("C1", "Validation Use Cases, Expected and Observed Outputs",
        ["Case ID", "RQ", "Input summary", "Expected output", "Observed output", "Match", "Comment"],
        [["[[ ]]", "RQ1", "[[ ]]", "[[ ]]", "[[ ]]", "[[ ]]", "[[ ]]"],
         ["[[ ]]", "RQ2", "[[ ]]", "[[ ]]", "[[ ]]", "[[ ]]", "[[ ]]"]],
        [1100, 600, 1700, 1600, 2000, 700, 1660], size=8,
        note="Full records: outputs/validation/validation_records.csv, produced by scripts/03b_validate.py.")
d.h1("Appendix D: Code and Reproducibility")
d.p("Table D1 gives the run order; each script reads the output of the one before it, so every table and figure in this "
    "report can be regenerated from the export. The code referenced in the Analysis and Modelling sections is in "
    "src/dispute_rag/; the citation-matching rules used for RQ2 and RQ3 are in refs.py, the driver model in drivers.py "
    "and the confirmatory tests and benchmark verdicts in analysis.py. CHANGES_v6.md documents the pre-run corrections "
    "described under Changes Since the Synopsis.", indent=False)
d.table("D1", "Run Order, Inputs and Outputs",
        ["Step", "Command", "Input", "Output"],
        [["0", "python scripts/00_check_export.py", "data/raw/cfpb/complaints.csv", "READY, or a list of problems"],
         ["1", "python scripts/01_prepare_data.py --n 160 --seed 42", "Validated export", "cases.jsonl, sample_160.csv, selection_report.json, labeling sheet"],
         ["1b", "python scripts/01b_eda.py", "Export, sample, gold (if any)", "outputs/eda/: cleaning log, missingness, EDA tables and figures"],
         ["2", "python scripts/02_import_gold.py", "Completed labeling sheet", "gold_cases.jsonl; κ and clause agreement"],
         ["3b", "python scripts/03b_validate.py --make-sheet, then --config …", "Gold set; validation run", "outputs/validation/"],
         ["3", "python scripts/03_run_experiment.py configs/<condition>.yaml", "Cases, gold, corpus", "outputs/runs/<condition>/ (resumable)"],
         ["4", "python scripts/04_compare_and_business.py", "All runs", "outputs/reports/: RQ1–RQ4 tables, figures, benchmarks"],
         ["Tests", "pytest tests -q", "None (offline)", "43 tests"]],
        [700, 3600, 2100, 2960], size=8,
        note="Python 3.11; package versions in requirements.txt; model version recorded with every response.")

d.save(args.out)
print(f"saved {args.out}; body words ≈ {d.body_words}; placeholders = {d.placeholders}")
print(json.dumps(d.headings))
