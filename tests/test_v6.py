"""Offline tests for the v6 changes: reference matching, chunk references, grounding flags,
failure categories, the cleaning funnel, control-vs-retrieval tests, the RQ3 driver model and the
cost model. Synthetic fixtures only; no API, no model downloads, no CFPB data."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from dispute_rag.config import ChunkingCfg
from dispute_rag.corpus import annotate_lines, chunk_fixed, chunk_section, corpus_section_index
from dispute_rag.refs import classify_citations, cites_gold, clause_in_context, normalize_ref
from dispute_rag.schemas import GoldLabel, RAGAnswer

REG_E = """§ 1005.6 Liability of consumer for unauthorized transfers.
(a) Conditions for liability. A consumer may be held liable only if the access device is accepted.
(b) Limitations on amount of liability.
(1) Timely notice given. If the consumer notifies the institution within two business days, liability shall not exceed $50.
(2) Timely notice not given. If the consumer fails to notify within two business days, liability shall not exceed $500.
§ 1005.11 Procedures for resolving errors.
(a) Definition of error. The term error means an unauthorized electronic fund transfer.
(b) Notice of error from consumer.
(1) Timing; contents. A financial institution shall comply with the requirements of this section.
(c) Time limits and extent of investigation.
(1) Ten-day period. A financial institution shall investigate promptly and determine whether an error occurred within 10 business days.
(2) Exception. The institution may take up to 45 days if it provisionally credits the account.
"""


def _answer(**kw):
    base = dict(case_id="c1", config_name="hybrid_rerank_512", run_idx=0, retrieved_chunk_ids=["x"],
                retrieved_section_refs=["1005.6(b)"], determination="approve_dispute", reasoning="",
                cited_section_refs=["1005.6(b)(1)"], latency_s=2.0, raw_response="{}")
    base.update(kw)
    return RAGAnswer(**base)


def _gold(**kw):
    base = dict(case_id="c1", governing_clauses=["1005.6(b)(1)"], determination="approve_dispute",
                rationale="r", labeled_by="DD")
    base.update(kw)
    return GoldLabel(**base)


# --- references -------------------------------------------------------------------------------

def test_normalize_ref_variants():
    for s in ["§ 1005.11(c)", "12 CFR 1005.11(c)", "12 C.F.R. § 1005.11 (c)", "[1005.11(c)]", "Section 1005.11(c)."]:
        assert normalize_ref(s) == "1005.11(c)", s
    assert normalize_ref("Reason Code 10.4") == "10.4"
    assert normalize_ref("Regulation E") is None


def test_cites_gold_is_hierarchical():
    assert cites_gold(["1005.6(b)(1)"], ["1005.6(b)"])        # more specific counts
    assert not cites_gold(["1005.6"], ["1005.6(b)(1)"])        # parent alone does not
    assert not cites_gold(["1005.6(b)(10)"], ["1005.6(b)(1)"])  # no prefix false positive


def test_clause_in_context_accepts_parent_passage():
    assert clause_in_context(["1005.6(b)"], ["1005.6(b)(1)"])
    assert not clause_in_context(["1005.11(c)"], ["1005.6(b)(1)"])


def test_fabrication_against_corpus_index():
    idx = corpus_section_index({"reg_e": REG_E})
    assert {"1005.6", "1005.6(b)", "1005.6(b)(1)", "1005.11(c)(2)"} <= idx
    cls = classify_citations(["1005.6(b)(1)", "1005.6(b)(7)", "1005.99", "Regulation E", "1005.11(c)(1)(i)"], idx)
    assert cls["fabricated"] == ["1005.6(b)(7)", "1005.99"]
    assert cls["unverifiable"] == ["Regulation E"]
    assert "1005.11(c)(1)(i)" in cls["valid"]   # third level checked at indexed depth


# --- chunk references (bug fix: fixed chunks used to carry no reference) ------------------------

def test_annotation_tracks_two_levels():
    finest = [f for _, _, f in annotate_lines(REG_E)]
    assert "1005.6(b)(1)" in finest and "1005.11(c)(2)" in finest


def test_fixed_chunks_carry_references():
    chunks = chunk_fixed("reg_e", REG_E, ChunkingCfg(chunk_size=40, overlap=8))
    assert all(c.section_refs for c in chunks)
    assert any("1005.11(c)(1)" in c.section_refs for c in chunks)


def test_fixed_chunks_with_messy_whitespace():
    messy = REG_E.replace("\n", "\n\n   ").replace(" shall ", "   shall   ")
    chunks = chunk_fixed("reg_e", messy, ChunkingCfg(chunk_size=512, overlap=64))
    assert chunks[0].section_ref == "1005.6"
    assert "1005.6(b)(2)" in chunks[0].section_refs


def test_section_chunks_split_on_letters():
    chunks = chunk_section("reg_e", REG_E, ChunkingCfg(strategy="section"))
    refs = [c.section_ref for c in chunks]
    assert "1005.6(b)" in refs and "1005.11(c)" in refs
    b = next(c for c in chunks if c.section_ref == "1005.6(b)")
    assert "1005.6(b)(1)" in b.section_refs and "$50" in b.text


def test_inline_reference_is_not_a_heading():
    text = "§ 1005.6 Liability.\n(a) Text.\n§ 1005.11(c) applies here as a cross-reference.\n"
    finest = [f for _, _, f in annotate_lines(text)]
    assert finest[-1] == "1005.6(a)"


# --- schema robustness ------------------------------------------------------------------------

def test_determination_coercion():
    assert _answer(determination="Approve").determination == "approve_dispute"
    assert _answer(determination="Request more information").determination == "request_more_info"
    assert _answer(determination="maybe").determination is None
    assert _answer(cited_section_refs="1005.6(b)").cited_section_refs == ["1005.6(b)"]


def test_parse_response_handles_fences():
    from dispute_rag.pipeline import parse_response
    obj, ok = parse_response('```json\n{"determination": "deny_dispute", "cited_sections": []}\n```')
    assert ok and obj["determination"] == "deny_dispute"
    assert parse_response("I cannot help")[1] is False


def test_system_prompt_is_condition_neutral():
    from dispute_rag.pipeline import SYSTEM_PROMPT, build_user_message
    assert "ONLY on the provided excerpts" not in SYSTEM_PROMPT
    assert "Do not choose either merely because excerpts are missing" in SYSTEM_PROMPT
    assert "(none provided)" in build_user_message("q", [])


# --- scoring ----------------------------------------------------------------------------------

def test_score_case_flags():
    from dispute_rag.evaluate import score_case
    idx = corpus_section_index({"reg_e": REG_E})
    runs = [_answer(run_idx=i) for i in range(3)]
    r = score_case(runs, _gold(), idx)
    assert r["determination_correct"] == 1 and r["correct_clause_cited"] == 1
    assert r["fabricated_citation"] == 0 and r["clause_in_context"] == 1
    assert r["consistent"] == 1 and r["review_required"] == 0
    runs[2] = _answer(run_idx=2, determination="deny_dispute")
    r = score_case(runs, _gold(), idx)
    assert r["consistent"] == 0 and r["review_required"] == 1


def test_control_has_no_context_measures():
    from dispute_rag.evaluate import score_case
    idx = corpus_section_index({"reg_e": REG_E})
    r = score_case([_answer(retrieved_chunk_ids=[], retrieved_section_refs=[])], _gold(), idx)
    assert np.isnan(r["clause_in_context"]) and np.isnan(r["any_hit"]) and np.isnan(r["citation_validity"])


def test_failure_category_order():
    from dispute_rag.failure_analysis import classify
    idx = corpus_section_index({"reg_e": REG_E})
    g = _gold()
    assert classify(_answer(determination=None), g, idx) == "unparseable"
    assert classify(_answer(determination="deny_dispute", cited_section_refs=["1005.99"]), g, idx) == "fabricated_citation"
    assert classify(_answer(determination="deny_dispute", retrieved_section_refs=["1005.11(a)"]), g, idx) == "retrieval_miss"
    assert classify(_answer(determination="escalate"), g, idx) == "over_escalation"
    assert classify(_answer(determination="deny_dispute"), g, idx) == "right_clause_wrong_reasoning"
    assert classify(_answer(determination="deny_dispute", retrieved_chunk_ids=[], retrieved_section_refs=[],
                            cited_section_refs=["1005.11(c)"]), g, idx) == "clause_miss"


# --- cleaning funnel --------------------------------------------------------------------------

def _raw_csv(path: Path):
    rows = []
    base = dict(Product="Checking or savings account", Issue="Managing an account",
                **{"Sub-issue": "Banking errors"}, Company="BANK X", State="MI",
                **{"Submitted via": "Web", "Company response to consumer": "Closed with explanation",
                   "Timely response?": "Yes", "Sub-product": "Checking account"})
    long_text = " ".join(["word"] * 80)
    for i in range(20):
        rows.append({**base, "Complaint ID": str(i), "Date received": "2025-03-01",
                     "Consumer complaint narrative": f"{long_text} case {i} XXXX"})
    rows.append({**base, "Complaint ID": "100", "Date received": "2025-03-01", "Consumer complaint narrative": ""})
    rows.append({**base, "Complaint ID": "101", "Date received": "2019-01-01", "Consumer complaint narrative": long_text + " old"})
    rows.append({**base, "Complaint ID": "102", "Date received": "2025-03-01", "Product": "Mortgage",
                 "Consumer complaint narrative": long_text + " mortgage"})
    rows.append({**base, "Complaint ID": "103", "Date received": "03/02/25", "Consumer complaint narrative": "too short"})
    rows.append({**base, "Complaint ID": "0", "Date received": "2025-03-01",
                 "Consumer complaint narrative": f"{long_text} case 0 XXXX"})          # duplicate ID
    rows.append({**base, "Complaint ID": "104", "Date received": "2025-03-01",
                 "Consumer complaint narrative": f"{long_text}  case 1   XXXX"})        # duplicate text
    rows.append({**base, "Complaint ID": "105", "Date received": "2025-03-01", "Sub-issue": None,
                 "Issue": "Unauthorized transactions or other transaction problem",
                 "Consumer complaint narrative": long_text + " unauthorized"})
    pd.DataFrame(rows).to_csv(path, index=False)


def test_funnel_counts_and_dedup(tmp_path):
    from dispute_rag.cfpb_cases import load_candidates, to_cases
    p = tmp_path / "complaints.csv"
    _raw_csv(p)
    funnel = {}
    cand = load_candidates(p, funnel=funnel)
    assert funnel["rows_read"] == 27
    assert funnel["narrative_present"] == 26
    assert funnel["in_date_window"] == 25
    assert funnel["in_scope_product"] == 24
    assert funnel["length_40_600"] == 23
    assert funnel["unique_complaint_id"] == 22
    assert funnel["unique_narrative"] == 21 == len(cand)
    cases = to_cases(cand)                       # blank sub-issue must not crash
    assert any(c.complaint.sub_issue is None for c in cases)


# --- confirmatory tests and drivers -----------------------------------------------------------

def _long_frame(n=160, seed=0):
    rng = np.random.default_rng(seed)
    types = ["unauthorized_transaction", "error_resolution", "duplicate_charge", "goods_not_received"]
    rows = []
    for i in range(n):
        dt, words, amb = types[i % 4], int(rng.integers(40, 600)), int(rng.random() < .15)
        base_ok = rng.random() < .55
        rows.append(dict(case_id=str(i), condition="C0", config="control_no_retrieval",
                         determination_correct=int(base_ok), fabricated_citation=int(rng.random() < .25),
                         correct_clause_cited=int(rng.random() < .4), clause_in_context=np.nan,
                         consistent=1, review_required=1, latency_s=3.0, dispute_type=dt,
                         narrative_words=words, ambiguous=amb))
        for cond, p_clause in (("A", .7), ("B", .7), ("C", .85)):
            clause = int(rng.random() < p_clause)
            p_ok = (.78 if clause else .5) - .15 * amb
            rows.append(dict(case_id=str(i), condition=cond, config=cond,
                             determination_correct=int(rng.random() < p_ok),
                             fabricated_citation=int(rng.random() < .05),
                             correct_clause_cited=int(rng.random() < .7), clause_in_context=clause,
                             consistent=int(rng.random() < .93), review_required=int(rng.random() < .6),
                             latency_s=float(rng.gamma(4, 2)), dispute_type=dt, narrative_words=words,
                             ambiguous=amb))
    return pd.DataFrame(rows)


def test_compare_to_control_and_benchmarks():
    from dispute_rag.analysis import benchmark_table, compare_to_control
    long = _long_frame()
    acc = compare_to_control(long, "determination_correct")
    assert set(acc["vs_control"]) == {"A", "B", "C"} and "cochran_q" in acc
    for v in acc["vs_control"].values():
        assert v["p_holm"] >= v["p_value"]
        lo, hi = v["difference_ci"]
        assert lo <= v["difference"] <= hi
    cla = compare_to_control(long, "correct_clause_cited")
    bt = benchmark_table(long, acc, cla, breakeven=0.47, cost_ok=True)
    assert set(bt["verdict"]) <= {"met", "not met", "inconclusive", "reported"}


def test_interval_verdicts():
    from dispute_rag.analysis import _interval_verdict
    assert _interval_verdict(.025, .010, .063, .05, "<=") == "inconclusive"
    assert _interval_verdict(.01, .002, .04, .05, "<=") == "met"
    assert _interval_verdict(.93, .91, .96, .90, ">=") == "met"
    assert _interval_verdict(.88, .82, .92, .90, ">=") == "not met"


def test_driver_analysis_recovers_clause_effect():
    from dispute_rag.drivers import driver_analysis
    long = _long_frame(seed=1)
    ret = long[long.condition.isin(["A", "B", "C"])].assign(config=lambda d: d.condition)
    out = driver_analysis(ret, n_boot=50)
    orr = out["odds_ratios"]
    assert orr.loc["clause_in_context", "odds_ratio"] > 1.5          # simulated true OR ~ 3.5
    assert orr.loc["clause_in_context", "p_value"] < .05
    assert out["n_complaints"] == 160 and out["n_observations"] == 480
    assert "share_via_clause" in out["ame"]["C"]


def test_cluster_logit_matches_known_solution():
    from dispute_rag.drivers import fit_logit_cluster
    rng = np.random.default_rng(3)
    X = np.column_stack([np.ones(4000), rng.normal(size=4000)])
    y = (rng.random(4000) < 1 / (1 + np.exp(-(0.5 + 1.0 * X[:, 1])))).astype(float)
    fit = fit_logit_cluster(y, X, np.arange(4000))
    assert abs(fit["params"][0] - 0.5) < 0.15 and abs(fit["params"][1] - 1.0) < 0.15


def test_dominant_failure_pools_when_few_errors():
    from dispute_rag.analysis import dominant_failure
    recs = pd.DataFrame({"case_id": [str(i) for i in range(60)], "condition": ["C"] * 20 + ["A"] * 40,
                         "category": ["retrieval_miss"] * 10 + ["correct"] * 10 + ["retrieval_miss"] * 25 + ["over_escalation"] * 15})
    d = dominant_failure(recs)
    assert d["pooled_A_B_C"] and d["category"] == "retrieval_miss" and d["n_errors"] == 50


def test_breakeven_closed_form_matches_grid():
    from dispute_rag.business_model import Assumptions, breakeven_accuracy, cost_per_case
    a = Assumptions()
    be = breakeven_accuracy(a)
    assert abs(cost_per_case(be, a)["minutes_saved_pct"]) < 1e-9
    a.review_share = 0.5
    assert breakeven_accuracy(a) < be
