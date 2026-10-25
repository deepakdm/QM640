"""Offline smoke tests (no API, no model downloads, no raw data).

Covers the logic that is easy to break and hard to notice: chunking and section references,
CFPB filtering and classification, the evaluation metrics, the inferential tests, and the
business model. Run with `pytest tests -q` after any edit.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from dispute_rag.business_model import breakeven_accuracy, sensitivity
from dispute_rag.cfpb_cases import classify_dispute_type, sample_cases, scope_for_product, to_cases
from dispute_rag.config import ChunkingCfg, ExperimentConfig
from dispute_rag.corpus import chunk_fixed, chunk_section
from dispute_rag.evaluate import citation_validity, retrieval_metrics
from dispute_rag.schemas import GoldLabel, RAGAnswer
from dispute_rag.stats import cochrans_q, compare_configs, holm_correction, mcnemar_exact

SAMPLE_CFR = """§ 1005.6 Liability of consumer for unauthorized transfers.
(a) Conditions for liability. A consumer may be held liable only if ...
(b) Limitations on amount of liability. A consumer's liability shall not exceed ...
§ 1005.11 Procedures for resolving errors.
(a) Definition of error. The term error means ...
(c) Time limits and extent of investigation. ...
§ 1026.13 Billing error resolution.
(a) Billing error. For purposes of this section, the term billing error means ...
"""

SAMPLE_NETWORK = """Condition 13.1 Merchandise/Services Not Received.
The cardholder did not receive the merchandise or services ...
Reason Code 10.4 Other Fraud - Card-Absent Environment.
The issuer received a cardholder letter stating ...
"""


def test_configs_load():
    for p in Path("configs").glob("*.yaml"):
        assert ExperimentConfig.load(p).name


def test_section_chunking_keeps_cfr_refs():
    chunks = chunk_section("reg_e", SAMPLE_CFR, ChunkingCfg(strategy="section"))
    refs = {c.section_ref for c in chunks}
    assert "1005.6(a)" in refs and "1005.11(c)" in refs


def test_section_chunking_handles_reg_z():
    chunks = chunk_section("reg_z", SAMPLE_CFR, ChunkingCfg(strategy="section"))
    assert "1026.13(a)" in {c.section_ref for c in chunks}


def test_section_chunking_handles_network_headings():
    chunks = chunk_section("network_rules", SAMPLE_NETWORK, ChunkingCfg(strategy="section"))
    refs = {c.section_ref for c in chunks}
    assert "13.1" in refs and "10.4" in refs


def test_fixed_chunking_overlaps():
    chunks = chunk_fixed("reg_e", SAMPLE_CFR * 20, ChunkingCfg(chunk_size=64, overlap=16))
    assert len(chunks) > 2


def test_scope_for_product():
    assert scope_for_product("Checking or savings account") == "reg_e"
    assert scope_for_product("Credit card or prepaid card") == "reg_z"
    assert scope_for_product("Mortgage") is None


def test_classify_dispute_type():
    assert classify_dispute_type("Problem caused by your funds being low",
                                 "Unauthorized transactions") == "unauthorized_transaction"
    assert classify_dispute_type("Managing an account",
                                 "Banking errors") == "error_resolution"
    assert classify_dispute_type("Mortgage servicing", None) is None


def _candidate_frame(n=12):
    return pd.DataFrame({
        "complaint_id": [str(1000 + i) for i in range(n)],
        "date_received": ["2024-01-0{}".format(i % 9 + 1) for i in range(n)],
        "product": ["Checking or savings account"] * n,
        "sub_product": ["Checking account"] * n,
        "issue": ["Managing an account"] * n,
        "sub_issue": ["Banking errors"] * (n // 2) + ["Unauthorized transactions"] * (n - n // 2),
        "narrative": ["word " * 60] * n,
        "company": ["BANK X"] * n,
        "state": ["MI"] * n,
        "submitted_via": ["Web"] * n,
        "company_response": ["Closed with explanation"] * n,
        "timely_response": ["Yes"] * n,
        "company_public_response": [""] * n,
        "scope": ["reg_e"] * n,
        "dispute_type": ["error_resolution"] * (n // 2) + ["unauthorized_transaction"] * (n - n // 2),
    })


def test_sample_is_stratified_and_deterministic():
    df = _candidate_frame()
    a = sample_cases(df, n=6, seed=7)
    b = sample_cases(df, n=6, seed=7)
    assert list(a.complaint_id) == list(b.complaint_id)
    assert a.dispute_type.nunique() == 2


def test_to_cases_round_trip():
    cases = to_cases(_candidate_frame(4))
    assert len(cases) == 4
    c = cases[0]
    assert c.case_id == c.complaint.complaint_id
    assert c.source == "cfpb_ccdb" and c.regulation_scope == "reg_e"


def test_metrics():
    a = RAGAnswer(case_id="c", config_name="x", run_idx=0, retrieved_chunk_ids=["1"],
                  retrieved_section_refs=["1005.6(b)", "1005.11(a)"], determination="deny_dispute",
                  reasoning="", cited_section_refs=["1005.6(b)", "9999.1"], latency_s=1.0,
                  raw_response="")
    g = GoldLabel(case_id="c", governing_clauses=["1005.6(b)"], determination="deny_dispute",
                  rationale="", labeled_by="me")
    m = retrieval_metrics(a, g)
    assert m["recall_at_k"] == 1.0 and m["precision_at_k"] == 0.5
    assert citation_validity(a) == 0.5


def test_mcnemar_and_cochran():
    a = [1, 1, 0, 0, 1, 0, 1, 1]
    b = [1, 0, 0, 1, 1, 0, 0, 1]
    r = mcnemar_exact(a, b)
    assert r["n10"] == 2 and r["n01"] == 1 and 0 <= r["p_value"] <= 1

    mat = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0], [1, 1, 1], [1, 0, 1], [0, 1, 0]])
    q = cochrans_q(mat)
    assert q["df"] == 2 and 0 <= q["p_value"] <= 1


def test_cochran_handles_identical_configs():
    mat = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]])
    assert cochrans_q(mat)["df"] == 2


def test_holm_is_monotone_and_bounded():
    adj = holm_correction({"a": 0.01, "b": 0.04, "c": 0.5})
    assert adj["a"] <= adj["b"] <= adj["c"] <= 1.0


def test_compare_configs_end_to_end():
    ids = [f"case_{i}" for i in range(20)]
    frames = {
        "cfg_a": pd.DataFrame({"case_id": ids,
                               "determination_correct": [1] * 15 + [0] * 5,
                               "any_hit": [1] * 18 + [0] * 2,
                               "latency_median_s": np.linspace(1, 2, 20)}),
        "cfg_b": pd.DataFrame({"case_id": ids,
                               "determination_correct": [1] * 10 + [0] * 10,
                               "any_hit": [1] * 14 + [0] * 6,
                               "latency_median_s": np.linspace(2, 3, 20)}),
    }
    out = compare_configs(frames)
    assert out["n_cases"] == 20
    lo, hi = out["per_config"]["cfg_a"]["bootstrap_ci"]
    assert lo <= out["per_config"]["cfg_a"]["estimate"] <= hi
    assert "cfg_a vs cfg_b" in out["pairwise_mcnemar"]


def test_business_model():
    assert 0 < breakeven_accuracy() < 1
    assert len(sensitivity()) == 11


# --- classification metrics (research-design guide Step 4.1-4.2) ----------------------

def test_confusion_matrix_totals_preserved():
    from dispute_rag.classification import confusion_matrix
    yt = ["approve_dispute", "deny_dispute", "escalate", "deny_dispute"]
    yp = ["approve_dispute", "approve_dispute", None, "deny_dispute"]
    m = confusion_matrix(yt, yp)
    assert m.to_numpy().sum() == len(yt)          # nothing silently dropped
    assert m.loc["escalate", "no_answer"] == 1     # unparseable answers are counted


def test_per_class_metrics_macro_and_weighted():
    from dispute_rag.classification import per_class_metrics
    yt = ["approve_dispute"] * 3 + ["deny_dispute"] * 3
    yp = ["approve_dispute"] * 3 + ["deny_dispute"] * 2 + ["approve_dispute"]
    df = per_class_metrics(yt, yp)
    assert df.loc["deny_dispute", "recall"] == 2 / 3
    assert df.loc["deny_dispute", "precision"] == 1.0
    assert "macro_avg" in df.index and "weighted_avg" in df.index


def test_kfold_stability_shape():
    from dispute_rag.classification import kfold_stability
    ks = kfold_stability([1] * 30 + [0] * 20, k=5)
    assert ks["k"] == 5 and len(ks["fold_scores"]) == 5
    assert ks["min"] <= ks["mean"] <= ks["max"]
    assert sum(ks["fold_sizes"]) == 50


def test_kfold_rejects_tiny_sets():
    from dispute_rag.classification import kfold_stability
    assert "error" in kfold_stability([1, 0], k=5)


def test_roc_auc_returns_none_without_scores():
    from dispute_rag.classification import roc_auc_escalation
    assert roc_auc_escalation(["escalate", "deny_dispute"], []) is None


# --- control condition ----------------------------------------------------------------

def test_control_config_retrieves_nothing():
    from dispute_rag.config import ExperimentConfig
    from dispute_rag.retriever import Retriever
    from dispute_rag.schemas import Chunk
    cfg = ExperimentConfig.load("configs/control_no_retrieval.yaml")
    assert cfg.retrieval.mode == "none"
    r = Retriever([Chunk(chunk_id="c", doc_id="d", section_ref="1005.6", text="x")], cfg.retrieval)
    assert r.retrieve("anything") == []
