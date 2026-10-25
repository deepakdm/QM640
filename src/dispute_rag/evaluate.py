"""Per-case scoring on the three evaluation pillars.

Accuracy        determination_correct (RQ1, RQ3)
Groundedness    fabricated_citation, correct_clause_cited (RQ2, primary, deterministic);
                citation_validity and judged claim support (RQ2, secondary)
Operational     consistent, review_required, latency (RQ4)
Mechanism       clause_in_context (RQ3 mediator), retrieval recall/precision@k (descriptive)

Primary outcomes use the first of the three runs; consistency uses all three. Reference matching
rules are documented in refs.py.
"""
from __future__ import annotations
import json, random, re
from collections import Counter
import numpy as np
import pandas as pd
from .llm import LLM
from .refs import classify_citations, cites_gold, clause_in_context, normalize_ref, related
from .schemas import GoldLabel, RAGAnswer

JUDGE_SYSTEM = """You are auditing an AI assistant's answer for groundedness.
Given the EXCERPTS the assistant saw and its REASONING, list each factual or rule-based claim in the
reasoning and mark it SUPPORTED (directly backed by the excerpts) or UNSUPPORTED.
Respond ONLY with JSON: {"claims":[{"claim":"...","status":"SUPPORTED|UNSUPPORTED"}]}"""

# Drafts routed to mandatory human review (synopsis, "Human review and the decision-support
# boundary"). The amount-authority rule cannot be applied because CFPB narratives carry no
# structured amount; it is listed as a limitation.
REVIEW_DETERMINATIONS = {"deny_dispute", "partial_credit", "escalate", "request_more_info"}


def retrieval_metrics(ans: RAGAnswer, gold: GoldLabel) -> dict:
    """Hierarchical recall/precision@k of governing clauses among the retrieved references."""
    retrieved = [n for n in (normalize_ref(r) for r in ans.retrieved_section_refs) if n]
    truth = [n for n in (normalize_ref(r) for r in gold.governing_clauses) if n]
    if not ans.retrieved_chunk_ids:  # control condition: no retrieval took place
        return {"recall_at_k": np.nan, "precision_at_k": np.nan, "any_hit": np.nan}
    hit_truth = sum(any(related(r, t) for r in retrieved) for t in truth)
    hit_retrieved = sum(any(related(r, t) for t in truth) for r in retrieved)
    return {"recall_at_k": hit_truth / len(truth) if truth else np.nan,
            "precision_at_k": hit_retrieved / len(retrieved) if retrieved else 0.0,
            "any_hit": int(hit_truth > 0)}


def determination_correct(ans: RAGAnswer, gold: GoldLabel) -> int:
    return int(ans.determination is not None and ans.determination == gold.determination)


def citation_validity(ans: RAGAnswer) -> float:
    """Share of the answer's citations that were present in its retrieved context (secondary
    RQ2 measure). Undefined for the control, which receives no context."""
    if not ans.cited_section_refs or not ans.retrieved_chunk_ids:
        return np.nan
    retrieved = [n for n in (normalize_ref(r) for r in ans.retrieved_section_refs) if n]
    cited = [normalize_ref(c) for c in ans.cited_section_refs]
    return float(np.mean([bool(c) and any(related(c, r) for r in retrieved) for c in cited]))


def grounding_flags(ans: RAGAnswer, gold: GoldLabel, corpus_index: set[str]) -> dict:
    cls = classify_citations(ans.cited_section_refs, corpus_index)
    return {
        "n_citations": len(ans.cited_section_refs),
        "fabricated_citation": int(bool(cls["fabricated"])),
        "fabricated_refs": "|".join(cls["fabricated"]),
        "unverifiable_citation": int(bool(cls["unverifiable"])),
        "correct_clause_cited": int(cites_gold(ans.cited_section_refs, gold.governing_clauses)),
        "clause_in_context": (int(clause_in_context(ans.retrieved_section_refs, gold.governing_clauses))
                              if ans.retrieved_chunk_ids else np.nan),
    }


def judge_support(ans: RAGAnswer, context_text: str, judge: LLM) -> float:
    """Share of claims in the reasoning that the judge marks SUPPORTED (secondary RQ2)."""
    raw = judge.complete(JUDGE_SYSTEM, f"EXCERPTS:\n{context_text}\n\nREASONING:\n{ans.reasoning}")
    m = re.search(r"\{.*\}", raw or "", re.S)
    try:
        claims = json.loads(m.group(0))["claims"]
    except Exception:
        return np.nan
    if not claims:
        return np.nan
    return float(np.mean([str(c.get("status", "")).upper() == "SUPPORTED" for c in claims]))


def consistency(answers_for_case: list[RAGAnswer]) -> float:
    """Share of runs agreeing with the modal determination (1.0 = all agree)."""
    dets = [a.determination for a in answers_for_case]
    return Counter(dets).most_common(1)[0][1] / len(dets)


def review_required(first: RAGAnswer, flags: dict, consistent: int) -> int:
    return int(first.determination is None
               or first.determination in REVIEW_DETERMINATIONS
               or flags["fabricated_citation"] == 1
               or consistent == 0)


def score_case(runs: list[RAGAnswer], gold: GoldLabel, corpus_index: set[str]) -> dict:
    runs = sorted(runs, key=lambda a: a.run_idx)
    first = runs[0]
    flags = grounding_flags(first, gold, corpus_index)
    cons = consistency(runs)
    consistent = int(cons == 1.0) if len(runs) > 1 else np.nan
    return {
        "case_id": first.case_id, "config": first.config_name, "n_runs": len(runs),
        "gold_determination": gold.determination,
        "pred_determination": first.determination,
        "unparseable": int(first.determination is None),
        "determination_correct": determination_correct(first, gold),
        **retrieval_metrics(first, gold),
        **flags,
        "citation_validity": citation_validity(first),
        "consistency": cons,
        "consistent": consistent,
        "review_required": review_required(first, flags, int(cons == 1.0) if len(runs) > 1 else 1),
        "latency_s": float(first.latency_s),
        "latency_median_s": float(np.median([x.latency_s for x in runs])),
        "judged_support": np.nan,
        "model": first.model,
    }


def summarize_run(answers: list[RAGAnswer], gold: dict[str, GoldLabel], corpus_index: set[str],
                  context_lookup: dict[str, str] | None = None, judge: LLM | None = None,
                  judge_sample: int = 0, seed: int = 42) -> pd.DataFrame:
    by_case: dict[str, list[RAGAnswer]] = {}
    for a in answers:
        by_case.setdefault(a.case_id, []).append(a)
    ids = sorted(c for c in by_case if c in gold)
    sample_ids = (set(random.Random(seed).sample(ids, min(judge_sample, len(ids))))
                  if judge and judge_sample else set())
    rows = []
    for cid in ids:
        r = score_case(by_case[cid], gold[cid], corpus_index)
        if cid in sample_ids and context_lookup and context_lookup.get(cid):
            first = min(by_case[cid], key=lambda a: a.run_idx)
            r["judged_support"] = judge_support(first, context_lookup[cid], judge)
        rows.append(r)
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.Series:
    return pd.Series({
        "n_cases": len(df),
        "determination_accuracy": df.determination_correct.mean(),
        "unparseable_rate": df.unparseable.mean(),
        "fabricated_citation_rate": df.fabricated_citation.mean(),
        "correct_clause_rate": df.correct_clause_cited.mean(),
        "unverifiable_citation_rate": df.unverifiable_citation.mean(),
        "clause_in_context_rate": df.clause_in_context.mean(),
        "recall_at_k": df.recall_at_k.mean(),
        "precision_at_k": df.precision_at_k.mean(),
        "citation_validity": df.citation_validity.mean(),
        "judged_support_(sample)": df.judged_support.mean(),
        "consistent_rate": df.consistent.mean(),
        "review_required_rate": df.review_required.mean(),
        "latency_median_s": df.latency_s.median(),
        "latency_p95_s": df.latency_s.quantile(0.95),
    })
