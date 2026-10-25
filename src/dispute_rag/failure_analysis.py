"""Failure categories for incorrect determinations (RQ4).

Each incorrect first-run determination is assigned to the FIRST matching category, in this order
(synopsis, RQ4 method):

  1. unparseable                   no valid determination could be read from the response
  2. fabricated_citation           cites a section that does not exist in the corpus index
  3. retrieval_miss                no governing clause reached the model (retrieval conditions);
                                   in the no-retrieval control, the model did not cite a governing
                                   clause ("clause_miss"), since nothing was retrieved
  4. over_escalation               escalate / request_more_info where the gold label decides the case
  5. right_clause_wrong_reasoning  the governing clause was available but the outcome is wrong
"""
from __future__ import annotations
import pandas as pd
from .evaluate import grounding_flags
from .schemas import GoldLabel, RAGAnswer

CATEGORIES = ["unparseable", "fabricated_citation", "retrieval_miss", "clause_miss",
              "over_escalation", "right_clause_wrong_reasoning"]
DECISIVE = {"approve_dispute", "deny_dispute", "partial_credit"}


def classify(ans: RAGAnswer, gold: GoldLabel, corpus_index: set[str]) -> str:
    if ans.determination is None:
        return "unparseable"
    if ans.determination == gold.determination:
        return "correct"
    f = grounding_flags(ans, gold, corpus_index)
    if f["fabricated_citation"]:
        return "fabricated_citation"
    if ans.retrieved_chunk_ids:
        if not f["clause_in_context"]:
            return "retrieval_miss"
    elif not f["correct_clause_cited"]:
        return "clause_miss"
    if ans.determination in ("escalate", "request_more_info") and gold.determination in DECISIVE:
        return "over_escalation"
    return "right_clause_wrong_reasoning"


def failure_records(answers: list[RAGAnswer], gold: dict[str, GoldLabel],
                    corpus_index: set[str]) -> pd.DataFrame:
    first_runs = {a.case_id: a for a in answers if a.run_idx == 0}
    return pd.DataFrame([{"case_id": cid, "config": a.config_name,
                          "category": classify(a, gold[cid], corpus_index)}
                         for cid, a in first_runs.items() if cid in gold])


def failure_table(answers: list[RAGAnswer], gold: dict[str, GoldLabel],
                  corpus_index: set[str]) -> pd.DataFrame:
    df = failure_records(answers, gold, corpus_index)
    if df.empty:
        return df
    return (df.groupby(["config", "category"]).size().unstack(fill_value=0)
              .reindex(columns=["correct"] + CATEGORIES, fill_value=0))
