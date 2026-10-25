"""End-to-end RAG pipeline: case -> query -> retrieve -> grounded, cited determination."""
from __future__ import annotations
import json, re, time
from .config import ExperimentConfig
from .corpus import build_chunks
from .llm import LLM
from .retriever import Retriever
from .schemas import DisputeCase, RAGAnswer

# The system prompt is IDENTICAL in every condition, including the no-retrieval control, so the
# only thing that differs between conditions is the EXCERPTS block. (v6) Earlier wording told the
# model to answer "ONLY from the provided excerpts" and to choose request_more_info/escalate when
# they were insufficient; with no excerpts that pushed the control towards over-escalation and
# would have confounded the retrieval effect with an instruction effect. The decision rule for
# request_more_info and escalate now depends on the complaint's facts, not on whether excerpts
# were supplied.
SYSTEM_PROMPT = """You are a decision-support assistant for a payments dispute investigator working under Regulation E.
You will receive a consumer dispute case and, when available, excerpts from regulations and guidance.
Rules:
1. When excerpts are provided, treat them as your authority and rely on them. When none are provided, rely on your own knowledge of the governing regulation.
2. Cite every rule you rely on using its section reference in square brackets, e.g. [1005.11(c)(1)].
3. Do not invent rules, section numbers, deadlines or amounts.
4. The narrative is a real consumer complaint with identifying details redacted as XXXX. Treat redactions as unknown facts, never as evidence, and do not guess what they concealed.
5. Choose "request_more_info" only when the complaint lacks facts needed to decide, and "escalate" only when the facts are sufficient but the decision exceeds first-line authority (for example a missed deadline or an unusually large amount). Do not choose either merely because excerpts are missing.
6. You recommend; the human investigator decides.
Respond ONLY with JSON:
{"determination": "<approve_dispute|deny_dispute|partial_credit|escalate|request_more_info>",
 "reasoning": "<3-6 sentences with bracketed citations>",
 "cited_sections": ["<section_ref>", ...]}"""


def case_to_query(case: DisputeCase) -> str:
    """Build the retrieval query from the published complaint.

    The company's disposition is deliberately excluded: it would leak the outcome into
    retrieval and inflate every downstream metric.
    """
    c = case.complaint
    return (f"Dispute type: {case.dispute_type}. Product: {c.product}"
            f"{f' / {c.sub_product}' if c.sub_product else ''}. "
            f"Issue: {c.issue}{f' — {c.sub_issue}' if c.sub_issue else ''}. "
            f"Consumer narrative: {case.narrative}")


def format_context(chunks) -> str:
    return "\n\n".join(f"[{c.section_ref or c.doc_id}] {c.text}" for c in chunks)


def build_user_message(query: str, retrieved) -> str:
    excerpts = format_context(retrieved) if retrieved else "(none provided)"
    return f"CASE:\n{query}\n\nEXCERPTS:\n{excerpts}"


def parse_response(raw: str) -> tuple[dict, bool]:
    """Extract the JSON object from a response; tolerate ```json fences and surrounding prose.
    Returns (parsed, ok). An unparseable response is kept and scored as incorrect."""
    text = re.sub(r"```(?:json)?", "", raw or "")
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}, False
    try:
        obj = json.loads(m.group(0))
        return (obj, True) if isinstance(obj, dict) else ({}, False)
    except json.JSONDecodeError:
        return {}, False


class RAGPipeline:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.chunks = build_chunks(cfg.corpus_dir, cfg.chunking)
        self.retriever = Retriever(self.chunks, cfg.retrieval)
        self.llm = LLM(temperature=cfg.generation.temperature, max_tokens=cfg.generation.max_tokens)

    def answer(self, case: DisputeCase, run_idx: int = 0) -> RAGAnswer:
        t0 = time.perf_counter()
        query = case_to_query(case)
        retrieved = self.retriever.retrieve(query)
        raw = self.llm.complete(SYSTEM_PROMPT, build_user_message(query, retrieved))
        latency = time.perf_counter() - t0
        parsed, ok = parse_response(raw)
        refs: list[str] = []
        for c in retrieved:
            for r in (c.section_refs or ([c.section_ref] if c.section_ref else [])):
                if r not in refs:
                    refs.append(r)
        return RAGAnswer(
            case_id=case.case_id, config_name=self.cfg.name, run_idx=run_idx,
            retrieved_chunk_ids=[c.chunk_id for c in retrieved],
            retrieved_section_refs=refs,
            determination=parsed.get("determination"),
            raw_determination=None if parsed.get("determination") is None else str(parsed.get("determination")),
            reasoning=str(parsed.get("reasoning", "")),
            cited_section_refs=parsed.get("cited_sections", []),
            latency_s=latency,
            raw_response=raw,
            model=self.llm.last_model,
            parse_ok=ok,
        )
