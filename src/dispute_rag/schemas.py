"""Shared data models. Keeping these explicit makes the gold set and outputs auditable."""
from __future__ import annotations
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

Determination = Literal["approve_dispute", "deny_dispute", "partial_credit", "escalate", "request_more_info"]
DETERMINATIONS = ("approve_dispute", "deny_dispute", "partial_credit", "escalate", "request_more_info")

# Obvious spelling variants a model may return. Anything else is recorded as unparseable (None),
# which the analysis scores as incorrect — it is never guessed into a class.
_DETERMINATION_ALIASES = {
    "approve": "approve_dispute", "approved": "approve_dispute",
    "deny": "deny_dispute", "denied": "deny_dispute",
    "partial": "partial_credit", "partial_approval": "partial_credit",
    "request_more_information": "request_more_info", "request_info": "request_more_info",
    "escalated": "escalate",
}


def normalize_determination(value) -> Optional[str]:
    if value is None:
        return None
    key = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    key = _DETERMINATION_ALIASES.get(key, key)
    return key if key in DETERMINATIONS else None


class Chunk(BaseModel):
    chunk_id: str                  # e.g. "reg_e__1005.11__c__3"
    doc_id: str                    # e.g. "reg_e"
    section_ref: Optional[str]     # e.g. "1005.11(c)" — first reference in the chunk
    section_refs: list[str] = Field(default_factory=list)  # every reference whose text is inside
    text: str


CompanyResponse = Literal[
    "Closed with monetary relief", "Closed with non-monetary relief",
    "Closed with explanation", "Closed without relief", "Closed with relief",
    "Closed", "In progress", "Untimely response",
]


class CFPBComplaint(BaseModel):
    """One published record from the CFPB Consumer Complaint Database (real, public data).

    Field names follow the CCDB codebook: https://cfpb.github.io/api/ccdb/fields.html
    """
    complaint_id: str
    date_received: Optional[str] = None
    product: str
    sub_product: Optional[str] = None
    issue: str
    sub_issue: Optional[str] = None
    company: Optional[str] = None
    state: Optional[str] = None
    submitted_via: Optional[str] = None
    # Company's own disposition. This is a WEAK, real-world reference only: it records what the
    # firm did, not what Reg E / Reg Z require. Never used as ground truth — see gold.py.
    company_response: Optional[str] = None
    timely_response: Optional[str] = None
    extra: dict = Field(default_factory=dict)


class DisputeCase(BaseModel):
    case_id: str                         # == complaint_id, so results trace back to the public record
    complaint: CFPBComplaint
    dispute_type: str                    # normalized from issue/sub-issue, e.g. "unauthorized_eft"
    narrative: str                       # consumer's own words, PII redacted as XXXX by the CFPB
    regulation_scope: Literal["reg_e", "reg_z", "either"] = "reg_e"
    source: Literal["cfpb_ccdb"] = "cfpb_ccdb"


class GoldLabel(BaseModel):
    """Human-assigned ground truth. The rater reads the corpus and decides; the company's own
    disposition is shown as context but is explicitly NOT the label."""
    case_id: str
    governing_clauses: list[str]         # section_refs that decide the case, e.g. ["1005.6(b)(1)"]
    determination: Determination
    rationale: str                       # one-sentence human rationale citing the clause
    labeled_by: str
    second_rater_determination: Optional[Determination] = None
    second_rater_clauses: list[str] = Field(default_factory=list)
    ambiguous: bool = False              # insufficient facts in the published narrative

    @field_validator("determination", "second_rater_determination", mode="before")
    @classmethod
    def _coerce_determination(cls, v):
        if v in (None, ""):
            return None
        n = normalize_determination(v)
        if n is None:
            raise ValueError(f"unknown determination {v!r}; use one of {DETERMINATIONS}")
        return n


class RAGAnswer(BaseModel):
    case_id: str
    config_name: str
    run_idx: int
    retrieved_chunk_ids: list[str]
    retrieved_section_refs: list[str]
    determination: Optional[Determination]
    reasoning: str
    cited_section_refs: list[str]
    latency_s: float
    raw_response: str
    raw_determination: Optional[str] = None   # exactly what the model returned, for audit
    model: Optional[str] = None               # provider model string, recorded per response
    parse_ok: bool = True                     # False if the response was not valid JSON

    @field_validator("determination", mode="before")
    @classmethod
    def _coerce_determination(cls, v):
        return normalize_determination(v)

    @field_validator("cited_section_refs", mode="before")
    @classmethod
    def _coerce_citations(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [str(x) for x in v]
