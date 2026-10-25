"""Load real dispute cases from the CFPB Consumer Complaint Database (CCDB).

Source:    https://www.consumerfinance.gov/data-research/consumer-complaints/
Bulk CSV:  https://files.consumerfinance.gov/ccdb/complaints.csv.zip
Codebook:  https://cfpb.github.io/api/ccdb/fields.html

This module replaces the earlier Kaggle/synthetic data layer. The QM 640 syllabus states that
"No project will be approved based on fictitious data (i.e. Kaggle data)", so every case here is a
real, published consumer complaint. Narratives are the consumer's own words with PII redacted as
XXXX by the CFPB before publication.

Two provenance facts belong in the methods section and are printed by `filter_report`:
  1. Complaints are published only after the company responds (confirming a commercial
     relationship) or after 15 days, whichever comes first.
  2. Complaints referred to other regulators (e.g. depository institutions under $10B in assets)
     are never published. The published set is therefore not a census of all disputes.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd

from .schemas import CFPBComplaint, DisputeCase

# CCDB ships under two column conventions: the bulk CSV uses display names, the API uses
# snake_case. Accept either so the loader works with a browser export or an API pull.
COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "complaint_id": ("Complaint ID", "complaint_id"),
    "date_received": ("Date received", "date_received"),
    "product": ("Product", "product"),
    "sub_product": ("Sub-product", "sub_product"),
    "issue": ("Issue", "issue"),
    "sub_issue": ("Sub-issue", "sub_issue"),
    "narrative": ("Consumer complaint narrative", "complaint_what_happened"),
    "company": ("Company", "company"),
    "state": ("State", "state"),
    "submitted_via": ("Submitted via", "submitted_via"),
    "company_response": ("Company response to consumer", "company_response"),
    "timely_response": ("Timely response?", "timely"),
    "company_public_response": ("Company public response", "company_public_response"),
}

# Products in scope. Reg E governs consumer EFTs on asset accounts; Reg Z governs credit-card
# billing disputes. Labels changed in 2017 and 2023, so both old and current names are listed.
REG_E_PRODUCTS = {
    "checking or savings account",
    "bank account or service",
    "prepaid card",
    "money transfer, virtual currency, or money service",
    "money transfers",
}
REG_Z_PRODUCTS = {
    "credit card or prepaid card",
    "credit card",
}

# Issue text -> normalized dispute_type, checked in order. Matching is on the combined
# issue + sub-issue string because CFPB moved detail between the two fields over time.
DISPUTE_TYPE_PATTERNS: list[tuple[str, str]] = [
    (r"unauthorized|not authorized|didn'?t authorize|fraudulent|did not make|didn'?t make", "unauthorized_transaction"),
    (r"did not (purchase|receive)|never received|not received", "goods_not_received"),
    (r"charged.*(twice|more than once)|duplicate", "duplicate_charge"),
    (r"wrong day|wrong amount|incorrect amount", "incorrect_amount"),
    (r"cancel|subscription|recurring|stop payment", "cancelled_recurring"),
    # "Banking errors" and "Problem using a debit or ATM card" are the live CCDB sub-issues that
    # map to Reg E error resolution; plain \berror\b catches them and the older label wording.
    (r"\berrors?\b|dispute|investigat|resolve", "error_resolution"),
    (r"purchase shown on your statement|billing", "billing_dispute"),
]

_MIN_NARRATIVE_WORDS = 40   # short narratives rarely carry enough facts to label a clause
_MAX_NARRATIVE_WORDS = 600  # keeps prompt size and cost predictable


def _resolve_columns(header: list[str]) -> dict[str, str]:
    """Map canonical field names to whichever column spelling this file uses."""
    present = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in header:
                present[canonical] = alias
                break
    missing = {"complaint_id", "product", "issue", "narrative"} - set(present)
    if missing:
        raise ValueError(
            f"CCDB file is missing required columns {sorted(missing)}. "
            f"Header seen: {header[:20]}"
        )
    return present


# CCDB issues that look like disputes to the keyword rules above but are not Regulation E error-resolution
# questions. Found by reading a real draw of 160 cases: all 36 "Funds not received from closed account"
# complaints were account-closure payouts (matched "not received"), all 6 "cancel" matches were credit-monitoring
# subscriptions, and 2 were credit-report investigations. None has a Regulation E clause that decides it.
OUT_OF_SCOPE_ISSUE = re.compile(
    r"closing an account|closed account|credit monitoring|identity theft protection|credit report|"
    r"investigation into an existing problem"
)


def classify_dispute_type(issue: str, sub_issue: str | None) -> Optional[str]:
    text = f"{issue or ''} {sub_issue or ''}".lower()
    if OUT_OF_SCOPE_ISSUE.search(text):
        return None
    for pattern, label in DISPUTE_TYPE_PATTERNS:
        if re.search(pattern, text):
            return label
    return None


def scope_for_product(product: str) -> Optional[str]:
    p = (product or "").strip().lower()
    if p in REG_E_PRODUCTS:
        return "reg_e"
    if p in REG_Z_PRODUCTS:
        return "reg_z"
    return None


# Study window (synopsis, Scope and Boundaries): complaints received in the two years up to the
# download date. Override with --start/--end in 01_prepare_data.py.
DEFAULT_START = "2024-09-18"
DEFAULT_END = "2026-08-14"   # CFPB stopped publishing narratives on 2026-08-14

FUNNEL_STEPS = [
    ("rows_read", "Rows in the export"),
    ("narrative_present", "Consumer narrative present and non-empty"),
    ("in_date_window", "Received within the study window"),
    ("in_scope_product", "Product governed by Regulation E"),
    ("dispute_type_mapped", "Issue / sub-issue mapped to one of seven dispute types"),
    ("length_40_600", "Narrative length 40-600 words"),
    ("unique_complaint_id", "Duplicate complaint IDs removed"),
    ("unique_narrative", "Exact duplicate narratives removed"),
]


def word_count(s: pd.Series) -> pd.Series:
    return s.fillna("").str.split().str.len()


def parse_dates(s: pd.Series) -> pd.Series:
    """CCDB browser exports use MM/DD/YY, the bulk file and API use YYYY-MM-DD."""
    return pd.to_datetime(s, errors="coerce", format="mixed")


def stream_candidates(
    csv_path: Path,
    include_reg_z: bool = False,
    chunksize: int = 200_000,
    start: str | None = DEFAULT_START,
    end: str | None = DEFAULT_END,
    funnel: dict | None = None,
) -> Iterator[pd.DataFrame]:
    """Read the CSV in chunks and yield only in-scope rows with usable narratives.

    The full download is several GB, so it is never loaded whole. Each chunk is filtered
    immediately and discarded. If `funnel` is a dict, the number of rows surviving each step is
    accumulated in it (the data-cleaning log in the report is built from these counts).
    """
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    cols = _resolve_columns(header)
    usecols = list(cols.values())
    f = funnel if funnel is not None else {}
    for key, _ in FUNNEL_STEPS[:6]:
        f.setdefault(key, 0)
    t0 = pd.Timestamp(start) if start else None
    t1 = pd.Timestamp(end) if end else None

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, dtype=str,
                             low_memory=False):
        chunk = chunk.rename(columns={v: k for k, v in cols.items()})
        f["rows_read"] += len(chunk)
        chunk = chunk[chunk["narrative"].fillna("").str.strip() != ""]
        f["narrative_present"] += len(chunk)
        if (t0 is not None or t1 is not None) and "date_received" in chunk:
            d = parse_dates(chunk["date_received"])
            keep = d.notna()
            if t0 is not None:
                keep &= d >= t0
            if t1 is not None:
                keep &= d <= t1
            chunk = chunk[keep]
        f["in_date_window"] += len(chunk)
        chunk = chunk.assign(scope=chunk["product"].map(scope_for_product))
        allowed = {"reg_e", "reg_z"} if include_reg_z else {"reg_e"}
        chunk = chunk[chunk["scope"].isin(allowed)]
        f["in_scope_product"] += len(chunk)
        if chunk.empty:
            continue
        sub = chunk["sub_issue"] if "sub_issue" in chunk else pd.Series([None] * len(chunk), index=chunk.index)
        chunk = chunk.assign(dispute_type=[classify_dispute_type(i, s) for i, s in zip(chunk["issue"], sub)])
        chunk = chunk[chunk["dispute_type"].notna()]
        f["dispute_type_mapped"] += len(chunk)
        wc = word_count(chunk["narrative"])
        chunk = chunk[(wc >= _MIN_NARRATIVE_WORDS) & (wc <= _MAX_NARRATIVE_WORDS)]
        f["length_40_600"] += len(chunk)
        if not chunk.empty:
            yield chunk


def deduplicate(df: pd.DataFrame, funnel: dict | None = None) -> pd.DataFrame:
    """Drop repeated complaint IDs (keep first), then exact duplicate narratives after
    whitespace normalization — the same story filed twice would otherwise count twice."""
    f = funnel if funnel is not None else {}
    df = df.drop_duplicates(subset="complaint_id", keep="first")
    f["unique_complaint_id"] = len(df)
    norm = df["narrative"].fillna("").str.split().str.join(" ").str.lower()
    df = df[~norm.duplicated(keep="first")]
    f["unique_narrative"] = len(df)
    return df.reset_index(drop=True)


def load_candidates(csv_path: Path, include_reg_z: bool = False,
                    start: str | None = DEFAULT_START, end: str | None = DEFAULT_END,
                    funnel: dict | None = None) -> pd.DataFrame:
    parts = list(stream_candidates(csv_path, include_reg_z, start=start, end=end, funnel=funnel))
    if not parts:
        raise ValueError(
            "No in-scope complaints found. Check that the export includes narratives "
            "(filter 'consumer complaint narrative' when exporting), an in-scope product, and "
            "dates inside the study window."
        )
    return deduplicate(pd.concat(parts, ignore_index=True), funnel)


def sample_cases(df: pd.DataFrame, n: int = 160, seed: int = 42) -> pd.DataFrame:
    """Stratified sample across dispute_type so no single issue dominates the gold set."""
    types = df["dispute_type"].unique()
    per_type = max(1, n // len(types))
    picked = [
        g.sample(min(per_type, len(g)), random_state=seed)
        for _, g in df.groupby("dispute_type", sort=True)
    ]
    out = pd.concat(picked, ignore_index=True)
    if len(out) < n:  # backfill from the remainder to reach n
        rest = df[~df["complaint_id"].isin(out["complaint_id"])]
        if len(rest):
            out = pd.concat([out, rest.sample(min(n - len(out), len(rest)), random_state=seed)],
                            ignore_index=True)
    return out.sample(min(n, len(out)), random_state=seed).reset_index(drop=True)


def _s(v) -> Optional[str]:
    """Blank CSV cells arrive as float NaN; the schema wants a string or None."""
    if v is None or (isinstance(v, float) and v != v):
        return None
    v = str(v).strip()
    return v or None


def to_cases(df: pd.DataFrame) -> list[DisputeCase]:
    cases = []
    for _, r in df.iterrows():
        complaint = CFPBComplaint(
            complaint_id=str(r["complaint_id"]),
            date_received=_s(r.get("date_received")),
            product=str(r["product"]),
            sub_product=_s(r.get("sub_product")),
            issue=str(r["issue"]),
            sub_issue=_s(r.get("sub_issue")),
            company=_s(r.get("company")),
            state=_s(r.get("state")),
            submitted_via=_s(r.get("submitted_via")),
            company_response=_s(r.get("company_response")),
            timely_response=_s(r.get("timely_response")),
            extra={"company_public_response": _s(r.get("company_public_response"))},
        )
        cases.append(DisputeCase(
            case_id=str(r["complaint_id"]),
            complaint=complaint,
            dispute_type=r["dispute_type"],
            narrative=" ".join(str(r["narrative"]).split()),
            regulation_scope=r["scope"],
        ))
    return cases


def filter_report(csv_path: Path, candidates: pd.DataFrame, sampled: pd.DataFrame,
                  funnel: dict | None = None, start: str | None = DEFAULT_START,
                  end: str | None = DEFAULT_END) -> dict:
    """Provenance record for the methods section: what was filtered, and what survived."""
    return {
        "source_file": str(csv_path),
        "funnel": [{"step": k, "description": d, "rows": (funnel or {}).get(k)} for k, d in FUNNEL_STEPS],
        "study_window": [start, end],
        "candidates_after_filters": int(len(candidates)),
        "candidate_dispute_type_counts": candidates["dispute_type"].value_counts().to_dict(),
        "sampled": int(len(sampled)),
        "dispute_type_counts": sampled["dispute_type"].value_counts().to_dict(),
        "product_counts": sampled["product"].value_counts().to_dict(),
        "company_response_counts": (
            sampled["company_response"].value_counts().to_dict()
            if "company_response" in sampled else {}
        ),
        "date_range": [str(sampled["date_received"].min()), str(sampled["date_received"].max())]
        if "date_received" in sampled else None,
        "filters": {
            "narrative_required": True,
            "min_narrative_words": _MIN_NARRATIVE_WORDS,
            "max_narrative_words": _MAX_NARRATIVE_WORDS,
            "products": sorted(set(sampled["product"])) if len(sampled) else [],
        },
    }


def save_cases(cases: list[DisputeCase], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for c in cases:
            f.write(c.model_dump_json() + "\n")


def load_cases(path: Path) -> list[DisputeCase]:
    return [DisputeCase.model_validate_json(l) for l in open(path) if l.strip()]
