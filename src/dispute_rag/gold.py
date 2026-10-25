"""Gold-set helpers: export a labeling sheet, import completed labels, compute inter-rater agreement."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .schemas import DisputeCase, GoldLabel


def export_labeling_sheet(cases: list[DisputeCase], out_csv: Path) -> None:
    """Human labels in a spreadsheet.

    `company_response` is shown as context only. It records what the firm did, not what the
    regulation requires, so the rater must not copy it into `determination` — disagreement
    between the two is itself a finding worth reporting.
    """
    rows = [{
        "case_id": c.case_id,
        "dispute_type": c.dispute_type,
        "regulation_scope": c.regulation_scope,
        "product": c.complaint.product,
        "issue": c.complaint.issue,
        "sub_issue": c.complaint.sub_issue or "",
        "date_received": c.complaint.date_received or "",
        "narrative": c.narrative,
        "company_response_CONTEXT_ONLY": c.complaint.company_response or "",
        # Rater fills these:
        "governing_clauses": "", "determination": "", "rationale": "", "labeled_by": "",
        "ambiguous": "", "second_rater_determination": "", "second_rater_clauses": "",
    } for c in cases]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def import_labels(csv_path: Path, out_jsonl: Path) -> list[GoldLabel]:
    df = pd.read_csv(csv_path).fillna("")
    labels = []
    for _, r in df.iterrows():
        if not r.determination:
            continue
        labels.append(GoldLabel(
            case_id=str(r.case_id),
            governing_clauses=[c for c in str(r.governing_clauses).split("|") if c],
            determination=r.determination, rationale=r.rationale, labeled_by=r.labeled_by,
            second_rater_determination=r.second_rater_determination or None,
            second_rater_clauses=[c for c in str(r.get("second_rater_clauses", "")).split("|") if c],
            ambiguous=str(r.get("ambiguous", "")).strip().lower() in ("1", "true", "yes", "y")))
    with open(out_jsonl, "w") as f:
        for g in labels:
            f.write(g.model_dump_json() + "\n")
    return labels


def load_gold(path: Path) -> dict[str, GoldLabel]:
    return {g.case_id: g for g in (GoldLabel.model_validate_json(l) for l in open(path) if l.strip())}


def cohen_kappa(labels: list[GoldLabel]) -> float | None:
    """Reliability of the determination label. Reported in the interim and final report;
    if fewer than 10 cases are double-rated, state single-rater labeling as a limitation."""
    from sklearn.metrics import cohen_kappa_score
    pairs = [(g.determination, g.second_rater_determination) for g in labels if g.second_rater_determination]
    if len(pairs) < 10:
        return None
    a, b = zip(*pairs)
    return float(cohen_kappa_score(a, b))


def clause_agreement(labels: list[GoldLabel]) -> float | None:
    """Jaccard overlap of governing clauses between raters — reliability of the clause label,
    which Cohen's kappa on the determination alone does not capture."""
    scores = []
    for g in labels:
        if not g.second_rater_clauses:
            continue
        a, b = {c.lower() for c in g.governing_clauses}, {c.lower() for c in g.second_rater_clauses}
        if a or b:
            scores.append(len(a & b) / len(a | b))
    return sum(scores) / len(scores) if scores else None


def company_response_divergence(labels: list[GoldLabel], cases: list[DisputeCase]) -> pd.DataFrame:
    """Cross-tab of the rater's regulation-based determination against the company's actual
    disposition. Descriptive only — it characterizes the sample, and a wide divergence is a
    finding about the marketplace, not an error in the labels."""
    by_id = {c.case_id: c for c in cases}
    rows = [{
        "case_id": g.case_id,
        "gold_determination": g.determination,
        "company_response": (by_id[g.case_id].complaint.company_response or "unknown")
        if g.case_id in by_id else "unknown",
    } for g in labels]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.groupby(["gold_determination", "company_response"]).size().unstack(fill_value=0)
