"""Step 1c (final report): draw the locked TEST sample, stratified by calendar quarter and dispute type.

The interim pilot (125 labelled complaints) becomes the DEVELOPMENT set. This script draws a new,
disjoint sample for the confirmatory results. It never touches the pilot files.

Design (fixed before any test output exists; recorded in data/splits/split_manifest.json):
  * Frame: the same cleaned candidate pool as the pilot (same filters, window, length limits,
    de-duplication) via cfpb_cases.load_candidates.
  * Exclusions: every pilot complaint ID (all 160 sampled, not just the 125 labelled) and every
    candidate whose normalised narrative is identical to a pilot narrative, so templated campaign
    texts cannot carry dev-set knowledge into the test set.
  * Allocation: the draw is spread over calendar quarters in proportion to the number of days of
    each quarter inside the window (so the two partial quarters get a pro-rata share), with a floor
    of `--min-per-quarter`. Volume does NOT drive allocation, so the January 2025 burst cannot
    dominate. Inside each quarter the draw is split as evenly as possible across dispute types.
  * Size: `--n-draw` complaints (default 205 = 160 labelled / 0.78, the in-scope rate of the pilot).

Usage:
    python scripts/01c_draw_test_sample.py                        # defaults: --n-draw 205 --seed 2026
    python scripts/01c_draw_test_sample.py --n-draw 205 --seed 2026 --csv data/raw/cfpb/complaints.csv

Writes:
    data/cases/test/cases.jsonl, sample_test.csv, selection_report.json
    data/gold/test/labeling_sheet.csv
    data/splits/split_manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from dispute_rag.cfpb_cases import (  # noqa: E402
    DEFAULT_START, FUNNEL_STEPS, load_candidates, require_raw_csv, load_cases, save_cases, to_cases,
)
from dispute_rag.gold import export_labeling_sheet  # noqa: E402
from dispute_rag.splits import MANIFEST, paths  # noqa: E402


def norm_text(s: str) -> str:
    return " ".join(str(s).split()).lower()


def quarter_label(d: pd.Timestamp) -> str:
    return f"{d.year}Q{(d.month - 1) // 3 + 1}"


def quarter_days(start: str, end: str) -> dict[str, int]:
    """Days of each calendar quarter that fall inside [start, end]."""
    days = pd.date_range(start, end, freq="D")
    return pd.Series(1, index=days).groupby([quarter_label(d) for d in days]).sum().to_dict()


def allocate(weights: dict[str, float], total: int, floor: int, capacity: dict[str, int]) -> dict[str, int]:
    """Largest-remainder allocation of `total` over strata, with a floor and a capacity cap.
    Each pass gives every open stratum the floor of its exact share, then hands the leftover units
    one each to the largest fractional parts; strata that hit capacity drop out and the next pass
    redistributes what they could not take."""
    alloc = {k: min(capacity.get(k, 0), floor) for k in weights}
    while True:
        remaining = total - sum(alloc.values())
        open_ = {k: w for k, w in weights.items() if alloc[k] < capacity.get(k, 0)}
        if remaining <= 0 or not open_:
            return alloc
        wsum = sum(open_.values())
        exact = {k: remaining * w / wsum for k, w in open_.items()}
        give = {k: math.floor(v) for k, v in exact.items()}
        order = sorted(open_, key=lambda k: (exact[k] - give[k], open_[k], k), reverse=True)
        for k in order[: remaining - sum(give.values())]:
            give[k] += 1
        for k, v in give.items():
            alloc[k] = min(capacity[k], alloc[k] + v)


def draw(pool: pd.DataFrame, n_draw: int, seed: int, start: str, end: str, min_per_quarter: int) -> tuple[pd.DataFrame, dict]:
    pool = pool.copy()
    pool["quarter"] = pd.to_datetime(pool["date_received"], errors="coerce").map(
        lambda d: quarter_label(d) if pd.notna(d) else None)
    pool = pool[pool["quarter"].notna()]
    qdays = quarter_days(start, end)
    capacity = pool.groupby("quarter").size().to_dict()
    weights = {q: qdays[q] / 91.0 for q in qdays}  # full quarter ~ 1.0
    q_alloc = allocate(weights, n_draw, min_per_quarter, capacity)

    picked = []
    for qi, (q, k) in enumerate(sorted(q_alloc.items())):
        if k <= 0:
            continue
        g = pool[pool["quarter"] == q]
        types = sorted(g["dispute_type"].unique())
        t_alloc = allocate({t: 1.0 for t in types}, k, 0, g.groupby("dispute_type").size().to_dict())
        for ti, t in enumerate(types):
            if t_alloc.get(t, 0):
                picked.append(g[g["dispute_type"] == t].sample(t_alloc[t], random_state=seed + 100 * qi + ti))
    out = pd.concat(picked, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    design = {"quarter_days": qdays, "quarter_weights": weights, "quarter_allocation": q_alloc,
              "quarter_capacity": capacity,
              "drawn_by_quarter_and_type": out.groupby(["quarter", "dispute_type"]).size()
                                             .unstack(fill_value=0).to_dict(orient="index")}
    return out, design


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="default: data/raw/cfpb/complaints.csv (a real complaints.csv.gz also works)")
    ap.add_argument("--n-draw", type=int, default=205)
    ap.add_argument("--seed", type=int, default=2026, help="different from the pilot seed (42)")
    ap.add_argument("--min-per-quarter", type=int, default=4)
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--end", default="2026-08-11",
                    help="last complaint in the narratives archive. The pilot used 2026-08-14 (the date "
                         "publication stopped); no complaint falls between the two, so the pool is identical. "
                         "Kept at 08-11 so an existing test draw reproduces exactly.")
    ap.add_argument("--force", action="store_true", help="overwrite an existing test sample")
    args = ap.parse_args()

    dev, test = paths("dev"), paths("test")
    if test.cases.exists() and not args.force:
        sys.exit(f"{test.cases} already exists. The test sample is drawn once; use --force only "
                 "if no test labels or outputs exist yet.")
    args.csv = str(require_raw_csv(args.csv))
    dev_cases = load_cases(dev.cases)
    dev_ids = {c.case_id for c in dev_cases}
    dev_texts = {norm_text(c.narrative) for c in dev_cases}
    print(f"Development set: {len(dev_ids)} sampled pilot complaints excluded")

    funnel: dict = {}
    pool = load_candidates(Path(args.csv), start=args.start, end=args.end, funnel=funnel)
    pool["complaint_id"] = pool["complaint_id"].astype(str)
    n0 = len(pool)
    pool = pool[~pool["complaint_id"].isin(dev_ids)]
    n_id = n0 - len(pool)
    pool = pool[~pool["narrative"].map(norm_text).isin(dev_texts)]
    n_txt = n0 - n_id - len(pool)
    print(f"Pool {n0:,}; removed {n_id} pilot IDs and {n_txt} narratives identical to a pilot narrative")

    sample, design = draw(pool, args.n_draw, args.seed, args.start, args.end, args.min_per_quarter)
    overlap = set(sample["complaint_id"]) & dev_ids
    assert not overlap, f"test sample overlaps the dev set: {sorted(overlap)[:5]}"

    cases = to_cases(sample)
    test.cases.parent.mkdir(parents=True, exist_ok=True)
    save_cases(cases, test.cases)
    sample.drop(columns=["scope"], errors="ignore").assign(
        narrative_words=sample["narrative"].str.split().str.len()
    ).to_csv(test.cases.parent / "sample_test.csv", index=False)
    export_labeling_sheet(cases, test.labeling)

    report = {"csv": str(args.csv), "window": [args.start, args.end], "seed": args.seed,
              "n_draw": args.n_draw, "pool_after_cleaning": n0, "excluded_pilot_ids": n_id,
              "excluded_identical_narratives": n_txt,
              # same list format as the pilot's selection_report.json, plus the two test-set exclusions
              "funnel": [{"step": k, "description": d, "rows": funnel.get(k)} for k, d in FUNNEL_STEPS]
                        + [{"step": "not_pilot_id", "description": "Not a pilot (development) complaint",
                            "rows": n0 - n_id},
                           {"step": "not_pilot_text", "description": "Narrative not identical to a pilot narrative",
                            "rows": n0 - n_id - n_txt}],
              **design,
              "dispute_type_counts": sample["dispute_type"].value_counts().to_dict(),
              # keys read by scripts/01b_eda.py, so the EDA can be recomputed on the test sample
              "sampled": len(cases), "candidates_after_filters": n0,
              "product_counts": sample["product"].value_counts().to_dict(),
              "company_response_counts": sample["company_response"].value_counts().to_dict()
                                         if "company_response" in sample else {},
              "date_range": [str(sample["date_received"].min()), str(sample["date_received"].max())],
              "study_window": [args.start, args.end]}
    (test.cases.parent / "selection_report.json").write_text(json.dumps(report, indent=2, default=str))

    test_ids = sorted(c.case_id for c in cases)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dev": {"role": "development (interim pilot); decisions made while viewing these results",
                "case_ids": sorted(dev_ids), "seed": 42},
        "test": {"role": "locked confirmatory test set; no output viewed before protocol lock",
                 "case_ids": test_ids, "seed": args.seed, "n_draw": args.n_draw,
                 "design": "quarter (days-in-window weights) x dispute type, pilot IDs and identical narratives excluded"},
        "sha256_test_ids": hashlib.sha256("\n".join(test_ids).encode()).hexdigest(),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2))

    print(f"\nDrew {len(cases)} test complaints -> {test.cases}")
    print("By quarter:", design["quarter_allocation"])
    print("By type:   ", report["dispute_type_counts"])
    print(f"Label {test.labeling}, then: python scripts/02_import_gold.py --split test")


if __name__ == "__main__":
    main()
