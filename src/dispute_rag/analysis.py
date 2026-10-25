"""Confirmatory analysis for RQ1, RQ2 and RQ4, and the adoption benchmark table (synopsis Table 1).

Every retrieval condition is compared with the no-retrieval control (C0) on the same complaints,
so the tests are paired: exact McNemar for each of the m = 3 comparisons, Holm-adjusted within
each outcome, with Cochran's Q as the omnibus check for accuracy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from .stats import bootstrap_ci, cochrans_q, holm_correction, mcnemar_exact, wilson_ci

# config.name in configs/*.yaml -> condition label used in the report
CONDITION_LABELS = {
    "control_no_retrieval": "C0",
    "baseline_dense_512": "A",
    "dense_256": "B",
    "hybrid_rerank_512": "C",
}
CONTROL, RETRIEVAL = "C0", ("A", "B", "C")
DEPLOYMENT_CANDIDATE = "C"

BENCHMARKS = {
    "accuracy_gain": 0.15,
    "fabricated_upper": 0.05,
    "consistency": 0.90,
    "p95_latency_s": 30.0,
}


def load_per_case(runs_dir: Path) -> pd.DataFrame:
    """Stack outputs/runs/<config>/per_case_metrics.csv into one long table with a `condition`."""
    frames = []
    for p in sorted(Path(runs_dir).glob("*/per_case_metrics.csv")):
        df = pd.read_csv(p, dtype={"case_id": str})
        df["condition"] = df["config"].map(CONDITION_LABELS).fillna(df["config"])
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _paired_matrix(long: pd.DataFrame, metric: str, conditions) -> pd.DataFrame:
    wide = long.pivot_table(index="case_id", columns="condition", values=metric, aggfunc="first")
    return wide[[c for c in conditions if c in wide.columns]].dropna()


def compare_to_control(long: pd.DataFrame, metric: str, higher_is_better: bool = True) -> dict:
    """Rate per condition with Wilson and bootstrap CIs; exact McNemar of each retrieval
    condition against C0, Holm-adjusted over the m comparisons; Cochran's Q across all four."""
    conds = [CONTROL] + list(RETRIEVAL)
    wide = _paired_matrix(long, metric, conds)
    out = {"metric": metric, "n_cases": int(len(wide)), "per_condition": {}, "vs_control": {}}
    for c in wide.columns:
        x = wide[c].astype(int).to_numpy()
        est, lo, hi = bootstrap_ci(x)
        wl, wh = wilson_ci(int(x.sum()), len(x))
        out["per_condition"][c] = {"successes": int(x.sum()), "n": int(len(x)), "rate": est,
                                   "wilson_ci": [wl, wh], "bootstrap_ci": [lo, hi]}
    if CONTROL not in wide.columns:
        return out
    raw = {}
    base = wide[CONTROL].astype(int).to_numpy()
    for c in [r for r in RETRIEVAL if r in wide.columns]:
        x = wide[c].astype(int).to_numpy()
        res = mcnemar_exact(x, base)          # n10 = retrieval 1 & control 0
        diff = x - base
        rng = np.random.default_rng(42)
        boots = rng.choice(diff, size=(10_000, len(diff)), replace=True).mean(axis=1)
        res["difference"] = float(diff.mean())
        res["difference_ci"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
        out["vs_control"][c] = res
        raw[c] = res["p_value"]
    for c, p in holm_correction(raw).items():
        out["vs_control"][c]["p_holm"] = p
    if len(wide.columns) >= 3 and metric == "determination_correct":
        out["cochran_q"] = cochrans_q(wide.astype(int).to_numpy())
    return out


def p95_ci(x: np.ndarray, n_boot: int = 10_000, seed: int = 42) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if not len(x):
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = np.percentile(rng.choice(x, size=(n_boot, len(x)), replace=True), 95, axis=1)
    return float(np.percentile(x, 95)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def dominant_failure(records: pd.DataFrame, min_errors: int = 43) -> dict:
    """Share of errors in the most common failure category: Wilson CI and one-sided exact
    binomial test against an even split over five categories (.20). Uses Config C's errors, or
    pools A, B and C if C has fewer than `min_errors` (synopsis, RQ4 sample size)."""
    err = records[records["category"] != "correct"]
    pool = err[err["condition"] == DEPLOYMENT_CANDIDATE]
    pooled = False
    if len(pool) < min_errors:
        pool, pooled = err[err["condition"].isin(RETRIEVAL)], True
    if pool.empty:
        return {"n_errors": 0}
    counts = pool["category"].value_counts()
    top, x, n = counts.index[0], int(counts.iloc[0]), int(counts.sum())
    lo, hi = wilson_ci(x, n)
    p = float(sps.binomtest(x, n, 0.20, alternative="greater").pvalue)
    return {"category": top, "count": x, "n_errors": n, "share": x / n, "wilson_ci": [lo, hi],
            "p_one_sided_vs_0.20": p, "pooled_A_B_C": pooled, "counts": counts.to_dict()}


def _interval_verdict(point, lo, hi, threshold, direction: str) -> str:
    """'met' if the whole interval satisfies the benchmark, 'inconclusive' if only the point
    estimate does, otherwise 'not met' (synopsis Table 1 note)."""
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (point, lo, hi)):
        return "not available"
    if direction == "<=":
        return "met" if hi <= threshold else ("inconclusive" if point <= threshold else "not met")
    return "met" if lo >= threshold else ("inconclusive" if point >= threshold else "not met")


def benchmark_table(long: pd.DataFrame, accuracy: dict, clause: dict,
                    breakeven: float | None = None, cost_ok: bool | None = None) -> pd.DataFrame:
    c = long[long["condition"] == DEPLOYMENT_CANDIDATE]
    rows = []
    if c.empty:
        return pd.DataFrame()
    vs = accuracy.get("vs_control", {}).get(DEPLOYMENT_CANDIDATE, {})
    gain, p_holm = vs.get("difference", np.nan), vs.get("p_holm", np.nan)
    rows.append(["Accuracy", "Accuracy gain over C0", f"{gain:+.3f} (Holm p = {p_holm:.4f})",
                 ">= .15 and Holm p < .05",
                 "met" if gain >= BENCHMARKS["accuracy_gain"] and p_holm < .05 else "not met"])
    acc = c["determination_correct"].mean()
    if breakeven is not None:
        rows.append(["Accuracy", "Absolute accuracy", f"{acc:.3f}", f"> break-even {breakeven:.2f}",
                     "met" if acc > breakeven else "not met"])
    n = int(c["fabricated_citation"].notna().sum())
    x = int(c["fabricated_citation"].sum())
    lo, hi = wilson_ci(x, n)
    rows.append(["Groundedness", "Fabricated-citation rate", f"{x / n:.3f} [{lo:.3f}, {hi:.3f}]",
                 "upper 95% Wilson bound <= .05",
                 _interval_verdict(x / n, lo, hi, BENCHMARKS["fabricated_upper"], "<=")])
    cv = clause.get("vs_control", {}).get(DEPLOYMENT_CANDIDATE, {})
    rows.append(["Groundedness", "Correct clause cited vs C0",
                 f"{cv.get('difference', np.nan):+.3f} (Holm p = {cv.get('p_holm', np.nan):.4f})",
                 "higher than C0, Holm p < .05",
                 "met" if cv.get("difference", 0) > 0 and cv.get("p_holm", 1) < .05 else "not met"])
    cons = c["consistent"].dropna()
    if len(cons):
        k, m = int(cons.sum()), int(len(cons))
        lo, hi = wilson_ci(k, m)
        rows.append(["Operational", "Run-to-run consistency", f"{k / m:.3f} [{lo:.3f}, {hi:.3f}]",
                     ">= .90 of cases identical in all runs",
                     _interval_verdict(k / m, lo, hi, BENCHMARKS["consistency"], ">=")])
    pt, lo, hi = p95_ci(c["latency_s"].to_numpy())
    rows.append(["Operational", "95th-percentile response time", f"{pt:.1f}s [{lo:.1f}, {hi:.1f}]",
                 "<= 30 s", _interval_verdict(pt, lo, hi, BENCHMARKS["p95_latency_s"], "<=")])
    rr = c["review_required"]
    lo, hi = wilson_ci(int(rr.sum()), int(rr.notna().sum()))
    rows.append(["Operational", "Review burden", f"{rr.mean():.3f} [{lo:.3f}, {hi:.3f}]",
                 "reported; enters cost model", "reported"])
    if cost_ok is not None:
        rows.append(["Operational", "Cost per case", "see sensitivity table",
                     "assisted < manual minutes at observed accuracy", "met" if cost_ok else "not met"])
    return pd.DataFrame(rows, columns=["pillar", "indicator", "estimate", "benchmark", "verdict"])
