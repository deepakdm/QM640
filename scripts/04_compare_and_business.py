"""Inferential statistics for the configuration comparison.

QM 640 learning outcomes require descriptive, inferential, and prescriptive analysis, and an
evaluation of the validity and reliability of the statistics used. This module supplies the
inferential layer:

  descriptive   -> evaluate.aggregate()        (means, medians, percentiles)
  inferential   -> this module                 (CIs, omnibus test, pairwise tests, effect sizes)
  prescriptive  -> business_model.py           (cost model, break-even, sensitivity)

Design notes for the methods section:
  * Every configuration is scored on the SAME gold cases, so the comparisons are PAIRED.
    Independent-sample tests (chi-square on a 2x2 of totals, two-proportion z) would be wrong here
    and would overstate the evidence.
  * Cochran's Q is the omnibus test for k paired binary outcomes; pairwise McNemar tests follow
    only if Q is significant, with Holm correction to control the family-wise error rate.
  * Accuracy CIs use the percentile bootstrap rather than a normal approximation, which is
    unreliable at n=150 with proportions near 0 or 1.
"""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats


def bootstrap_ci(successes: Sequence[int], n_boot: int = 10_000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a proportion. Returns (point_estimate, low, high)."""
    x = np.asarray(successes, dtype=float)
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    draws = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return (float(x.mean()),
            float(np.percentile(draws, 100 * alpha / 2)),
            float(np.percentile(draws, 100 * (1 - alpha / 2))))


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval — reported alongside the bootstrap as a closed-form cross-check."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    # Floating-point rounding can put the bound a hair on the wrong side of p (e.g. 1e-18 > 0 when
    # successes == 0), so keep the interval inside [0, 1] and around the point estimate.
    lo = min(max(0.0, float(centre - half)), p)
    hi = max(min(1.0, float(centre + half)), p)
    return (lo, hi)


def cochrans_q(matrix: np.ndarray) -> dict:
    """Omnibus test for k related binary samples (cases x configs, values in {0,1}).

    H0: all configurations have the same success probability.
    """
    m = np.asarray(matrix, dtype=float)
    n, k = m.shape
    if k < 3:
        raise ValueError("Cochran's Q needs at least 3 configurations; use McNemar for 2.")
    col = m.sum(axis=0)
    row = m.sum(axis=1)
    total = m.sum()
    denom = k * total - (row**2).sum()
    if denom == 0:
        return {"Q": np.nan, "df": k - 1, "p_value": np.nan, "n": n, "k": k,
                "note": "no discordant cases; every configuration scored identically"}
    q = (k - 1) * (k * (col**2).sum() - total**2) / denom
    return {"Q": float(q), "df": k - 1, "p_value": float(stats.chi2.sf(q, k - 1)),
            "n": int(n), "k": int(k)}


def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> dict:
    """Exact (binomial) McNemar test for two paired binary vectors.

    Discordant pairs only: n01 = a wrong & b right, n10 = a right & b wrong.
    The exact form is used because discordant counts are typically small at n=150.
    """
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    disc = n10 + n01
    if disc == 0:
        return {"n10": 0, "n01": 0, "discordant": 0, "p_value": 1.0, "odds_ratio": np.nan}
    p = float(stats.binomtest(min(n10, n01), disc, 0.5, alternative="two-sided").pvalue)
    return {"n10": n10, "n01": n01, "discordant": disc, "p_value": p,
            "odds_ratio": float(n10 / n01) if n01 else np.inf}


def holm_correction(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values, controlling family-wise error across the pairwise tests."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adjusted[key] = running
    return adjusted


def compare_configs(per_case: dict[str, pd.DataFrame],
                    metric: str = "determination_correct") -> dict:
    """Full inferential comparison across configurations on a paired binary metric.

    `per_case` maps config name -> the per_case_metrics.csv DataFrame for that config.
    Cases are aligned on case_id, so only cases scored by every configuration are used.
    """
    names = sorted(per_case)
    frames = {n: per_case[n].set_index("case_id")[metric] for n in names}
    common = sorted(set.intersection(*(set(s.index) for s in frames.values())))
    mat = np.column_stack([frames[n].loc[common].astype(int).to_numpy() for n in names])

    out: dict = {"metric": metric, "configs": names, "n_cases": len(common), "per_config": {}}
    for j, n in enumerate(names):
        col = mat[:, j]
        est, lo, hi = bootstrap_ci(col)
        w_lo, w_hi = wilson_ci(int(col.sum()), len(col))
        out["per_config"][n] = {
            "successes": int(col.sum()), "n": int(len(col)), "estimate": est,
            "bootstrap_ci": [lo, hi], "wilson_ci": [w_lo, w_hi],
        }

    if len(names) >= 3:
        out["omnibus"] = cochrans_q(mat)
    raw, detail = {}, {}
    for i, j in combinations(range(len(names)), 2):
        key = f"{names[i]} vs {names[j]}"
        res = mcnemar_exact(mat[:, i], mat[:, j])
        detail[key] = res
        raw[key] = res["p_value"]
    out["pairwise_mcnemar"] = detail
    out["pairwise_p_holm"] = holm_correction(raw) if raw else {}
    return out


def latency_test(per_case: dict[str, pd.DataFrame],
                 column: str = "latency_median_s") -> dict:
    """Wilcoxon signed-rank on paired per-case latency. Non-parametric because latency is
    right-skewed and a paired t-test's normality assumption does not hold."""
    names = sorted(per_case)
    frames = {n: per_case[n].set_index("case_id")[column] for n in names}
    common = sorted(set.intersection(*(set(s.index) for s in frames.values())))
    out = {"column": column, "n_cases": len(common), "medians": {}, "pairwise": {}}
    for n in names:
        out["medians"][n] = float(frames[n].loc[common].median())
    raw = {}
    for a, b in combinations(names, 2):
        x = frames[a].loc[common].to_numpy(dtype=float)
        y = frames[b].loc[common].to_numpy(dtype=float)
        if np.allclose(x, y):
            raw[f"{a} vs {b}"] = 1.0
            out["pairwise"][f"{a} vs {b}"] = {"statistic": np.nan, "p_value": 1.0}
            continue
        st, p = stats.wilcoxon(x, y)
        out["pairwise"][f"{a} vs {b}"] = {"statistic": float(st), "p_value": float(p)}
        raw[f"{a} vs {b}"] = float(p)
    out["pairwise_p_holm"] = holm_correction(raw) if raw else {}
    return out


def summarize_for_report(comparison: dict, alpha: float = 0.05) -> str:
    """Plain-language rendering of the inferential result, for pasting into the results section."""
    lines = [f"Metric: {comparison['metric']} (n = {comparison['n_cases']} paired cases)"]
    for name, s in comparison["per_config"].items():
        lo, hi = s["bootstrap_ci"]
        lines.append(f"  {name}: {s['estimate']:.3f} (95% bootstrap CI {lo:.3f}–{hi:.3f}, "
                     f"{s['successes']}/{s['n']})")
    if "omnibus" in comparison:
        o = comparison["omnibus"]
        verdict = "reject" if o.get("p_value", 1) < alpha else "fail to reject"
        lines.append(f"  Cochran's Q = {o['Q']:.3f}, df = {o['df']}, p = {o['p_value']:.4f} "
                     f"-> {verdict} H0 of equal accuracy at alpha = {alpha}")
    for key, p_adj in comparison.get("pairwise_p_holm", {}).items():
        d = comparison["pairwise_mcnemar"][key]
        sig = "significant" if p_adj < alpha else "not significant"
        lines.append(f"  McNemar {key}: discordant = {d['discordant']} "
                     f"(n10 = {d['n10']}, n01 = {d['n01']}), p = {d['p_value']:.4f}, "
                     f"Holm-adjusted p = {p_adj:.4f} -> {sig}")
    return "\n".join(lines)
