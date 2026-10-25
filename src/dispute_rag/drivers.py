"""RQ3: which system and case factors explain whether a retrieval-augmented determination is
correct, and how much of the configuration effect works through the governing clause reaching
the model.

Model. Logistic regression for `determination_correct` on the retrieval observations (complaints
x Configs A, B, C), estimated by generalized estimating equations (GEE) with an exchangeable
working correlation clustered on complaint (Liang & Zeger, 1986). Predictors: Config B vs A,
Config C vs A, clause in context, dispute type (reference = most frequent type), centred log
narrative length, and the ambiguity flag. The control is excluded because clause presence is
undefined without retrieval.

Mechanism. Each configuration effect is expressed as an average marginal effect (AME): the mean
change in predicted probability of a correct determination when every observation is set to that
configuration rather than Config A. The AME is estimated without the clause term (total) and with
it (direct); (total - direct) / total is the share of the effect that works through getting the
clause to the model. A cluster bootstrap over complaints gives its interval.

statsmodels is used when installed. Otherwise the module falls back to a logistic regression with
cluster-robust (sandwich) standard errors, which is GEE with an independence working correlation:
identical point-estimate logic, slightly less efficient. The output records which engine ran.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

CONFIG_LEVELS = ("A", "B", "C")


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def design_matrix(df: pd.DataFrame, include_clause: bool = True,
                  dispute_ref: str | None = None) -> tuple[pd.DataFrame, str]:
    """Build X with named columns. `df` needs: config (A/B/C), clause_in_context, dispute_type,
    narrative_words, ambiguous."""
    dispute_ref = dispute_ref or df["dispute_type"].value_counts().idxmax()
    X = pd.DataFrame(index=df.index)
    X["const"] = 1.0
    X["config_B"] = (df["config"] == "B").astype(float)
    X["config_C"] = (df["config"] == "C").astype(float)
    if include_clause:
        X["clause_in_context"] = df["clause_in_context"].astype(float)
    for t in sorted(df["dispute_type"].unique()):
        if t != dispute_ref:
            X[f"dispute_{t}"] = (df["dispute_type"] == t).astype(float)
    logw = np.log(df["narrative_words"].astype(float))
    X["log_words_c"] = logw - logw.mean()
    X["ambiguous"] = df["ambiguous"].astype(float)
    return X, dispute_ref


def fit_logit_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray,
                      max_iter: int = 100, ridge: float = 1e-6) -> dict:
    """Logistic regression by IRLS with cluster-robust sandwich covariance (CR1)."""
    n, k = X.shape
    beta = np.zeros(k)
    converged = False
    for _ in range(max_iter):
        p = _sigmoid(X @ beta)
        w = p * (1 - p)
        H = X.T @ (X * w[:, None]) + ridge * np.eye(k)
        step = np.linalg.solve(H, X.T @ (y - p))
        beta += step
        if np.max(np.abs(step)) < 1e-8:
            converged = True
            break
    p = _sigmoid(X @ beta)
    w = p * (1 - p)
    bread = np.linalg.inv(X.T @ (X * w[:, None]) + ridge * np.eye(k))
    resid = (y - p)[:, None] * X
    meat = np.zeros((k, k))
    uniq = np.unique(groups)
    for g in uniq:
        s = resid[groups == g].sum(axis=0)
        meat += np.outer(s, s)
    G = len(uniq)
    cov = bread @ meat @ bread * (G / (G - 1)) * ((n - 1) / (n - k))
    return {"params": beta, "cov": cov, "converged": converged, "max_abs_beta": float(np.max(np.abs(beta)))}


def fit_gee(y: np.ndarray, X: pd.DataFrame, groups: np.ndarray) -> dict:
    try:
        import statsmodels.api as sm
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sm.GEE(y, X, groups=groups, family=sm.families.Binomial(),
                         cov_struct=sm.cov_struct.Exchangeable()).fit()
        return {"params": np.asarray(res.params), "cov": np.asarray(res.cov_params()),
                "engine": "statsmodels GEE (exchangeable)",
                "working_correlation": float(res.model.cov_struct.dep_params),
                "converged": True}
    except ImportError:
        out = fit_logit_cluster(y, X.to_numpy(), groups)
        out["engine"] = "logit + cluster-robust SE (GEE, independence)"
        out["working_correlation"] = None
        return out


def odds_ratio_table(fit: dict, names: list[str]) -> pd.DataFrame:
    b = fit["params"]
    se = np.sqrt(np.diag(fit["cov"]))
    z = b / se
    return pd.DataFrame({
        "term": names, "beta": b, "se": se, "odds_ratio": np.exp(b),
        "or_ci_low": np.exp(b - 1.96 * se), "or_ci_high": np.exp(b + 1.96 * se),
        "z": z, "p_value": 2 * stats.norm.sf(np.abs(z)),
    }).set_index("term")


def ame(beta: np.ndarray, X: pd.DataFrame, config: str) -> float:
    """Average marginal effect of `config` (B or C) relative to Config A."""
    X1, X0 = X.copy(), X.copy()
    X1["config_B"], X1["config_C"] = float(config == "B"), float(config == "C")
    X0["config_B"], X0["config_C"] = 0.0, 0.0
    return float(np.mean(_sigmoid(X1.to_numpy() @ beta) - _sigmoid(X0.to_numpy() @ beta)))


def vif(X: pd.DataFrame) -> pd.Series:
    cols = [c for c in X.columns if c != "const" and X[c].std() > 0]
    out = {}
    for c in cols:
        others = [o for o in cols if o != c]
        A = np.column_stack([np.ones(len(X))] + [X[o] for o in others])
        coef, *_ = np.linalg.lstsq(A, X[c], rcond=None)
        r2 = 1 - np.sum((X[c] - A @ coef) ** 2) / np.sum((X[c] - X[c].mean()) ** 2)
        out[c] = 1 / (1 - r2) if r2 < 1 else np.inf
    return pd.Series(out, name="vif")


def driver_analysis(df: pd.DataFrame, n_boot: int = 500, seed: int = 42) -> dict:
    """Full RQ3 output. `df`: one row per complaint x retrieval configuration."""
    df = df.dropna(subset=["determination_correct", "clause_in_context"]).copy()
    df = df[df["config"].isin(CONFIG_LEVELS)]
    y = df["determination_correct"].astype(float).to_numpy()
    groups = df["case_id"].astype(str).to_numpy()

    X_full, ref = design_matrix(df, include_clause=True)
    X_tot, _ = design_matrix(df, include_clause=False, dispute_ref=ref)
    full = fit_gee(y, X_full, groups)
    tot = fit_gee(y, X_tot, groups)

    ames = {}
    for c in ("B", "C"):
        total, direct = ame(tot["params"], X_tot, c), ame(full["params"], X_full, c)
        ames[c] = {"ame_total": total, "ame_direct": direct,
                   "share_via_clause": (total - direct) / total if abs(total) > 1e-9 else np.nan}

    # Cluster bootstrap of the mediation share (fast logit fit; independence working correlation)
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    by_id = {g: np.flatnonzero(groups == g) for g in ids}
    boot = {c: [] for c in ("B", "C")}
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([by_id[g] for g in pick])
        new_groups = np.concatenate([[i] * len(by_id[g]) for i, g in enumerate(pick)])
        yb = y[idx]
        Xf, Xt = X_full.iloc[idx].reset_index(drop=True), X_tot.iloc[idx].reset_index(drop=True)
        try:
            bf = fit_logit_cluster(yb, Xf.to_numpy(), new_groups)["params"]
            bt = fit_logit_cluster(yb, Xt.to_numpy(), new_groups)["params"]
        except np.linalg.LinAlgError:
            continue
        for c in ("B", "C"):
            t, d = ame(bt, Xt, c), ame(bf, Xf, c)
            if abs(t) > 1e-9:
                boot[c].append((t - d) / t)
    for c in ("B", "C"):
        if boot[c]:
            ames[c]["share_ci"] = [float(np.percentile(boot[c], 2.5)), float(np.percentile(boot[c], 97.5))]

    n_err = int((y == 0).sum())
    return {
        "engine": full["engine"],
        "working_correlation": full.get("working_correlation"),
        "n_observations": int(len(df)), "n_complaints": int(len(ids)),
        "dispute_reference": ref,
        "events_per_parameter": n_err / X_full.shape[1],
        "odds_ratios": odds_ratio_table(full, list(X_full.columns)),
        "odds_ratios_total_model": odds_ratio_table(tot, list(X_tot.columns)),
        "ame": ames,
        "vif": vif(X_full),
        "converged": bool(full.get("converged", True)),
    }
