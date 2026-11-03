"""Episode metrics and bootstrap confidence intervals.

Detection times are censored at max_steps (an episode where a target was not found counts as max_steps).
coverage_auc is the mean of the coverage curve over the full horizon; after an early stop the curve
stays at its last value. revisit_ratio = 1 - distinct cells / agent-steps.
"""
from __future__ import annotations

import numpy as np


def coverage_curve(env, horizon: int | None = None) -> np.ndarray:
    horizon = env.cfg.max_steps if horizon is None else horizon
    cov = np.asarray(env.hist["coverage"], dtype=float)
    if len(cov) < horizon + 1:
        cov = np.concatenate([cov, np.full(horizon + 1 - len(cov), cov[-1])])
    return cov[:horizon + 1]


def episode_metrics(env) -> dict:
    c = env.cfg
    T = c.max_steps
    ft = env.found_t
    found = ft >= 0
    cens = np.where(found, ft, T)
    cov = coverage_curve(env)
    return {
        "success": float(found.all()),
        "frac_found": float(found.mean()),
        "t_first": float(cens.min()),
        "t_last": float(cens.max()),
        "coverage_final": float(env.hist["coverage"][-1]),
        "coverage_auc": float(cov.mean()),
        "revisit_ratio": float(1.0 - env.visited_global.sum() / max(env.total_visits, 1)),
        "messages": float(np.sum(env.hist["messages"])),
        "cells_tx": float(np.sum(env.hist["cells_tx"])),
        "steps": float(env.t),
        "n_failed": float((~env.active).sum()),
    }


def bootstrap_ci(x, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Mean and percentile-bootstrap (1-alpha) CI of the mean."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def robustness_ci(x_fail, x_nofail, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Ratio of means mean(x_fail)/mean(x_nofail) with an independent-resampling bootstrap CI."""
    a = np.asarray(x_fail, dtype=float)
    b = np.asarray(x_nofail, dtype=float)
    if len(a) == 0 or len(b) == 0 or b.mean() == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    ma = a[rng.integers(0, len(a), size=(n_boot, len(a)))].mean(1)
    mb = b[rng.integers(0, len(b), size=(n_boot, len(b)))].mean(1)
    ok = mb > 0
    ratio = ma[ok] / mb[ok]
    lo, hi = np.quantile(ratio, [alpha / 2, 1 - alpha / 2])
    return float(a.mean() / b.mean()), float(lo), float(hi)
