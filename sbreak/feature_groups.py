"""Optional feature groups, each ablatable on its own.

A group is a function ``(values, periods) -> dict[str, float]`` computed per
series and appended to the base feature set (``Config.feature_set``). Groups
are enabled with ``Config.feature_groups`` (``--groups`` in scripts.evaluate)
and cached separately, so adding one never invalidates the base table.
Column names are prefixed with ``g_<group>__`` to keep them distinct.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

GroupFn = Callable[[np.ndarray, np.ndarray], dict]
GROUPS: dict[str, GroupFn] = {}


def register(name: str):
    def deco(fn: GroupFn) -> GroupFn:
        GROUPS[name] = fn
        return fn

    return deco


def compute_groups(values: np.ndarray, periods: np.ndarray, groups) -> dict:
    out = {}
    for g in groups:
        if g not in GROUPS:
            raise KeyError(f"unknown feature group {g!r}; known: {sorted(GROUPS)}")
        for k, v in GROUPS[g](values, periods).items():
            out[f"g_{g}__{k}"] = v
    return out


EPS = 1e-12


def _safe(x) -> float:
    x = float(x)
    return x if np.isfinite(x) else 0.0


# ---------------------------------------------------------------------------
# scan: how special is the *given* boundary among all possible split points?
# A break at the boundary makes the boundary the (near-)argmax of a change-point
# scan; a break elsewhere, or none, does not. O(n) per statistic via cumsums.
# ---------------------------------------------------------------------------
def _scan(stat: np.ndarray, k0: int, n: int, prefix: str, out: dict) -> None:
    lo, hi = int(0.05 * n), int(0.95 * n)  # trim the ends, where scans are unstable
    s = stat[lo:hi]
    at = stat[k0 - 1]
    mx = s.max() if len(s) else at
    out[f"{prefix}_at"] = _safe(at)
    out[f"{prefix}_max"] = _safe(mx)
    out[f"{prefix}_at_over_max"] = _safe(at / (mx + EPS))
    out[f"{prefix}_argmax_dist"] = _safe(abs(lo + int(np.argmax(s)) + 1 - k0) / n) if len(s) else 0.0
    out[f"{prefix}_pct_rank"] = _safe((s < at).mean()) if len(s) else 0.0


@register("scan")
def scan_group(values: np.ndarray, periods: np.ndarray) -> dict:
    v = values
    n = len(v)
    k0 = int((periods == 0).sum())
    out: dict = {}
    if n < 20 or k0 < 2 or n - k0 < 2:
        return out
    k = np.arange(1, n)
    w = np.sqrt(k * (n - k) / n)
    sd = v.std() + EPS
    # mean change: standardised CUSUM of x
    _scan(np.abs(np.cumsum(v - v.mean())[:-1]) / (w * sd), k0, n, "mean_cusum", out)
    # mean change, rank-based (robust): CUSUM of ranks
    r = np.argsort(np.argsort(v)).astype(float)
    _scan(np.abs(np.cumsum(r - r.mean())[:-1]) / (w * (r.std() + EPS)), k0, n, "rank_cusum", out)
    # variance change: Inclan-Tiao CUSUM of squares
    c2 = np.cumsum((v - v.mean()) ** 2)
    _scan(np.abs(c2[:-1] / (c2[-1] + EPS) - k / n) * np.sqrt(n / 2), k0, n, "var_cusum", out)
    # robust volatility change: CUSUM of |x - median|
    av = np.abs(v - np.median(v))
    _scan(np.abs(np.cumsum(av - av.mean())[:-1]) / (w * (av.std() + EPS)), k0, n, "absdev_cusum", out)
    # Gaussian mean+variance likelihood ratio at every split, per point
    c1 = np.cumsum(v)
    cs = np.cumsum(v * v)
    ka = k.astype(float)
    va = cs[:-1] / ka - (c1[:-1] / ka) ** 2
    vb = (cs[-1] - cs[:-1]) / (n - ka) - ((c1[-1] - c1[:-1]) / (n - ka)) ** 2
    ll0 = n * np.log(v.var() + EPS)
    lr = 0.5 * (ll0 - ka * np.log(np.maximum(va, EPS)) - (n - ka) * np.log(np.maximum(vb, EPS))) / n
    _scan(lr, k0, n, "gauss_lr", out)
    return out


# ---------------------------------------------------------------------------
# tests: classical two-sample statistics at the boundary, unsaturated and on a
# scale that does not grow with segment length (the base set caps -log10 p at 8
# and leaves Levene's p raw).
# ---------------------------------------------------------------------------
@register("tests")
def tests_group(values: np.ndarray, periods: np.ndarray) -> dict:
    from scipy import stats

    a, b = values[periods == 0], values[periods == 1]
    na, nb = len(a), len(b)
    out: dict = {}
    if na < 5 or nb < 5:
        return out
    neff = na * nb / (na + nb)
    rs = np.sqrt(neff)
    t = stats.ttest_ind(a, b, equal_var=False)
    out["welch_t_abs_over_sqrt_neff"] = _safe(abs(t.statistic) / rs)
    out["welch_logp"] = _safe(-np.log10(max(float(t.pvalue), 1e-300)))
    ks = stats.ks_2samp(a, b)
    out["ks_scaled"] = _safe(ks.statistic * rs)  # Kolmogorov scaling, length-free under H0
    lev = stats.levene(a, b, center="median")  # Brown-Forsythe
    out["bf_logF"] = _safe(np.log1p(lev.statistic))
    out["bf_F_over_neff"] = _safe(lev.statistic / neff)
    aa, ab = np.abs(a - np.median(a)), np.abs(b - np.median(b))
    out["log_mad_ratio"] = _safe(np.log((np.median(ab) + EPS) / (np.median(aa) + EPS)))
    out["log_meanabsdev_ratio"] = _safe(np.log((ab.mean() + EPS) / (aa.mean() + EPS)))
    mw = stats.mannwhitneyu(a, b)
    out["mw_auc_dev"] = _safe(abs(mw.statistic / (na * nb) - 0.5))  # rank effect size
    # Gaussian LR for (mean, variance) and variance-only, per point
    v = values
    lr_mv = 0.5 * (len(v) * np.log(v.var() + EPS) - na * np.log(a.var() + EPS) - nb * np.log(b.var() + EPS))
    mu = v.mean()
    lr_v = 0.5 * (len(v) * np.log(v.var() + EPS) - na * np.log(((a - mu) ** 2).mean() + EPS)
                  - nb * np.log(((b - mu) ** 2).mean() + EPS))
    out["lr_meanvar_per_n"] = _safe(lr_mv / len(v))
    out["lr_var_per_n"] = _safe(lr_v / len(v))
    return out
