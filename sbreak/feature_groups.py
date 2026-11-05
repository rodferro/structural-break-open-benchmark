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


# ---------------------------------------------------------------------------
# selfnorm: is the change at the boundary larger than the changes the same
# series shows between adjacent windows *before* the boundary? Targets false
# alarms from series that are non-stationary throughout (error analysis:
# confident false alarms have large before/after distribution differences).
# ---------------------------------------------------------------------------
def _contrasts(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    from scipy.stats import ks_2samp

    sx, sz = x.std() + EPS, z.std() + EPS
    mx, mz = np.median(np.abs(x - np.median(x))) + EPS, np.median(np.abs(z - np.median(z))) + EPS
    return np.array([
        ks_2samp(x, z).statistic,
        abs(np.log(sz / sx)),
        abs(z.mean() - x.mean()) / np.sqrt(0.5 * (sx * sx + sz * sz)),
        abs(np.log(mz / mx)),
    ])


_CONTRASTS = ("ks", "logstd", "meanshift", "logmad")


@register("selfnorm")
def selfnorm_group(values: np.ndarray, periods: np.ndarray) -> dict:
    a, b = values[periods == 0], values[periods == 1]
    out: dict = {}
    w = int(min(len(b), len(a) // 3))
    if w < 30:
        return out
    bnd = _contrasts(a[-w:], b[:w])
    step = max(w // 2, 1)
    ref = [_contrasts(a[s : s + w], a[s + w : s + 2 * w]) for s in range(0, len(a) - 2 * w + 1, step)]
    ref = np.array(ref)
    mu, sd = ref.mean(axis=0), ref.std(axis=0) + EPS
    for i, name in enumerate(_CONTRASTS):
        out[f"{name}_boundary"] = _safe(bnd[i])
        out[f"{name}_ref_mean"] = _safe(mu[i])
        out[f"{name}_z"] = _safe((bnd[i] - mu[i]) / sd[i])
        out[f"{name}_over_refmax"] = _safe(bnd[i] / (ref[:, i].max() + EPS))
        out[f"{name}_pct"] = _safe((ref[:, i] < bnd[i]).mean())
    out["n_ref_windows"] = float(len(ref))
    return out


# ---------------------------------------------------------------------------
# Segment-descriptor comparisons. Idea credit: the 7th-place writeup of the
# original ADIA Lab challenge (after/before ratios of segment features,
# boundary windows at 10/30/60% of each segment, and the same features on the
# cumulative sum and absolute value of the series). Descriptor list, window
# convention and the ratio/difference rule are our own definitions.
# ---------------------------------------------------------------------------
_POS = ("std", "mad", "iqr", "meanabs", "diffstd", "dispersion_abs", "gini_abs", "hist_entropy",
        "perm_entropy", "turning_rate", "spec_entropy", "lowfreq_share", "range", "kurt_shift")
_SIGNED = ("mean", "median", "skew", "acf1", "acf1_abs", "slope", "zero_cross_rate")


def _perm_entropy3(x: np.ndarray) -> float:
    if len(x) < 4:
        return 0.0
    w = np.lib.stride_tricks.sliding_window_view(x, 3)
    code = np.argsort(w, axis=1) @ np.array([1, 3, 9])
    p = np.bincount(code, minlength=27).astype(float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(6))


def _descriptors(x: np.ndarray, lo: float, hi: float) -> dict:
    n = len(x)
    if n < 8:
        return {}
    med = np.median(x)
    sd = x.std()
    q25, q75 = np.percentile(x, [25, 75])
    ax = np.abs(x)
    d = np.diff(x)
    xc = x - x.mean()
    den = xc @ xc
    acf1 = float(xc[:-1] @ xc[1:] / den) if den > EPS else 0.0
    axc = ax - ax.mean()
    dena = axc @ axc
    s_ax = np.sort(ax)
    gini = float((2 * np.arange(1, n + 1) - n - 1) @ s_ax / (n * s_ax.sum() + EPS))
    h, _ = np.histogram(x, bins=16, range=(lo, hi) if hi > lo else None)
    ph = h[h > 0] / max(h.sum(), 1)
    ps = np.abs(np.fft.rfft(xc))[1:] ** 2
    pss = ps / (ps.sum() + EPS)
    t = np.arange(n, dtype=float) - (n - 1) / 2
    from scipy.stats import kurtosis, skew

    return {
        "std": sd, "mad": np.median(np.abs(x - med)), "iqr": q75 - q25, "meanabs": ax.mean(),
        "diffstd": d.std(), "dispersion_abs": ax.var() / (ax.mean() + EPS), "gini_abs": gini,
        "hist_entropy": float(-(ph * np.log(ph)).sum()), "perm_entropy": _perm_entropy3(x),
        "turning_rate": float(np.mean(d[:-1] * d[1:] < 0)) if n > 2 else 0.0,
        "spec_entropy": float(-(pss[pss > 0] * np.log(pss[pss > 0])).sum() / np.log(max(len(pss), 2))),
        "lowfreq_share": float(pss[: max(1, len(pss) // 10)].sum()), "range": x.max() - x.min(),
        "kurt_shift": kurtosis(x) + 3.0,
        "mean": x.mean(), "median": med, "skew": skew(x), "acf1": acf1,
        "acf1_abs": float(axc[:-1] @ axc[1:] / dena) if dena > EPS else 0.0,
        "slope": float(t @ x / (t @ t)), "zero_cross_rate": float(np.mean(np.diff(np.sign(x - med)) != 0)),
    }


def _compare(da: dict, db: dict, prefix: str, scale: float, out: dict) -> None:
    for k in _POS:
        if k in da and k in db:
            out[f"{prefix}{k}_ratio"] = _safe(db[k] / (da[k] + EPS))
    for k in _SIGNED:
        if k in da and k in db:
            v = db[k] - da[k]
            if k in ("mean", "median", "slope"):  # scale-free: in units of the series' spread
                v = v / (scale + EPS)
            out[f"{prefix}{k}_diff"] = _safe(v)


def _segment_and_windows(x: np.ndarray, periods: np.ndarray, out: dict, prefix: str,
                         segments: bool = True, windows=(0.1, 0.3, 0.6)) -> None:
    a, b = x[periods == 0], x[periods == 1]
    lo, hi = np.percentile(x, [0.5, 99.5])
    scale = x.std()
    if segments:
        _compare(_descriptors(a, lo, hi), _descriptors(b, lo, hi), f"{prefix}seg_", scale, out)
    for frac in windows:
        wa, wb = a[-max(8, int(frac * len(a))):], b[: max(8, int(frac * len(b)))]
        _compare(_descriptors(wa, lo, hi), _descriptors(wb, lo, hi), f"{prefix}w{int(frac * 100)}_", scale, out)


@register("seg_ratio")
def seg_ratio_group(values: np.ndarray, periods: np.ndarray) -> dict:
    out: dict = {}
    _segment_and_windows(values, periods, out, "raw_", segments=True, windows=())
    return out


@register("win_ratio")
def win_ratio_group(values: np.ndarray, periods: np.ndarray) -> dict:
    out: dict = {}
    _segment_and_windows(values, periods, out, "raw_", segments=False)
    return out


@register("xf_cumsum")
def cumsum_group(values: np.ndarray, periods: np.ndarray) -> dict:
    out: dict = {}
    _segment_and_windows(np.cumsum(values - values.mean()), periods, out, "cumsum_")
    return out


@register("xf_abs")
def abs_group(values: np.ndarray, periods: np.ndarray) -> dict:
    out: dict = {}
    _segment_and_windows(np.abs(values - np.median(values)), periods, out, "abs_")
    return out
