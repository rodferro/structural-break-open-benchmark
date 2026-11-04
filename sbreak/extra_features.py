"""Additional two-sample / dependence features (feature_set "v3" = v2 + these)."""

from __future__ import annotations

import numpy as np
import scipy.stats

EPS = 1e-12


def _safe(x, default=0.0) -> float:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if not np.isfinite(x) else x


def _nlog10p(p) -> float:
    return _safe(-np.log10(max(_safe(p, 1.0), 1e-300)))


def _acf(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 2:
        return 0.0
    x = x - x.mean()
    d = np.dot(x, x)
    return _safe(np.dot(x[:-lag], x[lag:]) / d) if d > EPS else 0.0


def _band_power(x: np.ndarray, n_bands: int = 4) -> np.ndarray:
    if len(x) < 16:
        return np.zeros(n_bands)
    p = np.abs(np.fft.rfft(x - x.mean())) ** 2
    p = p[1:]
    tot = p.sum()
    if tot <= EPS:
        return np.zeros(n_bands)
    return np.array([b.sum() / tot for b in np.array_split(p, n_bands)])


def extra_features(values: np.ndarray, periods: np.ndarray) -> dict:
    a = values[periods == 0]
    b = values[periods == 1]
    f: dict[str, float] = {}
    if len(a) < 10 or len(b) < 10:
        return f

    # location / distribution tests
    f["x_mwu_nlogp"] = _nlog10p(scipy.stats.mannwhitneyu(a, b).pvalue)
    f["x_cvm_stat"] = _safe(scipy.stats.cramervonmises_2samp(a, b).statistic)
    sd = np.std(values) + EPS
    f["x_energy_over_std"] = _safe(scipy.stats.energy_distance(a, b) / sd)
    try:
        ad = scipy.stats.anderson_ksamp([a, b])
        f["x_ad_stat"] = _safe(ad.statistic)
    except Exception:
        f["x_ad_stat"] = 0.0

    # scale tests (robust)
    f["x_ansari_nlogp"] = _nlog10p(scipy.stats.ansari(a, b).pvalue)
    f["x_fligner_nlogp"] = _nlog10p(scipy.stats.fligner(a, b).pvalue)
    mad_a = scipy.stats.median_abs_deviation(a) + EPS
    mad_b = scipy.stats.median_abs_deviation(b) + EPS
    f["x_log_mad_ratio"] = _safe(np.log(mad_b / mad_a))
    f["x_abs_log_mad_ratio"] = abs(f["x_log_mad_ratio"])
    f["x_abs_log_std_ratio"] = abs(_safe(np.log((np.std(b) + EPS) / (np.std(a) + EPS))))

    # higher moments
    ka, kb = scipy.stats.kurtosis(a), scipy.stats.kurtosis(b)
    f["x_kurt_a"], f["x_kurt_b"] = _safe(ka), _safe(kb)
    f["x_kurt_diff"] = _safe(kb - ka)
    f["x_abs_skew_diff"] = abs(_safe(scipy.stats.skew(b) - scipy.stats.skew(a)))

    # dependence structure: raw, |x|, x^2 autocorrelations at several lags
    for name, fa, fb in (("raw", a, b), ("abs", np.abs(a), np.abs(b)), ("sq", a * a, b * b)):
        for lag in (1, 2, 3, 5, 10):
            ra, rb = _acf(fa, lag), _acf(fb, lag)
            f[f"x_acf_{name}_l{lag}_diff"] = rb - ra
            f[f"x_acf_{name}_l{lag}_absdiff"] = abs(rb - ra)

    # spectral shape
    pa, pb = _band_power(a), _band_power(b)
    for i in range(len(pa)):
        f[f"x_band{i}_diff"] = _safe(pb[i] - pa[i])
    f["x_band_l1"] = _safe(np.abs(pb - pa).sum())

    # standardised tail exceedance of b w.r.t. a's distribution
    lo, hi = np.quantile(a, [0.01, 0.99])
    f["x_b_exceed_a_1pct"] = _safe(np.mean((b < lo) | (b > hi)))
    return f
