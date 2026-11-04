"""Hand-crafted features comparing the pre-boundary (period 0) and
post-boundary (period 1) segments of each series.

``extract_features_for_one_id`` is kept byte-for-byte equivalent to the
submission #28 version so that results stay comparable.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import scipy.stats
from joblib import Parallel, delayed


def extract_features_for_one_id(values: np.ndarray, periods: np.ndarray, feature_set: str = "v1") -> dict:
    """
    values: raw series values for a single id (1D)
    periods: 0/1 indicating before/after boundary aligned with values
    feature_set: "v1" = submission #28 (multi-context names embed the realised
        window lengths, giving ~17k mostly-empty columns); "v2" = same values
        named by nominal window (k, 2x, 3x), so every series has the same columns.
    """
    EPS = 1e-8

    values = np.asarray(values, dtype=float)
    periods = np.asarray(periods)

    a = values[periods == 0]
    b = values[periods == 1]

    feats = {}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def safe(x, default=0.0):
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return float(default)
        return x

    def mean_(x):
        return safe(np.mean(x))

    def median_(x):
        return safe(np.median(x))

    def std_(x):
        return safe(np.std(x))

    def var_(x):
        return safe(np.var(x))

    def min_(x):
        return safe(np.min(x))

    def max_(x):
        return safe(np.max(x))

    def slope_(x):
        x = np.asarray(x, dtype=float)
        if len(x) < 2:
            return 0.0
        t = np.arange(len(x), dtype=float)
        return safe(np.polyfit(t, x, 1)[0])

    def skew_(x):
        x = np.asarray(x, dtype=float)
        if len(x) < 3 or np.std(x) < EPS:
            return 0.0
        return safe(scipy.stats.skew(x, bias=False))

    def lag1_autocorr_(x):
        x = np.asarray(x, dtype=float)
        if len(x) < 3:
            return 0.0
        x0, x1 = x[:-1], x[1:]
        if np.std(x0) < EPS or np.std(x1) < EPS:
            return 0.0
        return safe(np.corrcoef(x0, x1)[0, 1])

    def diff_std_(x):
        x = np.asarray(x, dtype=float)
        if len(x) < 3:
            return 0.0
        return safe(np.std(np.diff(x)))

    def rolling_mean_(x, w):
        x = np.asarray(x, dtype=float)
        if len(x) < w:
            return np.full_like(x, np.nan)
        c = np.cumsum(np.insert(x, 0, 0.0))
        out = (c[w:] - c[:-w]) / w
        return np.concatenate([np.full(w - 1, np.nan), out])

    def rolling_std_(x, w):
        x = np.asarray(x, dtype=float)
        if len(x) < w:
            return np.full_like(x, np.nan)
        rm = rolling_mean_(x, w)
        rm2 = rolling_mean_(x * x, w)
        var = np.maximum(rm2 - rm * rm, 0.0)
        return np.sqrt(var)

    def quantiles_(x):
        x = np.asarray(x, dtype=float)
        if len(x) < 5:
            return {
                "q10": 0.0,
                "q50": 0.0,
                "q90": 0.0,
                "iqr": 0.0,
                "p90_10": 0.0,
            }
        q10, q25, q50, q75, q90 = np.percentile(x, [10, 25, 50, 75, 90])
        return {
            "q10": safe(q10),
            "q50": safe(q50),
            "q90": safe(q90),
            "iqr": safe(q75 - q25),
            "p90_10": safe(q90 - q10),
        }

    def basic_stats_(x):
        mu = mean_(x)
        med = median_(x)
        sd = std_(x)
        return {
            "mean": mu,
            "median": med,
            "max": max_(x),
            "min": min_(x),
            "std": sd,
            "var": var_(x),
            "skew": skew_(x),
            "mean_norm": safe(mu / (sd + EPS)),
            "median_norm": safe(med / (sd + EPS)),
            "slope": slope_(x),
        }

    def add_prefixed_stats(x, suffix):
        stats = basic_stats_(x)
        for k, v in stats.items():
            feats[f"{k}{suffix}"] = v

    def add_prefixed_quantiles(x, suffix):
        qs = quantiles_(x)
        for k, v in qs.items():
            feats[f"{k}{suffix}"] = v

    def compare_windows(x_left, x_right, prefix):
        """
        Compare right window vs left window.
        """
        m_left = mean_(x_left)
        m_right = mean_(x_right)
        med_left = median_(x_left)
        med_right = median_(x_right)
        sd_left = std_(x_left)
        sd_right = std_(x_right)
        pooled = safe(np.sqrt((var_(x_left) + var_(x_right)) / 2.0) + EPS)

        mean_diff = safe(m_right - m_left)

        feats[f"{prefix}_mean_diff"] = mean_diff
        feats[f"{prefix}_std_diff"] = safe(sd_right - sd_left)
        feats[f"{prefix}_med_diff"] = safe(med_right - med_left)
        feats[f"{prefix}_cohens_d"] = safe(mean_diff / pooled)

    def add_local_boundary_features():
        for k in (3, 5, 10, 20, 50, 100, 256):
            kk = min(k, len(a), len(b))
            a_tail = a[-kk:]
            b_head = b[:kk]

            local_mean = safe(np.mean(b_head) - np.mean(a_tail))
            local_med = safe(np.median(b_head) - np.median(a_tail))
            pooled = safe(np.sqrt((np.var(a_tail) + np.var(b_head)) / 2.0) + EPS)

            feats[f"local_mean_diff_k{k}"] = local_mean
            feats[f"local_mean_diff_k{k}_over_std"] = safe(local_mean / (feats["std"] + EPS))
            feats[f"local_mean_abs_diff_k{k}"] = safe(abs(local_mean))
            feats[f"local_median_diff_k{k}"] = local_med
            feats[f"local_cohens_d_k{k}"] = safe(local_mean / pooled)

    def add_multicontext_features():
        for k in (20, 60, 120, 256, 500, 1000):
            k1 = min(k, len(a), len(b))
            if k1 < 5:
                continue

            bh = b[:k1]
            a1 = a[-k1:]
            a2 = a[-min(2 * k1, len(a)):]
            a3 = a[-min(3 * k1, len(a)):]

            m_b = mean_(bh)
            s_b = std_(bh)
            med_b = median_(bh)

            if feature_set == "v1":
                n1, n2, n3 = f"b{k1}_a{k1}", f"b{k1}_a{len(a2)}", f"b{k1}_a{len(a3)}"
            else:
                n1, n2, n3 = f"b{k}_a1x", f"b{k}_a2x", f"b{k}_a3x"

            feats[f"mc_mean_{n1}"] = safe(m_b - mean_(a1))
            feats[f"mc_mean_{n2}"] = safe(m_b - mean_(a2))
            feats[f"mc_mean_{n3}"] = safe(m_b - mean_(a3))

            feats[f"mc_std_{n1}"] = safe(s_b - std_(a1))
            feats[f"mc_std_{n2}"] = safe(s_b - std_(a2))
            feats[f"mc_std_{n3}"] = safe(s_b - std_(a3))

            feats[f"mc_med_{n1}"] = safe(med_b - median_(a1))

    def add_derived_block(vx, name):
        ax = vx[periods == 0]
        bx = vx[periods == 1]

        ax = ax[np.isfinite(ax)]
        bx = bx[np.isfinite(bx)]

        if len(ax) < 5 or len(bx) < 5:
            return

        feats[f"{name}_mean_diff"] = safe(np.mean(bx) - np.mean(ax))
        feats[f"{name}_std_diff"] = safe(np.std(bx) - np.std(ax))
        feats[f"{name}_med_diff"] = safe(np.median(bx) - np.median(ax))
        feats[f"{name}_jump"] = safe(bx[0] - ax[-1])

        for k in (10, 20, 50, 100):
            kk = min(k, len(ax), len(bx))
            at = ax[-kk:]
            bh = bx[:kk]
            feats[f"{name}_local_mean_k{k}"] = safe(np.mean(bh) - np.mean(at))
            feats[f"{name}_local_std_k{k}"] = safe(np.std(bh) - np.std(at))

    def winsorized_mean(x, lower=0.05, upper=0.95):
        x = np.asarray(x, dtype=float)
        if len(x) == 0:
            return 0.0
        lo, hi = np.quantile(x, [lower, upper])
        return safe(np.mean(np.clip(x, lo, hi)))

    def add_outlier_features(feats, x, suffix):
        x = np.asarray(x, dtype=float)
        n = len(x)

        if n == 0:
            for thr in (2.5, 3.0, 4.0):
                tag = str(thr).replace(".", "p")
                feats[f"z_outlier_count_{tag}{suffix}"] = 0.0
                feats[f"z_outlier_frac_{tag}{suffix}"] = 0.0
                feats[f"z_pos_outlier_count_{tag}{suffix}"] = 0.0
                feats[f"z_neg_outlier_count_{tag}{suffix}"] = 0.0
            for thr in (3.5, 5.0, 7.0):
                tag = str(thr).replace(".", "p")
                feats[f"mad_outlier_count_{tag}{suffix}"] = 0.0
                feats[f"mad_outlier_frac_{tag}{suffix}"] = 0.0
                feats[f"mad_pos_outlier_count_{tag}{suffix}"] = 0.0
                feats[f"mad_neg_outlier_count_{tag}{suffix}"] = 0.0
            feats[f"winsor_mean{suffix}"] = 0.0
            feats[f"trimmed_mean_10pct{suffix}"] = 0.0
            feats[f"upper_tail_excess_95{suffix}"] = 0.0
            feats[f"lower_tail_excess_05{suffix}"] = 0.0
            return

        mu = np.mean(x)
        sd = np.std(x)
        if sd < EPS:
            z = np.zeros(n, dtype=float)
        else:
            z = (x - mu) / (sd + EPS)

        med = np.median(x)
        mad = np.median(np.abs(x - med))
        if mad < EPS:
            mz = np.zeros(n, dtype=float)
        else:
            mz = 0.67448975 * (x - med) / (mad + EPS)

        for thr in (2.5, 3.0, 4.0):
            mask = np.abs(z) > thr
            pos_mask = z > thr
            neg_mask = z < -thr
            tag = str(thr).replace(".", "p")

            feats[f"z_outlier_count_{tag}{suffix}"] = safe(np.sum(mask))
            feats[f"z_outlier_frac_{tag}{suffix}"] = safe(np.mean(mask))
            feats[f"z_pos_outlier_count_{tag}{suffix}"] = safe(np.sum(pos_mask))
            feats[f"z_neg_outlier_count_{tag}{suffix}"] = safe(np.sum(neg_mask))

        for thr in (3.5, 5.0, 7.0):
            mask = np.abs(mz) > thr
            pos_mask = mz > thr
            neg_mask = mz < -thr
            tag = str(thr).replace(".", "p")

            feats[f"mad_outlier_count_{tag}{suffix}"] = safe(np.sum(mask))
            feats[f"mad_outlier_frac_{tag}{suffix}"] = safe(np.mean(mask))
            feats[f"mad_pos_outlier_count_{tag}{suffix}"] = safe(np.sum(pos_mask))
            feats[f"mad_neg_outlier_count_{tag}{suffix}"] = safe(np.sum(neg_mask))

        feats[f"winsor_mean{suffix}"] = winsorized_mean(x)

        trim = int(0.1 * n)
        if 2 * trim < n:
            xs = np.sort(x)
            feats[f"trimmed_mean_10pct{suffix}"] = safe(np.mean(xs[trim:n - trim]))
        else:
            feats[f"trimmed_mean_10pct{suffix}"] = safe(np.mean(x))

        q05, q95 = np.quantile(x, [0.05, 0.95])
        feats[f"upper_tail_excess_95{suffix}"] = safe(np.mean(np.maximum(x - q95, 0.0)))
        feats[f"lower_tail_excess_05{suffix}"] = safe(np.mean(np.maximum(q05 - x, 0.0)))

    def add_outlier_comparisons(feats, name):
        feats[f"{name}_diff"] = safe(feats[f"{name}_b"] - feats[f"{name}_a"])
        feats[f"{name}_ratio"] = safe((feats[f"{name}_b"] + EPS) / (feats[f"{name}_a"] + EPS))

    def direction_ratio_(x):
        x = np.asarray(x, dtype=float)
        if len(x) < 2:
            return 0.0
        d = np.diff(x)
        return safe(np.mean(d > 0))
    
    def chow_f_stat_(x, split):
        x = np.asarray(x, dtype=float)
        n = len(x)
        if split < 2 or (n - split) < 2 or n < 6:
            return 0.0
    
        t = np.arange(n, dtype=float)
    
        coef = np.polyfit(t, x, 1)
        pred = coef[0] * t + coef[1]
        rss_p = np.sum((x - pred) ** 2)
    
        t_a = t[:split]
        y_a = x[:split]
        coef_a = np.polyfit(t_a, y_a, 1)
        pred_a = coef_a[0] * t_a + coef_a[1]
        rss_a = np.sum((y_a - pred_a) ** 2)
    
        t_b = t[split:]
        y_b = x[split:]
        coef_b = np.polyfit(t_b, y_b, 1)
        pred_b = coef_b[0] * t_b + coef_b[1]
        rss_b = np.sum((y_b - pred_b) ** 2)
    
        k = 2
        denom_df = n - 2 * k
        if denom_df <= 0:
            return 0.0
    
        numerator = (rss_p - (rss_a + rss_b)) / k
        denominator = (rss_a + rss_b) / (denom_df + EPS)
    
        return safe(numerator / (denominator + EPS))

    # ------------------------------------------------------------------
    # Length features
    # ------------------------------------------------------------------
    feats["len_total"] = float(len(values))
    feats["len_a"] = float(len(a))
    feats["len_b"] = float(len(b))
    feats["len_ratio_b_over_a"] = safe((len(b) + EPS) / (len(a) + EPS))

    # ------------------------------------------------------------------
    # Basic stats
    # ------------------------------------------------------------------
    add_prefixed_stats(a, "_a")
    add_prefixed_stats(b, "_b")
    add_prefixed_stats(values, "")

    # ------------------------------------------------------------------
    # Shift / difference features
    # ------------------------------------------------------------------
    for name in [
        "mean",
        "median",
        "max",
        "min",
        "std",
        "var",
        "skew",
        "mean_norm",
        "median_norm",
        "slope",
    ]:
        feats[f"{name}_diff"] = safe(feats[f"{name}_b"] - feats[f"{name}_a"])

    feats["diff_std_a"] = diff_std_(a)
    feats["diff_std_b"] = diff_std_(b)
    feats["diff_std_diff"] = safe(feats["diff_std_b"] - feats["diff_std_a"])

    # ------------------------------------------------------------------
    # Ratio features
    # ------------------------------------------------------------------
    feats["std_ratio"] = safe((feats["std_b"] + EPS) / (feats["std_a"] + EPS))
    feats["var_ratio"] = safe((feats["var_b"] + EPS) / (feats["var_a"] + EPS))
    feats["diff_std_ratio"] = safe((feats["diff_std_b"] + EPS) / (feats["diff_std_a"] + EPS))

    feats["log_std_ratio"] = safe(np.log(feats["std_b"] + EPS) - np.log(feats["std_a"] + EPS))
    feats["log_var_ratio"] = safe(np.log(feats["var_b"] + EPS) - np.log(feats["var_a"] + EPS))

    # ------------------------------------------------------------------
    # Effect sizes / normalized changes
    # ------------------------------------------------------------------
    pooled_ab = safe(np.sqrt((feats["var_a"] + feats["var_b"]) / 2.0) + EPS)
    mean_diff = safe(feats["mean_b"] - feats["mean_a"])

    feats["cohens_d"] = safe(mean_diff / pooled_ab)
    feats["mean_diff_over_std"] = safe(mean_diff / (feats["std"] + EPS))

    # ------------------------------------------------------------------
    # Boundary features
    # ------------------------------------------------------------------
    jump = safe(b[0] - a[-1])
    feats["boundary_jump"] = jump
    feats["boundary_jump_over_std"] = safe(jump / (feats["std"] + EPS))

    add_local_boundary_features()
    add_multicontext_features()

    # ------------------------------------------------------------------
    # Dynamics
    # ------------------------------------------------------------------
    feats["acf1_a"] = lag1_autocorr_(a)
    feats["acf1_b"] = lag1_autocorr_(b)
    feats["acf1_diff"] = safe(feats["acf1_b"] - feats["acf1_a"])

    ms_a = safe(np.mean(a * a))
    ms_b = safe(np.mean(b * b))
    feats["ms_a"] = ms_a
    feats["ms_b"] = ms_b
    feats["ms_ratio"] = safe((ms_b + EPS) / (ms_a + EPS))
    feats["log_ms_ratio"] = safe(np.log(ms_b + EPS) - np.log(ms_a + EPS))

    energy_a = safe(np.sum(a * a))
    energy_b = safe(np.sum(b * b))
    feats["energy_a"] = energy_a
    feats["energy_b"] = energy_b
    feats["energy_ratio"] = safe((energy_b + EPS) / (energy_a + EPS))
    feats["log_energy_ratio"] = safe(np.log(energy_b + EPS) - np.log(energy_a + EPS))

    mu = np.mean(values)
    cs = np.cumsum(values - mu)
    cus = safe(np.max(np.abs(cs)))
    feats["cusum_maxabs"] = cus
    feats["cusum_maxabs_norm"] = safe(cus / ((np.std(values) + EPS) * np.sqrt(len(values) + EPS)))

    feats["trend_ratio_a"] = direction_ratio_(a)
    feats["trend_ratio_b"] = direction_ratio_(b)
    feats["trend_ratio_diff"] = safe(feats["trend_ratio_b"] - feats["trend_ratio_a"])
    
    chow_f = chow_f_stat_(values, len(a))
    feats["chow_f_stat"] = chow_f
    feats["log_chow_f"] = safe(np.log1p(max(chow_f, 0.0)))

    # ------------------------------------------------------------------
    # Statistical tests / distances
    # ------------------------------------------------------------------
    t_res = scipy.stats.ttest_ind(a, b, equal_var=False)
    feats["t_pvalue"] = safe(t_res.pvalue, default=1.0)
    feats["neg_log10_p"] = safe(-np.log10(feats["t_pvalue"] + EPS))

    ks = scipy.stats.ks_2samp(a, b, alternative="two-sided", mode="auto")
    feats["ks_stat"] = safe(ks.statistic)
    feats["ks_neg_log10_p"] = safe(-np.log10(float(ks.pvalue) + EPS))

    feats["levene_pvalue"] = safe(scipy.stats.levene(a, b).pvalue, default=1.0)

    aa = np.abs(a)
    bb = np.abs(b)

    ks_abs = scipy.stats.ks_2samp(aa, bb, alternative="two-sided", mode="auto")
    feats["ks_stat_abs"] = safe(ks_abs.statistic)
    feats["ks_neg_log10_p_abs"] = safe(-np.log10(float(ks_abs.pvalue) + EPS))
    feats["levene_pvalue_abs"] = safe(scipy.stats.levene(aa, bb).pvalue, default=1.0)

    w = scipy.stats.wasserstein_distance(a, b)
    feats["wasserstein"] = safe(w)
    feats["wasserstein_over_std"] = safe(w / (feats["std"] + EPS))

    # ------------------------------------------------------------------
    # Quantiles
    # ------------------------------------------------------------------
    add_prefixed_quantiles(a, "_a")
    add_prefixed_quantiles(b, "_b")
    add_prefixed_quantiles(values, "_g")

    feats["q10_diff"] = safe(feats["q10_b"] - feats["q10_a"])
    feats["q50_diff"] = safe(feats["q50_b"] - feats["q50_a"])
    feats["q90_diff"] = safe(feats["q90_b"] - feats["q90_a"])
    feats["iqr_diff"] = safe(feats["iqr_b"] - feats["iqr_a"])
    feats["p90_10_diff"] = safe(feats["p90_10_b"] - feats["p90_10_a"])

    feats["abs_q10_diff"] = safe(abs(feats["q10_diff"]))
    feats["abs_q50_diff"] = safe(abs(feats["q50_diff"]))
    feats["abs_q90_diff"] = safe(abs(feats["q90_diff"]))
    feats["abs_iqr_diff"] = safe(abs(feats["iqr_diff"]))
    
    feats["log_p90_10_ratio"] = safe(np.log(feats["p90_10_b"] + EPS) - np.log(feats["p90_10_a"] + EPS))
    feats["q50_diff_over_iqr_a"] = safe(feats["q50_diff"] / (feats["iqr_a"] + EPS))

    # ------------------------------------------------------------------
    # Transformed-series blocks
    # ------------------------------------------------------------------
    z = (values - np.mean(values)) / (np.std(values) + EPS)
    add_derived_block(z, "z")

    rank = scipy.stats.rankdata(values, method="dense") / (len(values) + EPS)
    add_derived_block(rank, "rank")

    abs_values = np.abs(values)
    add_derived_block(abs_values, "abs")

    for k in (15, 30, 60, 120):
        rm = rolling_mean_(values, k)
        add_derived_block(rm, f"rm{k}")

        rs = rolling_std_(values, k)
        add_derived_block(rs, f"rs{k}")

    # ------------------------------------------------------------------
    # Outlier features
    # ------------------------------------------------------------------
    add_outlier_features(feats, a, "_a")
    add_outlier_features(feats, b, "_b")
    add_outlier_features(feats, values, "_g")

    # ------------------------------------------------------------------
    # Outlier comparisons
    # ------------------------------------------------------------------
    outlier_base_names = [
        "winsor_mean",
        "trimmed_mean_10pct",
        "upper_tail_excess_95",
        "lower_tail_excess_05",
        "z_outlier_count_2p5",
        "z_outlier_frac_2p5",
        "z_pos_outlier_count_2p5",
        "z_neg_outlier_count_2p5",
        "z_outlier_count_3p0",
        "z_outlier_frac_3p0",
        "z_pos_outlier_count_3p0",
        "z_neg_outlier_count_3p0",
        "z_outlier_count_4p0",
        "z_outlier_frac_4p0",
        "z_pos_outlier_count_4p0",
        "z_neg_outlier_count_4p0",
        "mad_outlier_count_3p5",
        "mad_outlier_frac_3p5",
        "mad_pos_outlier_count_3p5",
        "mad_neg_outlier_count_3p5",
        "mad_outlier_count_5p0",
        "mad_outlier_frac_5p0",
        "mad_pos_outlier_count_5p0",
        "mad_neg_outlier_count_5p0",
        "mad_outlier_count_7p0",
        "mad_outlier_frac_7p0",
        "mad_pos_outlier_count_7p0",
        "mad_neg_outlier_count_7p0",
    ]
    for name in outlier_base_names:
        add_outlier_comparisons(feats, name)

    feats["winsor_mean_diff"] = safe(feats["winsor_mean_b"] - feats["winsor_mean_a"])
    feats["winsor_mean_diff_over_std"] = safe(feats["winsor_mean_diff"] / (feats["std"] + EPS))

    # ------------------------------------------------------------------
    # Jump / diff outlier features
    # ------------------------------------------------------------------
    d_all = np.diff(values)
    d_a = np.diff(a)
    d_b = np.diff(b)

    def add_jump_outlier_features(feats, d, suffix):
        d = np.asarray(d, dtype=float)
        if len(d) == 0:
            feats[f"max_abs_jump{suffix}"] = 0.0
            feats[f"jump_p95{suffix}"] = 0.0
            feats[f"jump_outlier_count_3p0{suffix}"] = 0.0
            feats[f"jump_outlier_frac_3p0{suffix}"] = 0.0
            return

        abs_d = np.abs(d)
        feats[f"max_abs_jump{suffix}"] = safe(np.max(abs_d))
        feats[f"jump_p95{suffix}"] = safe(np.quantile(abs_d, 0.95))

        mu = np.mean(abs_d)
        sd = np.std(abs_d)
        if sd < EPS:
            z = np.zeros(len(abs_d), dtype=float)
        else:
            z = (abs_d - mu) / (sd + EPS)

        mask = z > 3.0
        feats[f"jump_outlier_count_3p0{suffix}"] = safe(np.sum(mask))
        feats[f"jump_outlier_frac_3p0{suffix}"] = safe(np.mean(mask))

    add_jump_outlier_features(feats, d_a, "_a")
    add_jump_outlier_features(feats, d_b, "_b")
    add_jump_outlier_features(feats, d_all, "_g")

    feats["max_abs_jump_diff"] = safe(feats["max_abs_jump_b"] - feats["max_abs_jump_a"])
    feats["jump_p95_diff"] = safe(feats["jump_p95_b"] - feats["jump_p95_a"])
    feats["jump_outlier_count_3p0_diff"] = safe(
        feats["jump_outlier_count_3p0_b"] - feats["jump_outlier_count_3p0_a"]
    )
    feats["jump_outlier_frac_3p0_diff"] = safe(
        feats["jump_outlier_frac_3p0_b"] - feats["jump_outlier_frac_3p0_a"]
    )

    # ------------------------------------------------------------------
    # Boundary-local outlier behavior
    # ------------------------------------------------------------------
    for k in (10, 25, 50):
        kk = min(k, len(a), len(b))
        a_tail = a[-kk:]
        b_head = b[:kk]

        add_outlier_features(feats, a_tail, f"_a_tail_k{k}")
        add_outlier_features(feats, b_head, f"_b_head_k{k}")

        feats[f"z_outlier_count_3p0_local_diff_k{k}"] = safe(
            feats[f"z_outlier_count_3p0_b_head_k{k}"] - feats[f"z_outlier_count_3p0_a_tail_k{k}"]
        )
        feats[f"mad_outlier_count_5p0_local_diff_k{k}"] = safe(
            feats[f"mad_outlier_count_5p0_b_head_k{k}"] - feats[f"mad_outlier_count_5p0_a_tail_k{k}"]
        )
        feats[f"winsor_mean_local_diff_k{k}"] = safe(
            feats[f"winsor_mean_b_head_k{k}"] - feats[f"winsor_mean_a_tail_k{k}"]
        )
    
    return feats


def _features_for_group(_id, values: np.ndarray, periods: np.ndarray, feature_set: str = "v1"):
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        base_set = "v2" if feature_set == "v3" else feature_set
        feats = extract_features_for_one_id(values, periods, base_set)
        if feature_set == "v3":
            from .extra_features import extra_features

            feats.update(extra_features(np.asarray(values, dtype=float), np.asarray(periods)))
        return _id, feats


def build_feature_table(X: pd.DataFrame, n_jobs: int | None = None, feature_set: str = "v1") -> pd.DataFrame:
    """Compute per-series features.

    X has a MultiIndex (id, time) and columns ['value', 'period'].
    Returns a DataFrame indexed by id (sorted), one row per series.
    Feature extraction is unsupervised and per-series, so it can be computed
    once for train+test without leaking labels.
    """
    if not isinstance(X.index, pd.MultiIndex):
        raise ValueError("Expected X.index to be a MultiIndex (id, time)")
    for col in ["value", "period"]:
        if col not in X.columns:
            raise ValueError(f"Missing required column: {col}")

    X = X.sort_index()
    ids = X.index.get_level_values(0).to_numpy()
    values = X["value"].to_numpy(dtype=float)
    periods = X["period"].to_numpy()
    # group boundaries (X is sorted by id)
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]

    if n_jobs is None:
        n_jobs = int(os.environ.get("SBREAK_N_JOBS", "-1"))
    jobs = (delayed(_features_for_group)(ids[s], values[s:e], periods[s:e], feature_set) for s, e in zip(starts, ends, strict=True))
    if n_jobs == 1 or len(starts) < 64:
        out = [_features_for_group(ids[s], values[s:e], periods[s:e], feature_set) for s, e in zip(starts, ends, strict=True)]
    else:
        out = Parallel(n_jobs=n_jobs, batch_size=32)(jobs)

    index = pd.Index([o[0] for o in out], name="id")
    return pd.DataFrame([o[1] for o in out], index=index)


def add_interactions(
    F: pd.DataFrame,
    top_features: list[str],
    top_k: int = 20,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """Pairwise products/ratios among the ``top_k`` first features."""
    selected = list(top_features)[:top_k]
    new_cols = {}
    for i in range(len(selected)):
        for j in range(i + 1, len(selected)):
            f1, f2 = selected[i], selected[j]
            x = F[f1].values
            y = F[f2].values
            new_cols[f"{f1}_mul_{f2}"] = x * y
            new_cols[f"{f1}_div_{f2}"] = x / (y + eps)
            new_cols[f"{f2}_div_{f1}"] = y / (x + eps)
            new_cols[f"{f1}_absmul_{f2}"] = np.abs(x * y)
    interactions = pd.DataFrame(new_cols, index=F.index)
    return pd.concat([F, interactions], axis=1)


def prepare_for_tabular_model(F: pd.DataFrame) -> pd.DataFrame:
    """Replace inf with NaN and fill NaN with 0 (CatBoost input)."""
    return F.replace([np.inf, -np.inf], np.nan).fillna(0.0)
