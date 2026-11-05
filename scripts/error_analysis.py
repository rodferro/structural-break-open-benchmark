"""Where does the model fail? Uses nested-CV out-of-fold predictions only.

    python -m scripts.error_analysis --tag baseline > reports/private/error_analysis.md

1. AUC within quantile bins of segment lengths, scale and volatility.
2. Confident mistakes: missed breaks (positives ranked low) vs caught breaks,
   false alarms (negatives ranked high) vs true negatives - which features
   separate them.
3. Residual-signal probe: cheap candidate statistics computed from the raw
   series, scored by AUC *within the model's uncertain region*. A candidate
   that separates the classes there carries information the model lacks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

from scripts.build_features import base_path

EPS = 1e-12


def auc(y, s) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, s))


def binned_auc(name, x, y, p, q=5):
    bins = pd.qcut(x.rank(method="first"), q, labels=False)
    rows = []
    for b in range(q):
        m = bins == b
        rows.append(f"{x[m].min():.4g}..{x[m].max():.4g}: AUC={auc(y[m], p[m]):.3f} (n={int(m.sum())}, pos={y[m].mean():.2f})")
    print(f"- **{name}**: " + " | ".join(rows))


# ---------------------------------------------------------------------------
# candidate statistics for the probe (raw series in, dict out)
# ---------------------------------------------------------------------------
def _acf(x, lag):
    x = x - x.mean()
    d = x @ x
    return float(x[:-lag] @ x[lag:] / d) if d > EPS and len(x) > lag + 2 else 0.0


def _gauss_ll(x):
    v = x.var()
    return -0.5 * len(x) * np.log(v + EPS)


def probe_stats(v: np.ndarray, p: np.ndarray) -> dict:
    a, b = v[p == 0], v[p == 1]
    n, na, nb = len(v), len(a), len(b)
    out = {}
    # shape after removing each segment's own location and scale
    za = (a - np.median(a)) / (np.median(np.abs(a - np.median(a))) + EPS)
    zb = (b - np.median(b)) / (np.median(np.abs(b - np.median(b))) + EPS)
    from scipy.stats import ks_2samp

    out["shape_ks_robust_std"] = ks_2samp(za, zb).statistic
    sa, sb = (a - a.mean()) / (a.std() + EPS), (b - b.mean()) / (b.std() + EPS)
    out["shape_ks_std"] = ks_2samp(sa, sb).statistic
    # dependence
    for lag in (1, 2, 5):
        out[f"acf_raw_l{lag}_absdiff"] = abs(_acf(b, lag) - _acf(a, lag))
        out[f"acf_sq_l{lag}_absdiff"] = abs(_acf(b * b, lag) - _acf(a * a, lag))
    # Gaussian likelihood ratio for a mean+variance change at the boundary, per point
    lr = _gauss_ll(a) + _gauss_ll(b) - _gauss_ll(v)
    out["lr_meanvar_per_n"] = lr / n
    out["lr_var_only_per_n"] = (
        -0.5 * na * np.log((a - v.mean()).var() + EPS) - 0.5 * nb * np.log((b - v.mean()).var() + EPS) - _gauss_ll(v)
    ) / n
    # scan: how special is the boundary among all split points?
    c = np.cumsum(v - v.mean())
    k = np.arange(1, n)
    w = np.sqrt(k * (n - k) / n)
    cus = np.abs(c[:-1]) / (w * (v.std() + EPS))  # standardised CUSUM of the mean
    out["cusum_mean_at_boundary_over_max"] = cus[na - 1] / (cus.max() + EPS)
    s2 = np.cumsum((v - v.mean()) ** 2)
    d_it = np.abs(s2[:-1] / s2[-1] - k / n) * np.sqrt(n / 2)  # Inclan-Tiao variance CUSUM
    out["cusum_var_at_boundary_over_max"] = d_it[na - 1] / (d_it.max() + EPS)
    out["cusum_var_at_boundary"] = d_it[na - 1]
    out["cusum_var_argmax_dist"] = abs(int(np.argmax(d_it)) + 1 - na) / n
    # spectral shape: low-frequency share of power, before vs after
    def lowshare(x):
        ps = np.abs(np.fft.rfft(x - x.mean()))[1:] ** 2
        return ps[: max(1, len(ps) // 8)].sum() / (ps.sum() + EPS)

    out["spec_lowshare_absdiff"] = abs(lowshare(b) - lowshare(a))
    # volatility clustering: autocorrelation of |x| at lag 1, before vs after
    out["volclust_absdiff"] = abs(_acf(np.abs(b), 1) - _acf(np.abs(a), 1))
    # tails beyond mean/variance
    from scipy.stats import kurtosis

    out["kurt_absdiff"] = abs(kurtosis(b) - kurtosis(a))
    return out


def compute_probes(X: pd.DataFrame) -> pd.DataFrame:
    X = X.sort_index()
    ids = X.index.get_level_values(0).to_numpy()
    vals, pers = X["value"].to_numpy(float), X["period"].to_numpy()
    starts = np.flatnonzero(np.r_[True, ids[1:] != ids[:-1]])
    ends = np.r_[starts[1:], len(ids)]

    def one(s, e):
        with np.errstate(all="ignore"):
            return probe_stats(vals[s:e], pers[s:e])

    rows = Parallel(n_jobs=-1, batch_size=64)(delayed(one)(s, e) for s, e in zip(starts, ends, strict=True))
    return pd.DataFrame(rows, index=pd.Index(ids[starts], name="id"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--feature-set", default="v2")
    ap.add_argument("--probe-cache", default=".cache/probes.parquet")
    args = ap.parse_args()

    oof = pd.read_parquet(f"runs/results/{args.tag}.oof.parquet")
    F = pd.read_parquet(base_path(Path(".cache"), args.feature_set)).loc[oof.index]
    y = pd.read_parquet("y_train.parquet")["structural_breakpoint"].astype(int).loc[oof.index]
    p = oof["stack"]
    print(f"# Error analysis: `{args.tag}` (nested-CV OOF, {len(y)} series)\n")
    print(f"pooled OOF AUC: stack={auc(y, p):.4f} " + " ".join(f"{c}={auc(y, oof[c]):.4f}" for c in oof.columns if c != "stack"))

    print("\n## 1. AUC by quintile")
    for name, x in {
        "len_a (before)": F["len_a"], "len_b (after)": F["len_b"], "len_total": F["len_total"],
        "len_b / len_a": F["len_ratio_b_over_a"], "scale: std (whole series)": F["std"],
        "volatility: diff_std_a": F["diff_std_a"], "kurtosis proxy: z_outlier_frac_3p0_g": F["z_outlier_frac_3p0_g"],
        "lag-1 autocorr (before)": F["acf1_a"],
    }.items():
        binned_auc(name, x, y, p)

    print("\n## 2. Confident mistakes")
    r = p.rank(pct=True)
    fn, tp = (y == 1) & (r < 0.4), (y == 1) & (r > 0.7)
    fp, tn = (y == 0) & (r > 0.8), (y == 0) & (r < 0.5)
    print(f"- missed breaks (positives below the 40th pct of scores): {int(fn.sum())} of {int(y.sum())} positives")
    print(f"- false alarms (negatives above the 80th pct): {int(fp.sum())} of {int((1 - y).sum())} negatives")
    sep = []
    for c in F.columns:
        x = F[c]
        sep.append((c, auc(np.r_[np.ones(fn.sum()), np.zeros(tp.sum())], np.r_[x[fn], x[tp]]),
                    auc(np.r_[np.ones(fp.sum()), np.zeros(tn.sum())], np.r_[x[fp], x[tn]])))
    S = pd.DataFrame(sep, columns=["feature", "missed_vs_caught", "falsealarm_vs_tn"]).dropna()
    S["d1"], S["d2"] = (S.missed_vs_caught - 0.5).abs(), (S.falsealarm_vs_tn - 0.5).abs()
    print("\nFeatures that best separate **missed** from **caught** breaks (AUC; <0.5 = lower in missed):")
    for _, row in S.sort_values("d1", ascending=False).head(15).iterrows():
        print(f"  - {row.feature}: {row.missed_vs_caught:.3f}")
    print("\nFeatures that best separate **false alarms** from true negatives:")
    for _, row in S.sort_values("d2", ascending=False).head(15).iterrows():
        print(f"  - {row.feature}: {row.falsealarm_vs_tn:.3f}")
    eff = F["cohens_d"].abs()
    vr = F["log_std_ratio"].abs()
    print(f"\n- missed breaks with |cohens_d| < 0.1 and |log std ratio| < 0.1 (no mean/variance change): "
          f"{float(((eff < 0.1) & (vr < 0.1))[fn].mean()):.1%}; among caught breaks: {float(((eff < 0.1) & (vr < 0.1))[tp].mean()):.1%}"
          f"; among negatives: {float(((eff < 0.1) & (vr < 0.1))[y == 0].mean()):.1%}")

    print("\n## 3. Residual-signal probe (candidate statistics from the raw series)")
    pc = Path(args.probe_cache)
    if pc.exists():
        P = pd.read_parquet(pc)
    else:
        P = compute_probes(pd.read_parquet("X_train.parquet"))
        P.to_parquet(pc)
    P = P.loc[y.index]
    unsure = (r > 0.25) & (r < 0.85)
    low = r < 0.5
    print(f"(AUC for the label; 'unsure' = middle of the score range, n={int(unsure.sum())}; 'low' = lower half, n={int(low.sum())}. "
          "0.5 = no extra signal. Partial: |corr| with the stack score)")
    rows = []
    for c in P.columns:
        rows.append((c, auc(y, P[c]), auc(y[unsure], P[c][unsure]), auc(y[low], P[c][low]),
                     abs(pd.Series(P[c]).corr(p, method="spearman"))))
    T = pd.DataFrame(rows, columns=["stat", "auc_all", "auc_unsure", "auc_low", "rho_score"])
    T["gain"] = (T.auc_unsure - 0.5).abs()
    print(T.sort_values("gain", ascending=False).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
