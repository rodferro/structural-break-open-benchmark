"""Static audit of a cached feature table (training data only).

    python -m scripts.feature_audit --feature-set v2 > reports/private/feature_audit.md
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from scripts.build_features import base_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-set", default="v2")
    ap.add_argument("--cache", default=".cache")
    args = ap.parse_args()
    F = pd.read_parquet(base_path(Path(args.cache), args.feature_set))
    y = pd.read_parquet("y_train.parquet")["structural_breakpoint"].astype(int).loc[F.index]
    print(f"# Feature audit: {args.feature_set} ({F.shape[1]} columns, {len(F)} series)\n")

    nun = F.nunique()
    top_share = F.apply(lambda c: c.value_counts(normalize=True, dropna=False).iloc[0])
    print(f"## Constant / near-constant\n- constant: {int((nun <= 1).sum())} {list(F.columns[nun <= 1])[:20]}")
    nc = top_share[(top_share > 0.99) & (nun > 1)].sort_values(ascending=False)
    print(f"- >99% one value: {len(nc)} {[f'{k} ({v:.3f})' for k, v in nc.head(15).items()]}\n")

    h = F.apply(lambda c: pd.util.hash_pandas_object(c, index=False).sum())
    dup = h[h.duplicated(keep=False)]
    groups = dup.groupby(dup).apply(lambda s: list(s.index)).tolist()
    print(f"## Exact duplicates: {sum(len(g) - 1 for g in groups)} redundant columns in {len(groups)} groups")
    for g in groups[:25]:
        print(f"- {g}")

    R = np.corrcoef(F.rank().to_numpy().T)
    iu = np.triu_indices_from(R, 1)
    hi = [(F.columns[i], F.columns[j], R[i, j]) for i, j in zip(*iu, strict=True) if abs(R[i, j]) > 0.999]
    print(f"\n## |Spearman| > 0.999 pairs (incl. duplicates): {len(hi)}")
    fam = pd.Series([re.sub(r"_(a|b|g|diff|ratio)$", "", a) for a, _, _ in hi]).value_counts()
    print(f"- most affected families: {fam.head(15).to_dict()}\n")

    neg = F[y == 0]
    seg = {"len_a": neg["len_a"], "len_b": neg["len_b"], "len_total": neg["len_total"]}
    rows = []
    for c in F.columns:
        if c.startswith("len_"):
            continue
        r = {k: spearmanr(neg[c], v, nan_policy="omit").statistic for k, v in seg.items()}
        rows.append((c, max(r.values(), key=abs), max(r, key=lambda k: abs(r[k]))))
    L = pd.DataFrame(rows, columns=["feature", "rho", "vs"]).dropna()
    L = L.reindex(L["rho"].abs().sort_values(ascending=False).index)
    print("## Length dependence under the null (label-0 series only)")
    print(f"- features with |rho| > 0.5 vs a length: {int((L['rho'].abs() > 0.5).sum())}; > 0.3: {int((L['rho'].abs() > 0.3).sum())}")
    for _, r in L.head(30).iterrows():
        print(f"  - {r.feature}: rho={r.rho:+.2f} vs {r.vs}")

    print("\n## Saturation of -log10 p features (EPS=1e-8 caps them at 8)")
    for c in [c for c in F.columns if "neg_log10" in c]:
        print(f"- {c}: {(F[c] >= 7.99).mean():.1%} of series at the cap")
    for c in [c for c in F.columns if c.endswith("pvalue") or c.endswith("pvalue_abs")]:
        print(f"- {c}: {(F[c] < 1e-6).mean():.1%} below 1e-6 (raw p, squashed near 0)")

    print("\n## Label-0 vs label-1 segment lengths")
    print(F.groupby(y)[["len_a", "len_b", "len_total"]].describe().T.round(0).to_string())


if __name__ == "__main__":
    main()
