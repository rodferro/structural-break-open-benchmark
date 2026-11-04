"""Paired comparison of nested-CV results (same outer folds) against a baseline.

    python -m scripts.compare baseline es_inner scan tests
    python -m scripts.compare baseline es_inner --markdown   # rows for reports/private/ablations.md

For each run: mean stack AUC, mean paired difference vs the baseline with its
standard error, the number of folds it wins, and the CPU time it took
(sum of the per-fold selection + model times recorded in the cache).
"""

import json
import sys
from pathlib import Path

import numpy as np

RESULTS = Path("runs/results")


def load(tag):
    return json.loads((RESULTS / f"{tag}.json").read_text())


def describe(r: dict) -> str:
    bits = [r.get("feature_set", "?")]
    if r.get("feature_groups"):
        bits.append("+" + "+".join(r["feature_groups"]))
    if r.get("config_overrides"):
        bits.append(" ".join(f"{k}={v}" for k, v in r["config_overrides"].items()))
    return " ".join(bits)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    md = "--markdown" in sys.argv
    base_tag, *others = args
    b = load(base_tag)
    base = np.array(b["auc"]["stack"]["folds"])
    if md:
        print("| run | config | features | stack AUC | Δ vs " + base_tag + " (± paired SE) | folds won | compute s |")
        print("|---|---|---|---|---|---|---|")
        print(f"| {base_tag} | {describe(b)} | {b.get('n_features', '')} | {base.mean():.4f} | — | — | {b.get('compute_seconds', {}).get('total', '')} |")
    else:
        print(f"{'run':22s} {'stack AUC':>9s} {'Δ vs ' + base_tag:>12s} {'± SE':>8s} {'wins':>5s} {'cpu s':>7s}")
        print(f"{base_tag:22s} {base.mean():9.4f}")
    for tag in others:
        r = load(tag)
        f = np.array(r["auc"]["stack"]["folds"])
        d = f - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        wins = int((d > 0).sum())
        if md:
            print(f"| {tag} | {describe(r)} | {r.get('n_features', '')} | {f.mean():.4f} | {d.mean():+.4f} ± {se:.4f} | {wins}/{len(d)} | {r.get('compute_seconds', {}).get('total', '')} |")
        else:
            print(f"{tag:22s} {f.mean():9.4f} {d.mean():+12.4f} {se:8.4f} {wins:>3d}/{len(d)} {r.get('compute_seconds', {}).get('total', ''):>7}")


if __name__ == "__main__":
    main()
