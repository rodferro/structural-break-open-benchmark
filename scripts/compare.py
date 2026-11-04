"""Paired comparison of nested-CV results (same outer folds) against a baseline.

    python -m scripts.compare lgb_cat lgb_cat_v2 lgb_cat_v3
"""

import json
import sys
from pathlib import Path

import numpy as np


def load(tag):
    return json.loads((Path("runs/results") / f"{tag}.json").read_text())


def main():
    base_tag, *others = sys.argv[1:]
    base = np.array(load(base_tag)["auc"]["stack"]["folds"])
    print(f"{'run':28s} {'stack AUC':>10s} {'delta vs ' + base_tag:>22s}")
    print(f"{base_tag:28s} {base.mean():10.4f}")
    for tag in others:
        r = np.array(load(tag)["auc"]["stack"]["folds"])
        d = r - base
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"{tag:28s} {r.mean():10.4f} {d.mean():+12.4f} ± {se:.4f} (paired SE)")


if __name__ == "__main__":
    main()
