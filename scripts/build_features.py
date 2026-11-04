"""Compute and cache feature tables for local experiments (written to .cache/, gitignored)."""

import argparse
import time
from pathlib import Path

import pandas as pd

from sbreak.features import build_feature_table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out", default=".cache")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for name in ("X_train", "X_test.reduced"):
        src = Path(args.data_dir) / f"{name}.parquet"
        if not src.exists():
            continue
        t = time.time()
        F = build_feature_table(pd.read_parquet(src))
        F.to_parquet(out / f"features_{name}.parquet")
        print(f"{name}: {F.shape} in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
