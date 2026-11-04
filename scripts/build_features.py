"""Compute and cache feature tables for local experiments (.cache/, gitignored).

    python -m scripts.build_features --feature-set v2            # base table
    python -m scripts.build_features --groups lr_scan shape      # one cache per group

Only X_train is read by default; build times are written next to the caches.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import sbreak  # noqa: F401
from sbreak.features import build_feature_table


def base_path(out: Path, feature_set: str, split: str = "X_train") -> Path:
    return out / f"features_{split}{'' if feature_set == 'v1' else '.' + feature_set}.parquet"


def group_path(out: Path, group: str, split: str = "X_train") -> Path:
    return out / "groups" / f"{split}.{group}.parquet"


def _record(out: Path, key: str, seconds: float, shape) -> None:
    meta = out / "build_times.json"
    d = json.loads(meta.read_text()) if meta.exists() else {}
    d[key] = {"seconds": round(seconds, 1), "shape": list(shape)}
    meta.write_text(json.dumps(d, indent=2, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--out", default=".cache")
    ap.add_argument("--feature-set", default=None, help="build this base feature set")
    ap.add_argument("--groups", nargs="*", default=[], help="build these feature groups (one cache each)")
    ap.add_argument("--splits", nargs="+", default=["X_train"])
    args = ap.parse_args()
    out = Path(args.out)
    (out / "groups").mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        X = pd.read_parquet(Path(args.data_dir) / f"{split}.parquet")
        if args.feature_set:
            t = time.time()
            F = build_feature_table(X, feature_set=args.feature_set)
            F.to_parquet(base_path(out, args.feature_set, split))
            _record(out, f"{split}:{args.feature_set}", time.time() - t, F.shape)
            print(f"{split} {args.feature_set}: {F.shape} in {time.time() - t:.0f}s", flush=True)
        for g in args.groups:
            t = time.time()
            G = build_feature_table(X, groups=(g,), base=False)
            G.to_parquet(group_path(out, g, split))
            _record(out, f"{split}:group:{g}", time.time() - t, G.shape)
            print(f"{split} group {g}: {G.shape} in {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
