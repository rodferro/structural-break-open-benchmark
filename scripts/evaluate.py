"""Nested-CV evaluation / ablations. Results go to runs/ (gitignored).

    python -m scripts.evaluate --models lgb cat --tag lgb_cat
    python -m scripts.evaluate --models lgb cat tabpfn --set tabpfn_max_context=3000
    python -m scripts.evaluate --leaky            # submission #28's own validation protocol
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import sbreak  # noqa: F401  (import order: torch before lightgbm)
from sbreak.config import Config
from sbreak.evaluation import NestedCV, leaky_cv


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=".cache/features_X_train.parquet")
    ap.add_argument("--feature-set", default="v1")
    ap.add_argument("--labels", default="y_train.parquet")
    ap.add_argument("--models", nargs="+", default=["lgb", "tabpfn", "cat"])
    ap.add_argument("--set", nargs="*", help="Config overrides key=json_value")
    ap.add_argument("--outer-splits", type=int, default=5)
    ap.add_argument("--leaky", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    F = pd.read_parquet(args.features)
    y = pd.read_parquet(args.labels)["structural_breakpoint"]
    cfg = Config(models=tuple(args.models), verbose=0).replace(**parse_overrides(args.set))
    t = time.time()
    if args.leaky:
        res = {"leaky": leaky_cv(F, y, cfg)}
    else:
        r = NestedCV(F, y, cfg, outer_splits=args.outer_splits, feature_set=args.feature_set).run()
        res = {k: v for k, v in r.items() if k != "pooled"}
    res |= {"config_overrides": parse_overrides(args.set), "feature_set": args.feature_set,
            "seconds": round(time.time() - t)}
    tag = args.tag or ("leaky" if args.leaky else "_".join(args.models))
    out = Path("runs/results")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res.get("auc", res), indent=1))


if __name__ == "__main__":
    main()
