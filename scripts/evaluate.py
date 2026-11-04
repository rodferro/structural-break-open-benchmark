"""Nested-CV evaluation / ablations. Results go to runs/ (gitignored).

    python -m scripts.evaluate --models lgb cat                        # current default config
    python -m scripts.evaluate --preset submission28 --models lgb cat  # submission #28
    python -m scripts.evaluate --models lgb cat tabpfn --set tabpfn_max_context=3000
    python -m scripts.evaluate --preset submission28 --leaky           # #28's own (leaky) protocol
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import sbreak  # noqa: F401  (import order: torch before lightgbm)
from sbreak.config import Config, submission28_config
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
    ap.add_argument("--preset", choices=["default", "submission28"], default="default")
    ap.add_argument("--features", default=None, help="feature cache (default: .cache/features_X_train[.<set>].parquet)")
    ap.add_argument("--labels", default="y_train.parquet")
    ap.add_argument("--models", nargs="+", default=["lgb", "tabpfn", "cat"])
    ap.add_argument("--set", nargs="*", help="Config overrides key=json_value")
    ap.add_argument("--outer-splits", type=int, default=5)
    ap.add_argument("--leaky", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    base = submission28_config() if args.preset == "submission28" else Config()
    cfg = base.replace(models=tuple(args.models), verbose=0).replace(**parse_overrides(args.set))
    fs = cfg.feature_set
    path = args.features or f".cache/features_X_train{'' if fs == 'v1' else '.' + fs}.parquet"
    F = pd.read_parquet(path)
    y = pd.read_parquet(args.labels)["structural_breakpoint"]
    t = time.time()
    if args.leaky:
        res = {"leaky": leaky_cv(F, y, cfg)}
    else:
        r = NestedCV(F, y, cfg, outer_splits=args.outer_splits, feature_set=fs).run()
        res = {k: v for k, v in r.items() if k != "pooled"}
    res |= {"preset": args.preset, "config_overrides": parse_overrides(args.set), "feature_set": fs,
            "seconds": round(time.time() - t)}
    tag = args.tag or ("leaky" if args.leaky else "_".join(args.models))
    out = Path("runs/results")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res.get("auc", res), indent=1))


if __name__ == "__main__":
    main()
