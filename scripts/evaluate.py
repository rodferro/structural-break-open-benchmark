"""Nested-CV evaluation / ablations. Results go to runs/ (gitignored).

    python -m scripts.evaluate --tag baseline                          # current default config
    python -m scripts.evaluate --groups lr_scan --tag lr_scan          # default + one feature group
    python -m scripts.evaluate --set early_stop_on_val_fold=false --tag es_inner
    python -m scripts.evaluate --preset submission28 --tag sub28       # submission #28
    python -m scripts.evaluate --preset submission28 --leaky           # #28's own (leaky) protocol

Writes runs/results/<tag>.json (fold AUCs, timings) and <tag>.oof.parquet
(pooled outer-fold predictions, for error analysis).
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

import sbreak  # noqa: F401
from sbreak.config import Config, submission28_config
from sbreak.evaluation import NestedCV, leaky_cv
from scripts.build_features import base_path, group_path


def parse_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def load_features(cache: Path, feature_set: str, groups) -> pd.DataFrame:
    F = pd.read_parquet(base_path(cache, feature_set))
    for g in groups:
        p = group_path(cache, g)
        if not p.exists():
            raise SystemExit(f"missing {p}; run: python -m scripts.build_features --groups {g}")
        G = pd.read_parquet(p)
        assert G.index.equals(F.index), g
        F = F.join(G)
    return F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["default", "submission28"], default="default")
    ap.add_argument("--cache", default=".cache")
    ap.add_argument("--groups", nargs="*", default=None, help="feature groups (overrides the config's)")
    ap.add_argument("--labels", default="y_train.parquet")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--set", nargs="*", help="Config overrides key=json_value")
    ap.add_argument("--outer-splits", type=int, default=5)
    ap.add_argument("--run-dir", default="runs/cpu", help="cache of per-fold selections/predictions")
    ap.add_argument("--leaky", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    base = submission28_config() if args.preset == "submission28" else Config()
    cfg = base.replace(verbose=0).replace(**parse_overrides(args.set))
    if args.models:
        cfg = cfg.replace(models=tuple(args.models))
    if args.groups is not None:
        cfg = cfg.replace(feature_groups=tuple(args.groups))
    fs = cfg.feature_set
    F = load_features(Path(args.cache), fs, cfg.feature_groups)
    y = pd.read_parquet(args.labels)["structural_breakpoint"]
    variant = "+".join([fs, *cfg.feature_groups])
    t = time.time()
    tag = args.tag or ("leaky" if args.leaky else variant)
    out = Path("runs/results")
    out.mkdir(parents=True, exist_ok=True)
    r = {}
    if args.leaky:
        res = {"leaky": leaky_cv(F, y, cfg)}
    else:
        r = NestedCV(F, y, cfg, run_dir=args.run_dir, outer_splits=args.outer_splits, feature_set=variant).run()
        res = {k: v for k, v in r.items() if k != "pooled"}
        pd.DataFrame(r["pooled"], index=F.index).to_parquet(out / f"{tag}.oof.parquet")
    res |= {"preset": args.preset, "config_overrides": parse_overrides(args.set), "feature_set": fs,
            "feature_groups": list(cfg.feature_groups), "n_features": int(F.shape[1]),
            "compute_seconds": r.get("compute_seconds", {}) if not args.leaky else {},
            "seconds": round(time.time() - t)}
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({m: round(v["mean"], 4) for m, v in res.get("auc", {}).items()} or res, indent=1))
    print(f"{tag}: {res['seconds']}s, {F.shape[1]} features", flush=True)


if __name__ == "__main__":
    main()
