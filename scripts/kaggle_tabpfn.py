"""Run the TabPFN part of the nested CV on a GPU (e.g. a Kaggle T4 notebook).

TabPFN at the submission's real configuration (full fold context, 8
estimators) is impractical on a laptop CPU, so this script computes only the
TabPFN cache files of ``sbreak.evaluation.NestedCV``. Copy the resulting
``runs/nested`` folder back into the local repo; ``scripts.evaluate`` then
stacks them with the LightGBM/CatBoost predictions already cached there.

On Kaggle (GPU T4, internet on), with the bundle from
``scripts/make_kaggle_bundle.py`` attached as a dataset and the competition
parquet files available::

    !pip install -q -r /kaggle/input/<bundle>/requirements.txt
    !cd /kaggle/input/<bundle> && python -m scripts.kaggle_tabpfn \
        --data-dir /kaggle/input/<data> --out /kaggle/working/runs
    # then download /kaggle/working/runs.zip

Feature selection is read from the bundled ``sel_*.json`` files (feature
names only), so the outer folds and selected features are identical to the
local runs; nothing derived from the data has to be uploaded.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

import sbreak  # noqa: F401  (import order: torch before lightgbm)
from sbreak.config import Config
from sbreak.evaluation import NestedCV
from sbreak.features import build_feature_table

# (tag, feature_set, config overrides) - must match the local runs being compared
RUNS = {
    "v1_baseline": ("v1", {}),
    "v3_noint": ("v3", {"n_interact": 0, "n_base": 200, "n_final": 200}),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder with X_train.parquet and y_train.parquet")
    ap.add_argument("--out", default="runs", help="writable runs/ directory")
    ap.add_argument("--runs", nargs="+", default=list(RUNS))
    ap.add_argument("--n-estimators", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N series (dry runs)")
    args = ap.parse_args()

    import torch

    if args.device == "cuda":
        if not torch.cuda.is_available():
            sys.exit("No CUDA device: enable the GPU accelerator in the notebook settings.")
        print("GPU:", torch.cuda.get_device_name(0), flush=True)

    out = Path(args.out)
    bundled = Path(__file__).resolve().parents[1] / "runs" / "nested"
    if bundled.exists() and bundled.resolve() != (out / "nested").resolve():
        shutil.copytree(bundled, out / "nested", dirs_exist_ok=True)  # selection files

    d = Path(args.data_dir)
    X = pd.read_parquet(d / "X_train.parquet")
    y = pd.read_parquet(d / "y_train.parquet")["structural_breakpoint"]
    if args.limit:
        ids = X.index.get_level_values("id").unique()[: args.limit]
        X, y = X.loc[ids], y.loc[ids]

    features: dict[str, pd.DataFrame] = {}
    for tag in args.runs:
        fs, overrides = RUNS[tag]
        if fs not in features:
            t = time.time()
            features[fs] = build_feature_table(X, feature_set=fs)
            print(f"features {fs}: {features[fs].shape} in {time.time() - t:.0f}s", flush=True)
        cfg = Config(models=("tabpfn",), verbose=0, tabpfn_device=args.device,
                     tabpfn_n_estimators=args.n_estimators).replace(**overrides)
        cv = NestedCV(features[fs], y, cfg, run_dir=out / "nested", feature_set=fs)
        t = time.time()
        res = cv.run(models=("tabpfn",))
        print(tag, json.dumps({k: round(v["mean"], 4) for k, v in res["auc"].items()}),
              f"{time.time() - t:.0f}s", flush=True)

    shutil.make_archive(str(out), "zip", root_dir=out.parent, base_dir=out.name)
    print("wrote", f"{out}.zip")


if __name__ == "__main__":
    main()
