"""Local stand-in for the CrunchDAO runner.

Mimics what the platform does with a submission:

1. import the submission's ``main.py`` as a module,
2. call ``train(X_train, y_train, model_directory_path)`` (unless --skip-train),
3. in a *fresh* process-like state, call ``infer(X_test, model_directory_path)``,
4. check the output (one finite score per test id), optionally twice for
   determinism, write ``prediction/prediction.parquet`` and score it if
   labels are available.

Usage::

    python -m sbreak.runner                      # full train + infer on ./*.parquet
    python -m sbreak.runner --skip-train         # infer only, reuse resources/
    python -m sbreak.runner --synthetic --smoke  # no competition data needed
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def load_submission(main_path: Path):
    spec = importlib.util.spec_from_file_location("submission_main", main_path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(main_path.parent))
    spec.loader.exec_module(mod)
    for fn in ("train", "infer"):
        if not callable(getattr(mod, fn, None)):
            raise SystemExit(f"{main_path} does not define {fn}()")
    return mod


def validate_prediction(pred, X_test: pd.DataFrame) -> pd.Series:
    ids = X_test.index.get_level_values("id").unique().sort_values()
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    if not isinstance(pred, pd.Series):
        pred = pd.Series(np.asarray(pred).ravel(), index=ids)
    pred = pred.sort_index()
    if len(pred) != len(ids) or not pred.index.equals(pd.Index(ids)):
        raise AssertionError(f"prediction has {len(pred)} rows / wrong ids; expected {len(ids)}")
    if not np.isfinite(pred.to_numpy(dtype=float)).all():
        raise AssertionError("prediction contains NaN/inf")
    pred.index.name = "id"
    return pred.rename("prediction")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", default="main.py", help="submission entry point")
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--model-dir", default="resources")
    ap.add_argument("--prediction-dir", default="prediction")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--determinism-check", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="use generated data instead of parquet files")
    ap.add_argument("--smoke", action="store_true", help="tiny model config (sets SBREAK_CONFIG=smoke)")
    args = ap.parse_args(argv)

    import os

    if args.smoke:
        os.environ["SBREAK_CONFIG"] = "smoke"

    if args.synthetic:
        from .synthetic import make_dataset

        X_train, y_train = make_dataset(160, seed=1)
        X_test, y_test = make_dataset(40, seed=2)
    else:
        d = Path(args.data_dir)
        X_train = pd.read_parquet(d / "X_train.parquet")
        y_train = pd.read_parquet(d / "y_train.parquet")["structural_breakpoint"]
        X_test = pd.read_parquet(d / "X_test.reduced.parquet")
        yp = d / "y_test.reduced.parquet"
        y_test = pd.read_parquet(yp)["structural_breakpoint"] if yp.exists() else None

    sub = load_submission(Path(args.main).resolve())
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        t = time.time()
        sub.train(X_train, y_train, str(model_dir))
        print(f"[runner] train done in {time.time() - t:.0f}s", flush=True)

    t = time.time()
    pred = validate_prediction(sub.infer(X_test, str(model_dir)), X_test)
    print(f"[runner] infer done in {time.time() - t:.0f}s ({len(pred)} ids)", flush=True)

    if args.determinism_check:
        again = validate_prediction(sub.infer(X_test, str(model_dir)), X_test)
        diff = float(np.max(np.abs(again.to_numpy() - pred.to_numpy())))
        print(f"[runner] determinism: max |diff| = {diff:.2e}")
        if diff > 1e-6:
            raise AssertionError("infer() is not deterministic")

    out = Path(args.prediction_dir)
    out.mkdir(parents=True, exist_ok=True)
    pred.to_frame().to_parquet(out / "prediction.parquet")

    if y_test is not None:
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(y_test.astype(int).loc[pred.index], pred)
        print(f"[runner] local ROC AUC on {len(pred)} test ids: {auc:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
