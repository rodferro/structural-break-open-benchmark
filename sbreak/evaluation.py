"""Honest (nested) cross-validation.

Outer loop: StratifiedKFold over series. For each outer fold, the *entire*
training procedure (feature selection, K-fold base models, stacker) is run on
the outer-train rows only, then scored on the held-out rows. This is the
estimate that should track the leaderboard.

``leaky_cv`` reproduces how submission #28 was validated (features chosen on
all rows, base models early-stopped on the fold they are scored on, stacker
scored in-sample), to quantify the optimism.

Per-fold results are cached under ``runs/`` (gitignored) so long runs can be
resumed and stacker/ensemble ablations can reuse base-model predictions.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .config import Config
from .features import add_interactions
from .pipeline import as_series
from .selection import select_features
from .stacking import StackedEnsemble

OUTER_SEED = 2024


def _hash(d: dict) -> str:
    return hashlib.sha1(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]


SELECTION_VERSION = 2  # 2: no second importance pass when there are no interactions


def selection_key(cfg: Config) -> dict:
    d = {
        k: getattr(cfg, k)
        for k in ("n_splits", "seed", "n_base", "n_interact", "n_final", "select_max_rounds", "lgb_params", "lgb_early_stopping")
    }
    d["selection_version"] = SELECTION_VERSION
    if cfg.selection_method != "cv_gain":  # only when changed, so existing cache keys stay valid
        d |= {k: getattr(cfg, k) for k in ("selection_method", "select_single_rounds", "select_single_params")}
    if cfg.exclude_features:  # only when set, so existing cache keys stay valid
        d["exclude_features"] = sorted(cfg.exclude_features)
    return d


def model_key(cfg: Config, model: str) -> dict:
    d = selection_key(cfg) | {"model": model, "early_stop_on_val_fold": cfg.early_stop_on_val_fold}
    prefix = {"lgb": "lgb_", "lgb5": "lgb5_", "cat": "cat_"}[model]
    d |= {k: v for k, v in cfg.to_dict().items() if k.startswith(prefix)}
    return d


class NestedCV:
    def __init__(self, F_raw: pd.DataFrame, y, cfg: Config, run_dir="runs/nested",
                 outer_splits: int = 5, outer_seed: int = OUTER_SEED, feature_set: str = "v1"):
        self.F_raw = F_raw
        self.y = as_series(y).loc[F_raw.index]
        self.cfg = cfg
        self.outer = list(
            StratifiedKFold(outer_splits, shuffle=True, random_state=outer_seed).split(F_raw, self.y)
        )
        self.dir = Path(run_dir) / f"{feature_set}_outer{outer_splits}_{outer_seed}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._check_fingerprint()

    def _check_fingerprint(self) -> None:
        """Cached files are keyed by config and fold only; make sure they belong to this data."""
        ids = pd.util.hash_pandas_object(self.y.reset_index(), index=False).to_numpy()
        # Hash the feature table too (names and values), so a change in features.py that keeps
        # the same columns cannot silently reuse selections/predictions computed on old features.
        cols = hashlib.sha1("\x1f".join(map(str, self.F_raw.columns)).encode()).hexdigest()[:16]
        vals = pd.util.hash_pandas_object(self.F_raw, index=True).to_numpy()
        fp = {"n_rows": int(len(self.y)), "hash": hashlib.sha1(ids.tobytes()).hexdigest()[:16],
              "n_features": int(self.F_raw.shape[1]), "columns": cols,
              "features": hashlib.sha1(vals.tobytes()).hexdigest()[:16]}
        path = self.dir / "data.json"
        if path.exists():
            old = json.loads(path.read_text())
            if "features" not in old and all(old.get(k) == fp[k] for k in ("n_rows", "hash", "n_features")):
                # cache written before feature hashing existed: adopt it, but say so
                warnings.warn(f"{self.dir}: cache predates feature hashing; assuming it matches the "
                              "current feature table", RuntimeWarning, stacklevel=2)
                path.write_text(json.dumps(fp))
                old = fp
            if old != fp:
                raise RuntimeError(f"{self.dir} holds results for different data ({old} != {fp}); use another run_dir")
        else:
            path.write_text(json.dumps(fp))

    # -- cached stages -------------------------------------------------------
    def _sel_path(self, k: int) -> Path:
        return self.dir / f"sel_{_hash(selection_key(self.cfg))}_fold{k}.json"

    def _pred_path(self, k: int, model: str) -> Path:
        return self.dir / f"pred_{model}_{_hash(model_key(self.cfg, model))}_fold{k}.npz"

    def _selection(self, k: int, trn_idx) -> tuple[list[str], list[str]]:
        path = self._sel_path(k)
        d = _read_cached(path, lambda p: json.loads(p.read_text()))
        if d is not None:
            return d["base"], d["final"]
        t = time.time()
        base, final = select_features(self.F_raw.iloc[trn_idx], self.y.iloc[trn_idx], self.cfg)
        _atomic_write(path, lambda p: p.write_text(json.dumps({"base": base, "final": final, "seconds": time.time() - t})))
        return base, final

    def _base_preds(self, k: int, trn_idx, val_idx, model: str) -> dict:
        """OOF (inner) predictions on outer-train and averaged predictions on outer-val."""
        path = self._pred_path(k, model)
        z = _read_cached(path, lambda p: dict(np.load(p)))
        if z is not None:
            return {"oof": z["oof"], "val": z["val"]}
        base, final = self._selection(k, trn_idx)
        F = add_interactions(self.F_raw[base], top_features=base, top_k=self.cfg.n_interact)[final]
        cfg = self.cfg.replace(models=(model,))
        t = time.time()
        ens = StackedEnsemble(cfg).fit(F.iloc[trn_idx], self.y.iloc[trn_idx])
        val = ens.predict_base(F.iloc[val_idx])[model]
        secs = time.time() - t
        _atomic_write(path, lambda p: np.savez(p, oof=ens.oof[model], val=val, seconds=secs))
        return {"oof": ens.oof[model], "val": val}

    # -- public API ----------------------------------------------------------
    def run(self, models: tuple[str, ...] | None = None, verbose: bool = True) -> dict:
        from .stacking import make_stacker

        models = tuple(models or self.cfg.models)
        y = self.y.to_numpy()
        pooled = {m: np.zeros(len(y)) for m in (*models, "stack")}
        fold_auc = {m: [] for m in (*models, "stack")}
        for k, (trn_idx, val_idx) in enumerate(self.outer):
            preds = {m: self._base_preds(k, trn_idx, val_idx, m) for m in models}
            Z_tr = np.column_stack([preds[m]["oof"] for m in models])
            Z_va = np.column_stack([preds[m]["val"] for m in models])
            stack = make_stacker(self.cfg).fit(Z_tr, y[trn_idx]).predict(Z_va)
            for m in models:
                pooled[m][val_idx] = preds[m]["val"]
                fold_auc[m].append(roc_auc_score(y[val_idx], preds[m]["val"]))
            pooled["stack"][val_idx] = stack
            fold_auc["stack"].append(roc_auc_score(y[val_idx], stack))
            if verbose:
                print(f"outer fold {k}: " + " ".join(f"{m}={fold_auc[m][-1]:.4f}" for m in fold_auc), flush=True)
        summary = {
            m: {"mean": float(np.mean(a)), "std": float(np.std(a)), "folds": [float(x) for x in a]}
            for m, a in fold_auc.items()
        }
        return {"models": list(models), "stacker": self.cfg.stacker, "auc": summary, "pooled": pooled,
                "compute_seconds": self.compute_seconds(models)}

    def compute_seconds(self, models) -> dict:
        """CPU-side time recorded when each cached stage was computed (not this call's wall time)."""
        out = {"selection": 0.0, **{m: 0.0 for m in models}}
        for k in range(len(self.outer)):
            d = _read_cached(self._sel_path(k), lambda p: json.loads(p.read_text()))
            out["selection"] += float(d.get("seconds", 0.0)) if d else 0.0
            for m in models:
                z = _read_cached(self._pred_path(k, m), lambda p: dict(np.load(p)))
                out[m] += float(z["seconds"]) if z is not None and "seconds" in z else 0.0
        out["total"] = sum(out.values())
        return {k: round(v, 3) for k, v in out.items()}


def _atomic_write(path: Path, write) -> None:
    """Write via a temp file + rename, so an interrupted run never leaves a truncated cache entry."""
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    write(tmp)
    os.replace(tmp, path)


def _read_cached(path: Path, read):
    """Load a cache entry; an unreadable one (e.g. from a killed run) is removed and recomputed."""
    if not path.exists():
        return None
    try:
        return read(path)
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"discarding unreadable cache entry {path.name}: {e!r}", RuntimeWarning, stacklevel=2)
        path.unlink(missing_ok=True)
        return None


def leaky_cv(F_raw: pd.DataFrame, y, cfg: Config) -> dict:
    """Validation protocol of submission #28 (optimistic by construction)."""
    y = as_series(y).loc[F_raw.index]
    base, final = select_features(F_raw, y, cfg)
    F = add_interactions(F_raw[base], top_features=base, top_k=cfg.n_interact)[final]
    ens = StackedEnsemble(cfg).fit(F, y)
    Z = np.column_stack([ens.oof[m] for m in cfg.models])
    out = {m: float(roc_auc_score(y, ens.oof[m])) for m in cfg.models}
    out["stack_in_sample"] = float(roc_auc_score(y, ens.stacker.predict(Z)))
    return out
