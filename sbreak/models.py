"""Base learners with version-robust persistence.

No fitted estimator is ever pickled:

* LightGBM -> native text model (``Booster.save_model``)
* CatBoost -> native ``.cbm`` (``save_model``)
* TabPFN   -> its *training context* (features + labels as parquet) and the
  checkpoint file; the classifier is re-fitted at inference. TabPFN's "fit" is
  in-context learning (preprocessing + storing the data), so refitting is
  cheap and deterministic, and it never depends on private attributes of a
  torch ``nn.Module`` surviving a library upgrade.
"""

from __future__ import annotations

import json
import shutil
import warnings
from importlib.metadata import version
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import Config
from .features import prepare_for_tabular_model

TARGET_COL = "__target__"


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # pragma: no cover
        return False


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------
class LGBModel:
    name = "lgb"

    def __init__(self, cfg: Config, booster: lgb.Booster | None = None):
        self.cfg = cfg
        self.booster = booster

    def fit(self, X_tr, y_tr, X_es, y_es):
        self.booster = lgb.train(
            self.cfg.lgb_params,
            lgb.Dataset(X_tr, label=y_tr),
            num_boost_round=self.cfg.lgb_max_rounds,
            valid_sets=[lgb.Dataset(X_es, label=y_es)],
            callbacks=[
                lgb.log_evaluation(self.cfg.verbose) if self.cfg.verbose else lgb.log_evaluation(0),
                lgb.early_stopping(self.cfg.lgb_early_stopping, verbose=bool(self.cfg.verbose)),
            ],
        )
        return self

    def predict(self, X) -> np.ndarray:
        it = self.booster.best_iteration or None
        return self.booster.predict(X, num_iteration=it)

    def save(self, directory: Path, tag: str) -> list[str]:
        fname = f"lgb_{tag}.txt"
        self.booster.save_model(str(directory / fname), num_iteration=self.booster.best_iteration or None)
        return [fname]

    @classmethod
    def load(cls, cfg: Config, directory: Path, tag: str) -> "LGBModel":
        booster = lgb.Booster(model_file=str(directory / f"lgb_{tag}.txt"))
        return cls(cfg, booster)


# ---------------------------------------------------------------------------
# CatBoost
# ---------------------------------------------------------------------------
class CatModel:
    name = "cat"

    def __init__(self, cfg: Config, model=None):
        self.cfg = cfg
        self.model = model

    def _task_type(self) -> str:
        t = self.cfg.cat_task_type
        if t == "auto":
            return "GPU" if _cuda_available() else "CPU"
        return t

    def fit(self, X_tr, y_tr, X_es, y_es):
        from catboost import CatBoostClassifier

        self.model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=self.cfg.cat_iterations,
            learning_rate=self.cfg.cat_learning_rate,
            depth=self.cfg.cat_depth,
            l2_leaf_reg=self.cfg.cat_l2_leaf_reg,
            random_seed=self.cfg.seed,
            task_type=self._task_type(),
            verbose=self.cfg.verbose or False,
            allow_writing_files=False,
        )
        self.model.fit(
            prepare_for_tabular_model(X_tr),
            y_tr,
            eval_set=(prepare_for_tabular_model(X_es), y_es),
            early_stopping_rounds=self.cfg.cat_early_stopping,
            verbose=self.cfg.verbose or False,
        )
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict_proba(prepare_for_tabular_model(X))[:, 1]

    def save(self, directory: Path, tag: str) -> list[str]:
        fname = f"cat_{tag}.cbm"
        self.model.save_model(str(directory / fname))
        return [fname]

    @classmethod
    def load(cls, cfg: Config, directory: Path, tag: str) -> "CatModel":
        from catboost import CatBoostClassifier

        m = CatBoostClassifier()
        m.load_model(str(directory / f"cat_{tag}.cbm"))
        return cls(cfg, m)


# ---------------------------------------------------------------------------
# TabPFN (refit-at-inference)
# ---------------------------------------------------------------------------
TABPFN_WEIGHTS_DIR = "tabpfn_weights"


def _default_tabpfn_checkpoint() -> Path | None:
    try:
        from tabpfn.model_loading import resolve_model_path

        paths, *_ = resolve_model_path(None, which="classifier")
        return paths[0]
    except Exception:  # pragma: no cover - API differs across versions
        return None


class TabPFNModel:
    name = "tabpfn"

    def __init__(self, cfg: Config, model_path: str | Path = "auto"):
        self.cfg = cfg
        self.model_path = model_path
        self.X_ctx: pd.DataFrame | None = None
        self.y_ctx: np.ndarray | None = None
        self.clf = None

    def _device(self) -> str:
        d = self.cfg.tabpfn_device
        if d == "auto":
            # MPS (Apple) runs out of memory at this context size; prefer CUDA, else CPU.
            return "cuda" if _cuda_available() else "cpu"
        return d

    def _build(self):
        from tabpfn import TabPFNClassifier

        device = self._device()
        return TabPFNClassifier(
            n_estimators=self.cfg.tabpfn_n_estimators,
            device=device,
            random_state=0,
            fit_mode=self.cfg.tabpfn_fit_mode,
            model_path=str(self.model_path),
        )

    def _subsample(self, X: pd.DataFrame, y) -> tuple[pd.DataFrame, np.ndarray]:
        y = np.asarray(y)
        n = self.cfg.tabpfn_max_context
        if n is None or len(X) <= n:
            return X, y
        from sklearn.model_selection import train_test_split

        idx, _ = train_test_split(np.arange(len(X)), train_size=n, stratify=y, random_state=self.cfg.seed)
        idx = np.sort(idx)
        return X.iloc[idx], y[idx]

    def fit(self, X_tr, y_tr, X_es=None, y_es=None):
        X_ctx, y_ctx = self._subsample(prepare_for_tabular_model(X_tr), y_tr)
        self.X_ctx, self.y_ctx = X_ctx, y_ctx
        self._refit()
        return self

    def _refit(self):
        self.clf = self._build()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.clf.fit(self.X_ctx, self.y_ctx)

    def predict(self, X) -> np.ndarray:
        if self.clf is None:
            self._refit()
        Xp = prepare_for_tabular_model(X)
        return self.clf.predict_proba(Xp)[:, 1]

    def save(self, directory: Path, tag: str) -> list[str]:
        fname = f"tabpfn_{tag}_context.parquet"
        ctx = self.X_ctx.copy()
        ctx[TARGET_COL] = self.y_ctx
        ctx.to_parquet(directory / fname)
        files = [fname]
        # Ship the checkpoint so inference needs neither network nor HF Hub.
        src = Path(self.model_path) if str(self.model_path) != "auto" else _default_tabpfn_checkpoint()
        if src is not None and src.exists():
            wdir = directory / TABPFN_WEIGHTS_DIR
            wdir.mkdir(exist_ok=True)
            dst = wdir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            files.append(f"{TABPFN_WEIGHTS_DIR}/{src.name}")
        return files

    @classmethod
    def load(cls, cfg: Config, directory: Path, tag: str) -> "TabPFNModel":
        wdir = directory / TABPFN_WEIGHTS_DIR
        ckpts = sorted(wdir.glob("*.ckpt")) if wdir.exists() else []
        m = cls(cfg, model_path=ckpts[0] if ckpts else "auto")
        ctx = pd.read_parquet(directory / f"tabpfn_{tag}_context.parquet")
        m.y_ctx = ctx.pop(TARGET_COL).to_numpy()
        m.X_ctx = ctx
        return m  # refit lazily on first predict


MODEL_CLASSES = {c.name: c for c in (LGBModel, CatModel, TabPFNModel)}


def library_versions() -> dict[str, str]:
    out = {}
    for pkg in ("tabpfn", "tabpfn-common-utils", "torch", "lightgbm", "catboost", "scikit-learn", "numpy", "pandas", "scipy"):
        try:
            out[pkg] = version(pkg)
        except Exception:
            out[pkg] = "missing"
    return out


def check_versions(recorded: dict[str, str]) -> list[str]:
    """Return human-readable mismatches between recorded and installed versions."""
    now = library_versions()
    return [f"{k}: trained with {v}, running {now.get(k)}" for k, v in recorded.items() if now.get(k) != v]


def dump_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))
