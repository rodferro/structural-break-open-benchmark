"""K-fold base models + level-2 stacker."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from .config import Config
from .models import MODEL_CLASSES


class LogisticStacker:
    """Logistic regression on base-model probabilities, stored as plain JSON."""

    kind = "logreg"

    def __init__(self, coef=None, intercept=0.0, seed: int = 99999):
        self.coef = None if coef is None else np.asarray(coef, dtype=float)
        self.intercept = float(intercept)
        self.seed = seed

    def fit(self, Z: np.ndarray, y) -> "LogisticStacker":
        lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=self.seed)
        lr.fit(Z, y)
        self.coef, self.intercept = lr.coef_.ravel(), float(lr.intercept_[0])
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(Z @ self.coef + self.intercept)))

    def to_dict(self):
        return {"kind": self.kind, "coef": self.coef.tolist(), "intercept": self.intercept}


class RankMeanStacker:
    """Average of per-model ranks (no fitted parameters)."""

    kind = "rank_mean"

    def fit(self, Z, y):
        return self

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return np.mean([rankdata(Z[:, j]) / len(Z) for j in range(Z.shape[1])], axis=0)

    def to_dict(self):
        return {"kind": self.kind}


def stacker_from_dict(d: dict):
    if d["kind"] == "logreg":
        return LogisticStacker(d["coef"], d["intercept"])
    if d["kind"] == "rank_mean":
        return RankMeanStacker()
    raise ValueError(d["kind"])


def make_stacker(cfg: Config):
    return LogisticStacker(seed=cfg.seed) if cfg.stacker == "logreg" else RankMeanStacker()


class StackedEnsemble:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.fold_models: dict[str, list] = {m: [] for m in cfg.models}
        self.oof: dict[str, np.ndarray] = {}
        self.stacker = None
        self.timings: dict[str, float] = {m: 0.0 for m in cfg.models}

    def _log(self, msg):
        if self.cfg.verbose:
            print(msg, flush=True)

    def fit(self, F: pd.DataFrame, y: pd.Series) -> "StackedEnsemble":
        cfg = self.cfg
        y = pd.Series(np.asarray(y).astype(int), index=F.index)
        folds = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
        self.oof = {m: np.zeros(len(F)) for m in cfg.models}

        for k, (trn_idx, val_idx) in enumerate(folds.split(F, y)):
            X_tr, y_tr = F.iloc[trn_idx], y.iloc[trn_idx]
            X_va, y_va = F.iloc[val_idx], y.iloc[val_idx]
            if cfg.early_stop_on_val_fold:
                X_fit, y_fit, X_es, y_es = X_tr, y_tr, X_va, y_va
            else:
                X_fit, X_es, y_fit, y_es = train_test_split(
                    X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=cfg.seed + k
                )
            msg = [f"fold {k}"]
            for name in cfg.models:
                t = time.time()
                model = MODEL_CLASSES[name](cfg)
                if name == "tabpfn":
                    model.fit(X_tr, y_tr)  # no early stopping; uses the full fold
                else:
                    model.fit(X_fit, y_fit, X_es, y_es)
                self.oof[name][val_idx] = model.predict(X_va)
                self.fold_models[name].append(model)
                self.timings[name] += time.time() - t
                msg.append(f"{name}={roc_auc_score(y_va, self.oof[name][val_idx]):.4f}")
            self._log(" | ".join(msg))

        Z = self._stack_matrix(self.oof)
        self.stacker = make_stacker(cfg).fit(Z, y)
        for name in cfg.models:
            self._log(f"{name:7s} OOF AUC {roc_auc_score(y, self.oof[name]):.5f}")
        self._log(f"stacker in-sample AUC on OOF {roc_auc_score(y, self.stacker.predict(Z)):.5f}")
        return self

    def _stack_matrix(self, preds: dict[str, np.ndarray]) -> np.ndarray:
        return np.column_stack([preds[m] for m in self.cfg.models])

    def predict_base(self, F: pd.DataFrame) -> dict[str, np.ndarray]:
        out = {}
        for name in self.cfg.models:
            models = self.fold_models[name]
            out[name] = np.mean([m.predict(F) for m in models], axis=0)
        return out

    def predict(self, F: pd.DataFrame) -> np.ndarray:
        return self.stacker.predict(self._stack_matrix(self.predict_base(F)))

    # -- persistence ---------------------------------------------------------
    def save(self, directory: Path) -> dict:
        files = {}
        for name, models in self.fold_models.items():
            files[name] = []
            for k, m in enumerate(models):
                files[name] += m.save(directory, f"fold_{k}")
        (directory / "stacker.json").write_text(json.dumps(self.stacker.to_dict(), indent=2))
        return files

    @classmethod
    def load(cls, cfg: Config, directory: Path) -> "StackedEnsemble":
        ens = cls(cfg)
        for name in cfg.models:
            ens.fold_models[name] = [
                MODEL_CLASSES[name].load(cfg, directory, f"fold_{k}") for k in range(cfg.n_splits)
            ]
        ens.stacker = stacker_from_dict(json.loads((directory / "stacker.json").read_text()))
        return ens
