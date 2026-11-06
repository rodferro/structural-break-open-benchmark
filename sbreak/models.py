"""Base learners with version-robust persistence.

No fitted estimator is ever pickled:

* LightGBM -> native text model (``Booster.save_model``)
* CatBoost -> native ``.cbm`` (``save_model``)

(Submission #28 also used TabPFN, pickled with joblib; that is what broke when
tabpfn was upgraded between training and inference. TabPFN has since been
dropped: it needs a GPU to be practical and added little to the stack.)
"""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import lightgbm as lgb
import numpy as np

from .config import Config
from .features import prepare_for_tabular_model

TARGET_COL = "__target__"


def _cuda_available() -> bool:
    try:
        from catboost.utils import get_gpu_device_count

        return get_gpu_device_count() > 0
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
class LGBMultiModel:
    """Average of several LightGBM configurations (Config.lgb5_configs), each early-stopped.

    Model seeds are offset by Config.seed so seed replicates differ.
    """

    name = "lgb5"

    def __init__(self, cfg: Config, boosters: list | None = None):
        self.cfg = cfg
        self.boosters = boosters or []

    def fit(self, X_tr, y_tr, X_es, y_es):
        self.boosters = []
        for c in self.cfg.lgb5_configs:
            params = {"objective": "binary", "metric": "auc", "verbosity": -1, **c}
            params["seed"] = int(c.get("seed", 0)) + int(self.cfg.seed)
            b = lgb.train(
                params,
                lgb.Dataset(X_tr, label=y_tr),
                num_boost_round=self.cfg.lgb5_max_rounds,
                valid_sets=[lgb.Dataset(X_es, label=y_es)],
                callbacks=[lgb.early_stopping(self.cfg.lgb5_early_stopping, verbose=False)],
            )
            self.boosters.append(b)
        return self

    def predict(self, X) -> np.ndarray:
        return np.mean([b.predict(X, num_iteration=b.best_iteration or None) for b in self.boosters], axis=0)

    def save(self, directory: Path, tag: str) -> list[str]:
        files = []
        for j, b in enumerate(self.boosters):
            fname = f"lgb5_{tag}_m{j}.txt"
            b.save_model(str(directory / fname), num_iteration=b.best_iteration or None)
            files.append(fname)
        return files

    @classmethod
    def load(cls, cfg: Config, directory: Path, tag: str) -> "LGBMultiModel":
        boosters = [lgb.Booster(model_file=str(directory / f"lgb5_{tag}_m{j}.txt")) for j in range(len(cfg.lgb5_configs))]
        return cls(cfg, boosters)


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


MODEL_CLASSES = {c.name: c for c in (LGBModel, LGBMultiModel, CatModel)}


def library_versions() -> dict[str, str]:
    out = {}
    for pkg in ("lightgbm", "catboost", "scikit-learn", "numpy", "pandas", "scipy", "pyarrow"):
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
