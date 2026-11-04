"""Pipeline configuration.

``Config()`` is the current default; ``submission28_config()`` reproduces submission #28.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass
class Config:
    # cross-validation used to build the stacking OOF predictions
    n_splits: int = 5
    seed: int = 99999

    # "v1" = submission #28 features; "v2" = length-independent names (see features.py)
    feature_set: str = "v2"
    # optional additive groups from sbreak/feature_groups.py (each ablatable alone)
    feature_groups: tuple[str, ...] = ()

    # two-stage LightGBM-importance feature selection
    # Nested CV: dropping the interaction stage helps (submission #28: 80 / 20 / 100)
    n_base: int = 200  # top-k raw features kept after stage 1
    n_interact: int = 0  # top-k of those used to build pairwise interactions (0 = none)
    n_final: int = 200  # top-k features kept after stage 2
    select_max_rounds: int = 1_000_000

    # base models included in the stack
    models: tuple[str, ...] = ("lgb", "cat")

    # LightGBM
    lgb_params: dict = field(
        default_factory=lambda: {
            "verbosity": -1,
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",
            "learning_rate": 0.03,
        }
    )
    lgb_max_rounds: int = 1_000_000
    lgb_early_stopping: int = 500

    # CatBoost
    cat_iterations: int = 5000
    cat_learning_rate: float = 0.03
    cat_depth: int = 6
    cat_l2_leaf_reg: float = 5.0
    cat_early_stopping: int = 500
    cat_task_type: str = "auto"  # "auto" -> GPU when CUDA is available, else CPU

    # stacker
    stacker: str = "logreg"  # "logreg" | "rank_mean"

    # True (submission #28's behaviour) early-stops LightGBM/CatBoost on the same
    # fold whose OOF predictions then train the stacker, which biases the stacker
    # toward the early-stopped models. Nested-CV outer scores stay honest either way.
    # False early-stops on an inner split of the fold's training rows instead.
    # Kept at True until an ablation shows False is no worse:
    #   python -m scripts.evaluate --models lgb cat --set early_stop_on_val_fold=false
    early_stop_on_val_fold: bool = True

    verbose: int = 500

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["models"] = list(self.models)
        d["feature_groups"] = list(self.feature_groups)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = dict(d)
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        for k in ("models", "feature_groups"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)

    def replace(self, **kw) -> "Config":
        return dataclasses.replace(self, **kw)


def submission28_config() -> Config:
    """The configuration behind submission #28 (public LB 0.8089)."""
    return Config(feature_set="v1", n_base=80, n_interact=20, n_final=100)


def smoke_config() -> Config:
    """Tiny configuration for CI: seconds, not hours."""
    return Config(
        n_splits=2,
        n_base=12,
        n_interact=4,
        n_final=16,
        select_max_rounds=50,
        lgb_max_rounds=50,
        lgb_early_stopping=10,
        cat_iterations=50,
        cat_early_stopping=10,
        verbose=0,
    )
