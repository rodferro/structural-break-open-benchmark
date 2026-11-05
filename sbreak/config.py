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
    # optional additive groups from sbreak/feature_groups.py (each ablatable alone).
    # Default = the groups adopted after nested CV with three seeds.
    feature_groups: tuple[str, ...] = ("dispersion", "cumsum_dist", "compress")
    # base columns dropped before selection: names, or presets from sbreak/exclusions.py
    exclude_features: tuple[str, ...] = ()

    # two-stage LightGBM-importance feature selection
    # Nested CV: dropping the interaction stage helps (submission #28: 80 / 20 / 100)
    n_base: int = 200  # top-k raw features kept after stage 1
    n_interact: int = 0  # top-k of those used to build pairwise interactions (0 = none)
    n_final: int = 200  # top-k features kept after stage 2
    select_max_rounds: int = 1_000_000
    # "cv_gain": mean gain over n_splits early-stopped LightGBM fits (default)
    # "single_gain": gain of one fixed-length LightGBM fit on the training rows
    selection_method: str = "cv_gain"
    select_single_rounds: int = 100
    select_single_params: dict = field(
        default_factory=lambda: {"objective": "binary", "num_leaves": 31, "learning_rate": 0.05, "verbosity": -1}
    )

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

    # "lgb5": average of several LightGBM configurations (each early-stopped);
    # the default list mirrors the 5-model ensemble of the public reference
    # solution compared in reports/private (hyperparameters only)
    lgb5_configs: tuple = (
        {"seed": 42, "num_leaves": 63, "learning_rate": 0.03, "reg_alpha": 0.1, "reg_lambda": 0.1},
        {"seed": 123, "num_leaves": 127, "learning_rate": 0.03, "reg_alpha": 0.3, "reg_lambda": 0.3},
        {"seed": 456, "num_leaves": 31, "learning_rate": 0.05, "reg_alpha": 0.0, "reg_lambda": 0.0},
        {"seed": 789, "num_leaves": 63, "learning_rate": 0.03, "subsample": 0.9, "colsample_bytree": 0.9},
        {"seed": 999, "num_leaves": 63, "learning_rate": 0.01, "reg_alpha": 0.1, "reg_lambda": 0.1},
    )
    lgb5_max_rounds: int = 1000
    lgb5_early_stopping: int = 50

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
    # False early-stops on an inner 15% split of the fold's training rows instead.
    # Ablated in nested CV: False was clearly worse in every outer fold (the
    # models lose training data), so True stays the default.
    early_stop_on_val_fold: bool = True

    verbose: int = 500

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["models"] = list(self.models)
        d["feature_groups"] = list(self.feature_groups)
        d["exclude_features"] = list(self.exclude_features)
        d["lgb5_configs"] = [dict(c) for c in self.lgb5_configs]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = dict(d)
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        for k in ("models", "feature_groups", "exclude_features", "lgb5_configs"):
            if k in d:
                d[k] = tuple(d[k])
        return cls(**d)

    def replace(self, **kw) -> "Config":
        return dataclasses.replace(self, **kw)


def submission28_config() -> Config:
    """The configuration behind submission #28 (public LB 0.8089)."""
    return Config(feature_set="v1", n_base=80, n_interact=20, n_final=100, feature_groups=())


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
