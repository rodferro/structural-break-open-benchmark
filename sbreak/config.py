"""Pipeline configuration. Defaults reproduce submission #28."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field


@dataclass
class Config:
    # cross-validation used to build the stacking OOF predictions
    n_splits: int = 5
    seed: int = 99999

    # two-stage LightGBM-importance feature selection
    n_base: int = 80  # top-k raw features kept after stage 1
    n_interact: int = 20  # top-k of those used to build pairwise interactions
    n_final: int = 100  # top-k features kept after stage 2
    select_max_rounds: int = 1_000_000

    # base models included in the stack
    models: tuple[str, ...] = ("lgb", "tabpfn", "cat")

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

    # TabPFN
    tabpfn_n_estimators: int = 8
    tabpfn_device: str = "auto"
    tabpfn_fit_mode: str = "fit_preprocessors"
    tabpfn_max_context: int | None = None  # subsample the training context (None = all rows)

    # stacker
    stacker: str = "logreg"  # "logreg" | "rank_mean"

    # the original stacked on early-stopped OOF where the early-stopping set
    # was the validation fold itself; set False to early-stop on an inner split
    early_stop_on_val_fold: bool = True

    verbose: int = 500

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["models"] = list(self.models)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = dict(d)
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        if "models" in d:
            d["models"] = tuple(d["models"])
        return cls(**d)

    def replace(self, **kw) -> "Config":
        return dataclasses.replace(self, **kw)


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
        tabpfn_n_estimators=1,
        tabpfn_device="cpu",
        verbose=0,
    )
