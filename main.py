"""CrunchDAO entry point for the ADIA Lab Structural Break Open Benchmark.

The platform imports this file and calls ``train`` then ``infer``.
All logic lives in the ``sbreak`` package next to this file.
"""

import os

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import pandas as pd  # noqa: E402

from sbreak.config import Config, smoke_config  # noqa: E402
from sbreak.pipeline import infer as _infer  # noqa: E402
from sbreak.pipeline import train as _train  # noqa: E402


def _config() -> Config:
    cfg = smoke_config() if os.environ.get("SBREAK_CONFIG") == "smoke" else Config()
    # e.g. SBREAK_MODELS=lgb,cat for a CPU-only local end-to-end check
    if os.environ.get("SBREAK_MODELS"):
        cfg = cfg.replace(models=tuple(os.environ["SBREAK_MODELS"].split(",")))
    return cfg


def train(X_train: pd.DataFrame, y_train: pd.Series, model_directory_path: str) -> None:
    _train(X_train, y_train, model_directory_path, cfg=_config())


def infer(X_test: pd.DataFrame, model_directory_path: str) -> pd.Series:
    # The config used at train time is read back from the manifest.
    return _infer(X_test, model_directory_path)
