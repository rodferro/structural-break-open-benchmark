"""CrunchDAO-compatible ``train`` / ``infer`` built on the package modules."""

from __future__ import annotations

import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config
from .features import add_interactions, build_feature_table
from .models import check_versions, library_versions
from .selection import apply_selection, select_features
from .stacking import StackedEnsemble

MANIFEST = "manifest.json"
FORMAT_VERSION = 2


def as_series(y) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    return pd.Series(y).astype(int)


@dataclass
class FittedPipeline:
    cfg: Config
    base_features: list[str]
    final_features: list[str]
    ensemble: StackedEnsemble

    def model_matrix(self, F_raw: pd.DataFrame) -> pd.DataFrame:
        return apply_selection(F_raw, self.base_features, self.final_features, self.cfg)

    def predict(self, F_raw: pd.DataFrame) -> np.ndarray:
        return self.ensemble.predict(self.model_matrix(F_raw))

    def predict_base(self, F_raw: pd.DataFrame) -> dict[str, np.ndarray]:
        return self.ensemble.predict_base(self.model_matrix(F_raw))


def fit_pipeline(F_raw: pd.DataFrame, y: pd.Series, cfg: Config) -> FittedPipeline:
    """Feature selection + stacked ensemble, using only the rows given."""
    y = as_series(y).loc[F_raw.index]
    t = time.time()
    base, final = select_features(F_raw, y, cfg)
    if cfg.verbose:
        print(f"selected {len(final)} features in {time.time() - t:.0f}s", flush=True)
    F = add_interactions(F_raw[base], top_features=base, top_k=cfg.n_interact)[final]
    ens = StackedEnsemble(cfg).fit(F, y)
    return FittedPipeline(cfg, base, final, ens)


def save_pipeline(p: FittedPipeline, model_directory_path: str | Path) -> Path:
    d = Path(model_directory_path)
    d.mkdir(parents=True, exist_ok=True)
    files = p.ensemble.save(d)
    manifest = {
        "format_version": FORMAT_VERSION,
        "config": p.cfg.to_dict(),
        "base_features": p.base_features,
        "final_features": p.final_features,
        "files": files,
        "versions": library_versions(),
    }
    (d / MANIFEST).write_text(json.dumps(manifest, indent=2))
    return d


def load_pipeline(model_directory_path: str | Path) -> FittedPipeline:
    d = Path(model_directory_path)
    manifest = json.loads((d / MANIFEST).read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise RuntimeError(f"Unsupported model format {manifest.get('format_version')} in {d}")
    for msg in check_versions(manifest["versions"]):
        warnings.warn(f"library version drift: {msg}", RuntimeWarning, stacklevel=2)
    cfg = Config.from_dict(manifest["config"])
    ens = StackedEnsemble.load(cfg, d)
    return FittedPipeline(cfg, manifest["base_features"], manifest["final_features"], ens)


def train(
    X_train: pd.DataFrame,
    y_train,
    model_directory_path: str,
    cfg: Config | None = None,
    F_raw: pd.DataFrame | None = None,
) -> FittedPipeline:
    cfg = cfg or Config()
    if F_raw is None:
        F_raw = build_feature_table(X_train, feature_set=cfg.feature_set)
    p = fit_pipeline(F_raw, as_series(y_train), cfg)
    save_pipeline(p, model_directory_path)
    return p


def infer(X_test: pd.DataFrame, model_directory_path: str, F_raw: pd.DataFrame | None = None) -> pd.Series:
    p = load_pipeline(model_directory_path)
    if F_raw is None:
        F_raw = build_feature_table(X_test, feature_set=p.cfg.feature_set)
    pred = p.predict(F_raw)
    return pd.Series(pred, index=F_raw.index, name="prediction")
