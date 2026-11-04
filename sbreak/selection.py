"""Two-stage LightGBM-importance feature selection.

Stage 1: rank all raw features by mean gain over K folds, keep ``n_base``.
Stage 2: add pairwise interactions among the top ``n_interact``, re-rank,
keep ``n_final``.

In submission #28 this was run once on the *full* training set and the
result (``F_100``) was then cross-validated, which leaks the labels of the
validation folds into the feature choice. Here selection is a function of the
training rows only, so it can be nested inside an outer CV loop.
"""

from __future__ import annotations

import re

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import Config
from .features import add_interactions


def lgb_importance(F: pd.DataFrame, y: pd.Series, cfg: Config) -> pd.DataFrame:
    """Mean LightGBM gain per feature over ``cfg.n_splits`` folds."""
    folds = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.seed)
    gains = np.zeros(F.shape[1])
    for trn_idx, val_idx in folds.split(F, y):
        trn = lgb.Dataset(F.iloc[trn_idx], label=y.iloc[trn_idx])
        val = lgb.Dataset(F.iloc[val_idx], label=y.iloc[val_idx])
        booster = lgb.train(
            cfg.lgb_params,
            trn,
            num_boost_round=cfg.select_max_rounds,
            valid_sets=[val],
            callbacks=[lgb.early_stopping(cfg.lgb_early_stopping, verbose=False)],
        )
        gains += booster.feature_importance(importance_type="gain")
    imp = pd.DataFrame({"feature": F.columns, "gain_mean": gains / cfg.n_splits})
    # stable sort so ties keep column order (matches pandas sort_values default for this case)
    return imp.sort_values("gain_mean", ascending=False, kind="stable").reset_index(drop=True)


def select_features(F_raw: pd.DataFrame, y: pd.Series, cfg: Config) -> tuple[list[str], list[str]]:
    """Return (base_features, final_features)."""
    if cfg.exclude_features:
        from .exclusions import resolve

        drop = resolve(cfg.exclude_features)
        F_raw = F_raw[[c for c in F_raw.columns if c not in drop]]
    imp1 = lgb_importance(F_raw, y, cfg)
    base = imp1["feature"].head(cfg.n_base).tolist()
    if cfg.n_interact < 2:
        # no interactions: a second importance pass would only re-rank the same
        # columns, so keep the stage-1 order (saves ~half the selection time)
        return base, base[: cfg.n_final]
    F_int = add_interactions(F_raw[base], top_features=base, top_k=cfg.n_interact)
    imp2 = lgb_importance(F_int, y, cfg)
    final = imp2["feature"].head(cfg.n_final).tolist()
    return base, final


_V1_MC = re.compile(r"^mc_(mean|std|med)_b\d+_a\d+$")
_V2_MC = re.compile(r"^mc_(mean|std|med)_b\d+_a[123]x$")


def _detect_feature_set(columns) -> str | None:
    """'v1' / 'v2+' from the multi-context column naming, None if undecidable."""
    if any(_V1_MC.match(c) for c in columns):
        return "v1"
    if any(_V2_MC.match(c) for c in columns):
        return "v2+"
    return None


def apply_selection(F_raw: pd.DataFrame, base: list[str], final: list[str], cfg: Config) -> pd.DataFrame:
    """Rebuild the model matrix for new rows from the stored feature lists.

    Columns that a batch of series does not produce at all (some feature names
    depend on segment lengths) are filled with NaN, which is what the training
    rows that lack them contained. Submission #28 filled them with 0.0, a
    silent train/infer skew for LightGBM.
    """
    detected = _detect_feature_set(F_raw.columns)
    expected = "v1" if cfg.feature_set == "v1" else "v2+"
    if detected is not None and detected != expected:
        raise ValueError(
            f"features look like {detected!r} but the model was trained on feature_set="
            f"{cfg.feature_set!r}; build features with feature_set={cfg.feature_set!r}"
        )
    F_base = F_raw.reindex(columns=base, fill_value=np.nan)
    F_int = add_interactions(F_base, top_features=base, top_k=cfg.n_interact)
    return F_int.reindex(columns=final, fill_value=np.nan)
