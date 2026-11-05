"""The nested-CV cache must not be reused for a different feature table."""

import json

import numpy as np
import pandas as pd
import pytest

from sbreak.config import smoke_config
from sbreak.evaluation import NestedCV


def _data(seed=0, n=60, p=5):
    rng = np.random.default_rng(seed)
    idx = pd.Index(range(n), name="id")
    F = pd.DataFrame(rng.normal(size=(n, p)), index=idx, columns=[f"f{i}" for i in range(p)])
    y = pd.Series(rng.integers(0, 2, size=n), index=idx)
    return F, y


def test_same_data_reuses_cache(tmp_path):
    F, y = _data()
    NestedCV(F, y, smoke_config(), run_dir=tmp_path, outer_splits=2)
    NestedCV(F.copy(), y.copy(), smoke_config(), run_dir=tmp_path, outer_splits=2)


def test_changed_feature_values_are_rejected(tmp_path):
    F, y = _data()
    NestedCV(F, y, smoke_config(), run_dir=tmp_path, outer_splits=2)
    F2 = F.copy()
    F2.iloc[0, 0] += 1.0  # same shape, same columns, different features
    with pytest.raises(RuntimeError, match="different data"):
        NestedCV(F2, y, smoke_config(), run_dir=tmp_path, outer_splits=2)


def test_renamed_columns_are_rejected(tmp_path):
    F, y = _data()
    NestedCV(F, y, smoke_config(), run_dir=tmp_path, outer_splits=2)
    with pytest.raises(RuntimeError, match="different data"):
        NestedCV(F.rename(columns={"f0": "g0"}), y, smoke_config(), run_dir=tmp_path, outer_splits=2)


def test_old_cache_without_feature_hash_is_adopted(tmp_path):
    F, y = _data()
    cv = NestedCV(F, y, smoke_config(), run_dir=tmp_path, outer_splits=2)
    path = cv.dir / "data.json"
    fp = json.loads(path.read_text())
    path.write_text(json.dumps({k: fp[k] for k in ("n_rows", "hash", "n_features")}))
    with pytest.warns(RuntimeWarning, match="predates feature hashing"):
        NestedCV(F, y, smoke_config(), run_dir=tmp_path, outer_splits=2)
    assert json.loads(path.read_text()) == fp


def test_truncated_cache_entry_is_recomputed(tmp_path):
    F, y = _data(n=80)
    cfg = smoke_config().replace(models=("lgb",), n_base=3, n_final=3)
    cv = NestedCV(F, y, cfg, run_dir=tmp_path, outer_splits=2)
    first = cv.run(verbose=False)
    pred = sorted(cv.dir.glob("pred_lgb_*_fold0.npz"))[0]
    pred.write_bytes(pred.read_bytes()[:20])  # simulate a run killed mid-write
    with pytest.warns(RuntimeWarning, match="unreadable cache entry"):
        again = NestedCV(F, y, cfg, run_dir=tmp_path, outer_splits=2).run(verbose=False)
    assert again["auc"]["stack"]["folds"] == first["auc"]["stack"]["folds"]
    assert not list(cv.dir.glob("*.tmp*"))
    cs = first["compute_seconds"]
    assert set(cs) == {"selection", "lgb", "total"} and min(cs.values()) >= 0
    assert abs(cs["total"] - cs["selection"] - cs["lgb"]) < 0.01
