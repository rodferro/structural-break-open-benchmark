import numpy as np
import pandas as pd
import pytest

from sbreak.config import smoke_config
from sbreak.exclusions import AUDIT, resolve
from sbreak.feature_groups import GROUPS, compute_groups
from sbreak.features import build_feature_table
from sbreak.pipeline import infer, train
from sbreak.synthetic import make_dataset


@pytest.mark.parametrize("group", sorted(GROUPS))
def test_group_is_finite_and_prefixed(group):
    X, _ = make_dataset(12, seed=6, min_len=1200, max_len=1500)
    G = build_feature_table(X, n_jobs=1, groups=(group,), base=False)
    assert len(G) == 12 and G.shape[1] > 0
    assert all(c.startswith(f"g_{group}__") for c in G.columns)
    assert np.isfinite(G.to_numpy()).all()


def test_groups_append_to_base_table():
    X, _ = make_dataset(8, seed=7, min_len=1200, max_len=1300)
    base = build_feature_table(X, n_jobs=1, feature_set="v2")
    both = build_feature_table(X, n_jobs=1, feature_set="v2", groups=("tests",))
    only = build_feature_table(X, n_jobs=1, groups=("tests",), base=False)
    pd.testing.assert_frame_equal(both[base.columns], base)
    pd.testing.assert_frame_equal(both[only.columns], only)


def test_unknown_group_is_rejected():
    with pytest.raises(KeyError, match="unknown feature group"):
        compute_groups(np.zeros(10), np.r_[np.zeros(5), np.ones(5)], ("nope",))


def test_exclusion_preset():
    assert resolve(["@audit"]) == set(AUDIT)
    assert resolve(["@audit", "foo"]) == set(AUDIT) | {"foo"}


def test_train_infer_with_group_and_exclusions(tmp_path):
    X, y = make_dataset(160, seed=1)
    X_test, _ = make_dataset(20, seed=2)
    cfg = smoke_config().replace(feature_groups=("tests",), exclude_features=("@audit",))
    p = train(X, y, str(tmp_path), cfg=cfg)
    assert not set(p.base_features) & set(AUDIT)
    pred = infer(X_test, str(tmp_path))
    assert len(pred) == 20 and np.isfinite(pred).all()
