import numpy as np
import pandas as pd

from sbreak.features import add_interactions, build_feature_table, extract_features_for_one_id
from sbreak.synthetic import make_dataset


def test_feature_table_shape_and_determinism():
    X, _ = make_dataset(80, seed=3)
    F1 = build_feature_table(X, n_jobs=1)
    F2 = build_feature_table(X, n_jobs=2)
    assert F1.index.name == "id" and len(F1) == 80
    pd.testing.assert_frame_equal(F1, F2)


def test_single_series_features_are_finite_scalars():
    rng = np.random.default_rng(0)
    v = rng.normal(size=500)
    p = np.r_[np.zeros(300), np.ones(200)]
    feats = extract_features_for_one_id(v, p)
    assert len(feats) > 500
    assert all(np.isfinite(x) for x in feats.values())


def test_interactions_count():
    F = pd.DataFrame(np.random.rand(5, 6), columns=list("abcdef"))
    out = add_interactions(F, list(F.columns), top_k=4)
    assert out.shape[1] == 6 + 4 * (4 * 3 // 2)
