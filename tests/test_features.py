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


def test_v2_v3_columns_do_not_depend_on_series_length():
    X_short, _ = make_dataset(20, seed=4, min_len=1200, max_len=1400)
    X_long, _ = make_dataset(20, seed=5, min_len=3000, max_len=3500)
    for fs in ("v2", "v3"):
        short = build_feature_table(X_short, n_jobs=1, feature_set=fs)
        long = build_feature_table(X_long, n_jobs=1, feature_set=fs)
        assert list(short.columns) == list(long.columns), fs
    v1_short = build_feature_table(X_short, n_jobs=1, feature_set="v1")
    v1_long = build_feature_table(X_long, n_jobs=1, feature_set="v1")
    assert set(v1_short.columns) != set(v1_long.columns)


def test_v2_has_no_impossible_or_duplicate_columns():
    X, _ = make_dataset(6, seed=8, min_len=1200, max_len=1400)
    F = build_feature_table(X, n_jobs=1, feature_set="v2")
    assert not [c for c in F.columns if c.startswith("z_") and ("3p0" in c or "4p0" in c) and c.endswith("_k10")]
    assert "z_outlier_count_3p0_local_diff_k10" not in F.columns
    for k in (20, 256):
        assert f"mc_mean_b{k}_a1x" not in F.columns and f"mc_med_b{k}_a1x" not in F.columns
        assert f"local_mean_diff_k{k}" in F.columns
    assert "z_outlier_count_2p5_a_tail_k10" in F.columns  # reachable threshold kept
    v1 = build_feature_table(X, n_jobs=1, feature_set="v1")
    assert "z_outlier_count_3p0_a_tail_k10" in v1.columns  # v1 stays as submitted
