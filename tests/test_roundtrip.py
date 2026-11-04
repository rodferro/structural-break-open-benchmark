"""Smoke test: train tiny models, then load *every* saved model in a fresh
interpreter and predict on a small sample."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sbreak.config import smoke_config
from sbreak.features import build_feature_table
from sbreak.pipeline import load_pipeline, train
from sbreak.synthetic import make_dataset

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".joblib", ".pkl", ".pickle"}


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    model_dir = tmp_path_factory.mktemp("resources")
    X, y = make_dataset(160, seed=1)
    p = train(X, y, str(model_dir), cfg=smoke_config())
    X_test, _ = make_dataset(30, seed=2)
    F_test = build_feature_table(X_test)
    return model_dir, p, X_test, p.predict(F_test)


def test_no_pickled_models(trained):
    model_dir, *_ = trained
    files = [f for f in model_dir.rglob("*") if f.is_file()]
    assert files
    assert not [f for f in files if f.suffix in FORBIDDEN_SUFFIXES]


def test_manifest_lists_existing_files(trained):
    model_dir, p, *_ = trained
    manifest = json.loads((model_dir / "manifest.json").read_text())
    for model in p.cfg.models:
        assert len(manifest["files"][model]) >= p.cfg.n_splits
        for f in manifest["files"][model]:
            assert (model_dir / f).exists(), f
    assert manifest["versions"]["tabpfn"] != "missing"


def test_every_model_loads_and_predicts(trained):
    model_dir, p, X_test, _ = trained
    loaded = load_pipeline(model_dir)
    F = loaded.model_matrix(build_feature_table(X_test))
    for name, models in loaded.ensemble.fold_models.items():
        assert len(models) == p.cfg.n_splits
        for m in models:
            pred = m.predict(F)
            assert pred.shape == (len(F),) and np.isfinite(pred).all(), name


def test_fresh_process_matches_in_memory(trained, tmp_path):
    model_dir, _, X_test, expected = trained
    xp = tmp_path / "X_test.parquet"
    out = tmp_path / "pred.npy"
    X_test.to_parquet(xp)
    code = (
        "import sys, numpy as np, pandas as pd; sys.path.insert(0, sys.argv[1]);"
        "import main; p = main.infer(pd.read_parquet(sys.argv[2]), sys.argv[3]);"
        "np.save(sys.argv[4], p.to_numpy())"
    )
    subprocess.run([sys.executable, "-c", code, str(ROOT), str(xp), str(model_dir), str(out)], check=True)
    np.testing.assert_allclose(np.load(out), expected, rtol=0, atol=1e-6)
