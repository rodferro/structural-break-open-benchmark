"""Synthetic data in the competition's format, for tests and CI (no real data)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_dataset(n_series: int = 120, seed: int = 0, min_len: int = 300, max_len: int = 600):
    """Return (X, y) shaped like the competition data.

    X: MultiIndex (id, time), columns ['value', 'period'].
    y: Series of bool indexed by id; True series get a mean/variance shift at the boundary.
    """
    rng = np.random.default_rng(seed)
    frames, labels = [], {}
    for i in range(n_series):
        n = int(rng.integers(min_len, max_len))
        split = int(n * rng.uniform(0.4, 0.8))
        brk = bool(rng.random() < 0.3)
        v = rng.standard_t(5, size=n) * 0.01
        if brk:
            v[split:] = v[split:] * rng.uniform(1.5, 3.0) + rng.normal(0, 0.01)
        period = np.r_[np.zeros(split, int), np.ones(n - split, int)]
        idx = pd.MultiIndex.from_arrays([np.full(n, i), np.arange(n)], names=["id", "time"])
        frames.append(pd.DataFrame({"value": v, "period": period}, index=idx))
        labels[i] = brk
    X = pd.concat(frames)
    y = pd.Series(labels, name="structural_breakpoint")
    y.index.name = "id"
    return X, y
