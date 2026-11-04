"""Structural-break detection pipeline (ADIA Lab Structural Break Open Benchmark)."""

import os

# Set in the original submission before importing scipy; kept for parity.
os.environ.setdefault("SCIPY_ARRAY_API", "1")

# TabPFN refuses >1000 training rows on CPU unless this is set, and reads it at
# import time. It only lifts that guard; on GPU it has no effect.
os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "1")

# Import torch before lightgbm. On macOS both ship an OpenMP runtime; if
# LightGBM loads first, two libomp copies end up in the process and the first
# multi-threaded torch op (TabPFN model construction) deadlocks. Loading torch
# first makes LightGBM's @rpath/libomp.dylib bind to torch's copy.
try:  # pragma: no cover - torch is a hard dependency, but keep import-safe
    import torch  # noqa: F401
except ImportError:  # pragma: no cover
    pass
