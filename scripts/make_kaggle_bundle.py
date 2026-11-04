"""Build dist/kaggle_bundle.zip: code + feature-selection JSONs (no data, no scores).

Upload the zip as a private Kaggle dataset and run scripts/kaggle_tabpfn.py there.
"""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    dst = ROOT / "dist" / "kaggle_bundle"
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    for d in ("sbreak", "scripts"):
        shutil.copytree(ROOT / d, dst / d, ignore=ignore)
    for f in ("requirements.txt", "main.py"):
        shutil.copy2(ROOT / f, dst / f)
    n = 0
    nested = ROOT / "runs" / "nested"
    for sel in [*nested.glob("*/sel_*.json"), *nested.glob("*/data.json")]:
        target = dst / "runs" / "nested" / sel.parent.name / sel.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(sel, target)
        n += 1
    for bad in ("*.parquet", "*.npz", "*.joblib", "*.ckpt"):
        assert not list(dst.rglob(bad)), f"refusing to bundle {bad}"
    archive = shutil.make_archive(str(dst), "zip", root_dir=dst)
    print(f"bundled {n} selection files -> {archive}")


if __name__ == "__main__":
    main()
