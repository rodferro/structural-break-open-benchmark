import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_local_runner_synthetic(tmp_path):
    cmd = [
        sys.executable, "-m", "sbreak.runner", "--synthetic", "--smoke", "--determinism-check",
        "--model-dir", str(tmp_path / "resources"), "--prediction-dir", str(tmp_path / "prediction"),
    ]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    assert "determinism" in r.stdout
    assert (tmp_path / "prediction" / "prediction.parquet").exists()
