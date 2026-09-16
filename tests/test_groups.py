"""pytest entry points: the S5 self-test (rep theory + pipeline against Chughtai's released seed 2, if present) and the affine self-tests."""
import subprocess, sys, pytest
from pathlib import Path
from cosetprobe.affine import Affine
from cosetprobe.models import RUNS_DIR


@pytest.mark.parametrize("n", [11, 13, 15, 16, 21])
def test_affine_selftest(n):
    assert Affine(n).selftest()


@pytest.mark.skipif(not (RUNS_DIR / "S5_MLP_seed2").exists(), reason="Chughtai's released models not cloned (see external/README.md)")
def test_s5_selftest():
    out = subprocess.run([sys.executable, str(Path(__file__).parent / "selftest_s5.py")], capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    assert "ALL PASS" in out.stdout, out.stdout[-2000:]
