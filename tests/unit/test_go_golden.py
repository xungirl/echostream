"""The Go worker's golden files must be regenerated whenever the Python side changes."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_go_golden_files_are_up_to_date():
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_go_golden.py"), "--check"],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stdout + r.stderr
