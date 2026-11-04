"""ADR-002: code that serves screens must run without OpenSeesPy, whose licence covers only offline use."""

import subprocess
import sys


def test_screening_code_never_imports_openseespy() -> None:
    modules = "earthquake_screening.screening.refusal, earthquake_screening.screening.probes"
    check = f"import sys, {modules}; sys.exit('openseespy' in sys.modules)"
    # A fresh interpreter, so modules other tests imported (OpenSees among them) don't count.
    assert subprocess.run([sys.executable, "-c", check], check=False).returncode == 0  # noqa: S603 (fixed command)
