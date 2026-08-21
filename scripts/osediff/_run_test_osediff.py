"""Launch upstream OSEDiff inference with the shared torchvision shim installed."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OSEDIFF_ROOT = PROJECT_ROOT.parent / "baselines" / "repos" / "OSEDiff"
RESSHIFT_SCRIPTS = PROJECT_ROOT / "scripts" / "resshift"

sys.path.insert(0, str(RESSHIFT_SCRIPTS))
import _torchvision_compat  # noqa: F401,E402

sys.path.insert(0, str(OSEDIFF_ROOT))
os.chdir(OSEDIFF_ROOT)
runpy.run_path(str(OSEDIFF_ROOT / "test_osediff.py"), run_name="__main__")
