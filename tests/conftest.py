"""Make the repository root importable, so tests can compare against the
unchanged phlab-neurips25 code in ``data/`` and ``utils/``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
