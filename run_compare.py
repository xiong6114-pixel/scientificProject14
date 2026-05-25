"""Run the current typed multi-algorithm comparison from the project root."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
COMPARE_DIR = PROJECT_DIR / "compare"

for path in (str(COMPARE_DIR), str(PROJECT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from runme_50_typed_compare import main


if __name__ == "__main__":
    main()
