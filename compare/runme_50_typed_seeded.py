from __future__ import annotations

import os
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Small quick-run wrapper that reuses the aligned 50-point typed-seeded pipeline.
os.environ.setdefault("MAX_ITER", "3")
os.environ.setdefault("SEARCH_AGENTS_NO", "10")
os.environ.setdefault("INIT_NN_SEED_COUNT", "2")
os.environ.setdefault("SEED_DEVICE", "cpu")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

from run_ev_typed_mogabka_seeded import main


if __name__ == "__main__":
    main()
