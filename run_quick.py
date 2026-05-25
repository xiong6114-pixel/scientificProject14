"""Quick smoke run for the current 50-point typed seeded pipeline."""

from __future__ import annotations

import os


os.environ.setdefault("MAX_ITER", "3")
os.environ.setdefault("SEARCH_AGENTS_NO", "10")
os.environ.setdefault("INIT_NN_SEED_COUNT", "0")
os.environ.setdefault("SEED_DEVICE", "cpu")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

from run_ev_typed_mogabka_seeded import main


if __name__ == "__main__":
    main()
