from __future__ import annotations

from pathlib import Path
import shutil

import numpy as np

from runme_matlab_aligned import run_runme_matlab_aligned


def test_runme_matlab_aligned_smoke() -> None:
    fixture_dir = Path(__file__).resolve().parent / "fixtures"
    out_dir = Path(__file__).resolve().parent / "_tmp_runme"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    settings = {
        "data1": str(fixture_dir / "demand_points_info_10.csv"),
        "data2": str(fixture_dir / "charge_points_info_10.csv"),
        "maxgen": 2,
        "popnum": 8,
    }

    out = run_runme_matlab_aligned(
        settings=settings,
        seed=2,
        output_dir=out_dir,
        make_plots=False,
        show_plots=False,
    )

    for key in ["final_obj_MOBKAGA", "final_obj_imobka", "final_obj_nsga2", "final_obj_nsga3", "final_obj_moead"]:
        arr = np.asarray(out[key], dtype=np.float64)
        assert arr.ndim == 2
        assert arr.shape[1] == 2

    mobkaga_sol = np.asarray(out["final_sol_MOBKAGA"], dtype=np.float64)
    mobkaga_obj = np.asarray(out["final_obj_MOBKAGA"], dtype=np.float64)
    assert mobkaga_sol.ndim == 2
    assert mobkaga_sol.shape[0] == mobkaga_obj.shape[0]

    hv = np.asarray(out["hv"], dtype=np.float64)
    cov = np.asarray(out["coverage"], dtype=np.float64)
    assert hv.shape == (5,)
    assert cov.shape == (4,)
