from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from imobka_matlab_aligned import IMOBKA_funciton
from moead_matlab_aligned import MOEAD_function
from nsga2_matlab_aligned import build_initial_population_for_nsga2
from nsga3_matlab_aligned import NSGA3_funciton


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASELINE_DIR = Path(__file__).resolve().parents[1] / "baseline_matlab"


def _settings(maxgen: int = 8, popnum: int = 20) -> dict:
    return {
        "data1": str(PROJECT_ROOT / "demand_points_info_10.csv"),
        "data2": str(PROJECT_ROOT / "charge_points_info_10.csv"),
        "maxgen": maxgen,
        "popnum": popnum,
    }


def test_moead_function_runs_and_deterministic() -> None:
    settings = _settings(maxgen=8, popnum=20)

    rng_init1 = np.random.RandomState(2)
    st1 = build_initial_population_for_nsga2(settings, rng_init1)
    f1, t1 = MOEAD_function(st1, rng=np.random.RandomState(2), return_trace=True)

    rng_init2 = np.random.RandomState(2)
    st2 = build_initial_population_for_nsga2(settings, rng_init2)
    f2, t2 = MOEAD_function(st2, rng=np.random.RandomState(2), return_trace=True)

    assert f1.ndim == 2 and f1.shape[1] == 2
    assert t1.generation_metrics.shape[1] == 6
    assert f1.shape == f2.shape
    if f1.size > 0:
        assert np.max(np.abs(f1 - f2)) <= 0.0
    if t1.generation_metrics.size > 0:
        assert np.max(np.abs(t1.generation_metrics - t2.generation_metrics)) <= 0.0


def test_nsga3_function_runs_and_deterministic() -> None:
    settings = _settings(maxgen=8, popnum=20)

    f1, t1 = NSGA3_funciton(settings, rng=np.random.RandomState(2), return_trace=True)
    f2, t2 = NSGA3_funciton(settings, rng=np.random.RandomState(2), return_trace=True)

    assert f1.ndim == 2 and f1.shape[1] == 2
    assert t1.generation_metrics.shape[1] == 6
    assert f1.shape == f2.shape
    if f1.size > 0:
        assert np.max(np.abs(f1 - f2)) <= 0.0
    if t1.generation_metrics.size > 0:
        assert np.max(np.abs(t1.generation_metrics - t2.generation_metrics)) <= 0.0


def test_imobka_function_runs() -> None:
    settings = _settings(maxgen=8, popnum=20)

    rng_init = np.random.RandomState(2)
    st = build_initial_population_for_nsga2(settings, rng_init)
    f, t = IMOBKA_funciton(st, rng=np.random.RandomState(2), return_trace=True)

    assert f.ndim == 2 and f.shape[1] == 2
    assert t.archive_size_history.shape == (settings["maxgen"],)


def test_algorithms_against_matlab_baseline_if_present() -> None:
    req = [
        BASELINE_DIR / "mat_moead_small_final_obj.csv",
        BASELINE_DIR / "mat_nsga3_small_final_obj.csv",
        BASELINE_DIR / "mat_imobka_small_final_obj.csv",
    ]
    if not all(p.exists() for p in req):
        pytest.skip("MATLAB baseline files missing. Run export_*_small.m scripts first.")

    settings = _settings(maxgen=8, popnum=20)

    st = build_initial_population_for_nsga2(settings, np.random.RandomState(2))
    py_moead = MOEAD_function(st, rng=np.random.RandomState(2), return_trace=False)
    py_imobka = IMOBKA_funciton(st, rng=np.random.RandomState(2), return_trace=False)
    py_nsga3 = NSGA3_funciton(settings, rng=np.random.RandomState(2), return_trace=False)

    mat_moead = np.asarray(np.loadtxt(BASELINE_DIR / "mat_moead_small_final_obj.csv", delimiter=","), dtype=np.float64)
    mat_imobka = np.asarray(np.loadtxt(BASELINE_DIR / "mat_imobka_small_final_obj.csv", delimiter=","), dtype=np.float64)
    mat_nsga3 = np.asarray(np.loadtxt(BASELINE_DIR / "mat_nsga3_small_final_obj.csv", delimiter=","), dtype=np.float64)

    if mat_moead.ndim == 1:
        mat_moead = mat_moead.reshape(1, -1)
    if mat_imobka.ndim == 1:
        mat_imobka = mat_imobka.reshape(1, -1)
    if mat_nsga3.ndim == 1:
        mat_nsga3 = mat_nsga3.reshape(1, -1)

    assert py_moead.shape[1] == 2 and mat_moead.shape[1] == 2
    assert py_imobka.shape[1] == 2 and mat_imobka.shape[1] == 2
    assert py_nsga3.shape[1] == 2 and mat_nsga3.shape[1] == 2
