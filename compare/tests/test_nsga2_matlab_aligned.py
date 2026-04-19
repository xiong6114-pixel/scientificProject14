from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nsga2_matlab_aligned import NSGA2_funciton, build_initial_population_for_nsga2


BASELINE_DIR = Path(__file__).resolve().parents[1] / "baseline_matlab"
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def test_nsga2_deterministic_same_seed() -> None:
    settings = {
        "data1": str(PROJECT_ROOT / "demand_points_info_10.csv"),
        "data2": str(PROJECT_ROOT / "charge_points_info_10.csv"),
        "maxgen": 10,
        "popnum": 20,
    }

    rng1 = np.random.RandomState(2)
    st1 = build_initial_population_for_nsga2(settings=settings, rng=rng1)
    f1, t1 = NSGA2_funciton(st1, rng=rng1, return_trace=True)

    rng2 = np.random.RandomState(2)
    st2 = build_initial_population_for_nsga2(settings=settings, rng=rng2)
    f2, t2 = NSGA2_funciton(st2, rng=rng2, return_trace=True)

    assert f1.shape == f2.shape
    if f1.size > 0:
        assert np.max(np.abs(f1 - f2)) <= 0.0
    assert t1.generation_metrics.shape == t2.generation_metrics.shape
    if t1.generation_metrics.size > 0:
        assert np.max(np.abs(t1.generation_metrics - t2.generation_metrics)) <= 0.0


def test_nsga2_output_shape_and_finite() -> None:
    settings = {
        "data1": str(PROJECT_ROOT / "demand_points_info_10.csv"),
        "data2": str(PROJECT_ROOT / "charge_points_info_10.csv"),
        "maxgen": 10,
        "popnum": 20,
    }

    rng = np.random.RandomState(2)
    st = build_initial_population_for_nsga2(settings=settings, rng=rng)
    final_obj, trace = NSGA2_funciton(st, rng=rng, return_trace=True)

    assert final_obj.ndim == 2
    assert final_obj.shape[1] == 2
    assert trace.generation_metrics.ndim == 2
    assert trace.generation_metrics.shape[1] == 6

    if final_obj.size > 0:
        assert np.all(np.isfinite(final_obj))


def test_nsga2_against_matlab_baseline_if_present() -> None:
    req = [
        BASELINE_DIR / "nsga2_small_seed.csv",
        BASELINE_DIR / "nsga2_small_maxgen.csv",
        BASELINE_DIR / "nsga2_small_popnum.csv",
        BASELINE_DIR / "mat_nsga2_small_final_obj.csv",
        BASELINE_DIR / "mat_nsga2_small_generation_metrics.csv",
    ]
    if not all(p.exists() for p in req):
        pytest.skip("MATLAB NSGA-II baseline files missing. Run matlab/export_nsga2_baseline_small.m")

    seed = int(np.loadtxt(BASELINE_DIR / "nsga2_small_seed.csv", delimiter=","))
    maxgen = int(np.loadtxt(BASELINE_DIR / "nsga2_small_maxgen.csv", delimiter=","))
    popnum = int(np.loadtxt(BASELINE_DIR / "nsga2_small_popnum.csv", delimiter=","))

    settings = {
        "data1": str(PROJECT_ROOT / "demand_points_info_10.csv"),
        "data2": str(PROJECT_ROOT / "charge_points_info_10.csv"),
        "maxgen": maxgen,
        "popnum": popnum,
    }

    rng = np.random.RandomState(seed)
    st = build_initial_population_for_nsga2(settings=settings, rng=rng)
    py_final_obj, py_trace = NSGA2_funciton(st, rng=rng, return_trace=True)

    mat_final_obj = np.asarray(np.loadtxt(BASELINE_DIR / "mat_nsga2_small_final_obj.csv", delimiter=","), dtype=np.float64)
    if mat_final_obj.ndim == 1:
        mat_final_obj = mat_final_obj.reshape(1, -1)

    mat_metrics = np.asarray(np.loadtxt(BASELINE_DIR / "mat_nsga2_small_generation_metrics.csv", delimiter=","), dtype=np.float64)
    if mat_metrics.ndim == 1:
        mat_metrics = mat_metrics.reshape(1, -1)

    # Baseline check focuses on requested metrics.
    assert py_final_obj.shape[0] == mat_final_obj.shape[0]
    assert py_trace.generation_metrics.shape == mat_metrics.shape

    if py_trace.generation_metrics.size > 0:
        assert np.max(np.abs(py_trace.generation_metrics - mat_metrics)) <= 1e-8
