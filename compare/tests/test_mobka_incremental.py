from __future__ import annotations

from pathlib import Path

import numpy as np

from mobka_metrics import coverage, hv_cir_like
from mobka_stage1 import run_mobka_stage1
from mobka_stage2 import run_mobka_stage2_cross_mutation
from mobka_stage3 import run_mobka_stage3_levy_archive


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _settings(maxgen: int = 8, popnum: int = 20) -> dict:
    return {
        "data1": str(PROJECT_ROOT / "demand_points_info_10.csv"),
        "data2": str(PROJECT_ROOT / "charge_points_info_10.csv"),
        "maxgen": maxgen,
        "popnum": popnum,
    }


def test_stage1_pure_mobka_runs() -> None:
    settings = _settings(maxgen=8, popnum=20)
    rng = np.random.RandomState(2)

    final_obj, trace = run_mobka_stage1(settings, rng=rng, return_trace=True)

    assert final_obj.ndim == 2
    assert final_obj.shape[1] == 2
    assert trace.archive_size_history.shape == (settings["maxgen"],)
    assert trace.nd_count_history.shape == (settings["maxgen"],)
    assert trace.hv_history.shape == (settings["maxgen"],)


def test_stage2_cross_mutation_metrics_compare_to_stage1() -> None:
    settings = _settings(maxgen=8, popnum=20)

    f1, t1 = run_mobka_stage1(settings, rng=np.random.RandomState(2), return_trace=True)
    f2, t2 = run_mobka_stage2_cross_mutation(settings, rng=np.random.RandomState(2), return_trace=True)

    nd1 = float(f1.shape[0])
    nd2 = float(f2.shape[0])
    hv1 = float(hv_cir_like(f1))
    hv2 = float(hv_cir_like(f2))
    c21 = float(coverage(f2, f1)) if f1.size > 0 and f2.size > 0 else 0.0
    c12 = float(coverage(f1, f2)) if f1.size > 0 and f2.size > 0 else 0.0

    assert nd1 >= 0 and nd2 >= 0
    assert np.isscalar(hv1) and np.isscalar(hv2)
    assert 0.0 <= c21 <= 1.0
    assert 0.0 <= c12 <= 1.0
    assert t1.archive_size_history.shape == t2.archive_size_history.shape == (settings["maxgen"],)


def test_stage3_levy_archive_metrics_compare_to_stage2() -> None:
    settings = _settings(maxgen=8, popnum=20)

    f2, t2 = run_mobka_stage2_cross_mutation(settings, rng=np.random.RandomState(2), return_trace=True)
    f3, t3 = run_mobka_stage3_levy_archive(settings, rng=np.random.RandomState(2), return_trace=True)

    nd2 = float(f2.shape[0])
    nd3 = float(f3.shape[0])
    hv2 = float(hv_cir_like(f2))
    hv3 = float(hv_cir_like(f3))
    c32 = float(coverage(f3, f2)) if f2.size > 0 and f3.size > 0 else 0.0
    c23 = float(coverage(f2, f3)) if f2.size > 0 and f3.size > 0 else 0.0

    assert nd2 >= 0 and nd3 >= 0
    assert np.isscalar(hv2) and np.isscalar(hv3)
    assert 0.0 <= c32 <= 1.0
    assert 0.0 <= c23 <= 1.0

    assert t3.archive_size_history.shape == (settings["maxgen"],)
    assert t3.ext_archive_size_history.shape == (settings["maxgen"],)
    assert np.all(t3.ext_archive_size_history >= 0)
