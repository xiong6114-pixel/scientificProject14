from __future__ import annotations

from pathlib import Path

import numpy as np

from hnsga2_kmeans_typed import (
    HNSGA2_KMeans_funciton,
    build_initial_population_for_hnsga2_kmeans,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _settings(maxgen: int = 6, popnum: int = 14) -> dict:
    return {
        "data1": str(PROJECT_ROOT / "demand_points_info_10.csv"),
        "data2": str(PROJECT_ROOT / "charge_points_info_10.csv"),
        "maxgen": maxgen,
        "popnum": popnum,
        "kmeans_seed_count": 4,
    }


def test_hnsga2_kmeans_runs_and_is_deterministic() -> None:
    settings = _settings()

    rng1 = np.random.RandomState(7)
    st1 = build_initial_population_for_hnsga2_kmeans(settings, rng1)
    f1, t1 = HNSGA2_KMeans_funciton(st1, rng=rng1, return_trace=True)

    rng2 = np.random.RandomState(7)
    st2 = build_initial_population_for_hnsga2_kmeans(settings, rng2)
    f2, t2 = HNSGA2_KMeans_funciton(st2, rng=rng2, return_trace=True)

    assert f1.ndim == 2 and f1.shape[1] == 2
    assert t1.generation_metrics.shape == (settings["maxgen"], 6)
    assert t1.cluster_count_history.shape == (settings["maxgen"],)
    assert f1.shape == f2.shape
    if f1.size > 0:
        assert np.max(np.abs(f1 - f2)) <= 0.0
    if t1.generation_metrics.size > 0:
        assert np.max(np.abs(t1.generation_metrics - t2.generation_metrics)) <= 0.0


def test_hnsga2_kmeans_output_shape_and_finite() -> None:
    settings = _settings(maxgen=4, popnum=12)

    rng = np.random.RandomState(3)
    st = build_initial_population_for_hnsga2_kmeans(settings, rng)
    final_obj, trace = HNSGA2_KMeans_funciton(st, rng=rng, return_trace=True)

    assert final_obj.ndim == 2
    assert final_obj.shape[1] == 2
    assert trace.generation_metrics.ndim == 2
    assert trace.generation_metrics.shape[1] == 6
    if final_obj.size > 0:
        assert np.all(np.isfinite(final_obj))
