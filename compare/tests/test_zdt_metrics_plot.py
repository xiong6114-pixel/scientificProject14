from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

from zdt_metrics_plot import (
    calc_coverage,
    calc_hv_2d,
    plot_pareto_front,
    prepare_pareto_plot_data,
    select_front_points,
    zdt_cost,
    zdt_hv_reference,
    zdt_true_pf,
)


matplotlib.use("Agg")


def test_zdt_cost_manual_points() -> None:
    x = np.zeros(30, dtype=np.float64)

    f1 = zdt_cost(x, 1)
    f2 = zdt_cost(x, 2)
    f3 = zdt_cost(x, 3)

    x4 = np.zeros(10, dtype=np.float64)
    f4 = zdt_cost(x4, 4)

    # Manual check with x=0 => g=1 for all these settings
    assert np.max(np.abs(f1 - np.array([0.0, 1.0], dtype=np.float64))) <= 1e-12
    assert np.max(np.abs(f2 - np.array([0.0, 1.0], dtype=np.float64))) <= 1e-12
    assert np.max(np.abs(f3 - np.array([0.0, 1.0], dtype=np.float64))) <= 1e-12
    assert np.max(np.abs(f4 - np.array([0.0, 1.0], dtype=np.float64))) <= 1e-12


def test_zdt_true_pf_shapes_and_values() -> None:
    pf1 = zdt_true_pf(1, 5)
    pf4 = zdt_true_pf(4, 5)

    assert pf1.shape == (5, 2)
    assert pf4.shape == (5, 2)

    # ZDT4 true PF is same formula as ZDT1 in MATLAB file.
    assert np.max(np.abs(pf1 - pf4)) <= 1e-12


def test_hv_2d_manual_example() -> None:
    p = np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float64)
    ref = np.array([1.0, 1.0], dtype=np.float64)

    hv = calc_hv_2d(p, ref)

    # Manual area:
    # (1.0-0.4)*(1.0-0.6) + (0.4-0.2)*(1.0-0.8) = 0.24 + 0.04 = 0.28
    assert abs(float(hv) - 0.28) <= 1e-12


def test_coverage_direction_cab() -> None:
    a = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    b = np.array([[1.5, 1.5], [0.5, 2.5]], dtype=np.float64)

    c_ab = calc_coverage(a, b)
    c_ba = calc_coverage(b, a)

    # C(A,B)=0.5, C(B,A)=0.5 in this small case.
    assert abs(float(c_ab) - 0.5) <= 1e-12
    assert abs(float(c_ba) - 0.5) <= 1e-12


def test_select_front_points_manual() -> None:
    p = np.array(
        [
            [0.9, 0.1],
            [0.1, 0.9],
            [0.5, 0.5],
            [0.3, 0.7],
            [0.7, 0.3],
        ],
        dtype=np.float64,
    )

    sel = select_front_points(p, 3)

    # sorted by f1 then pick approx 1st/mid/last (MATLAB round(linspace(1,n,k)))
    expected = np.array([[0.1, 0.9], [0.5, 0.5], [0.9, 0.1]], dtype=np.float64)
    assert np.max(np.abs(sel - expected)) <= 1e-12


def test_plot_positions_consistency() -> None:
    true_pf = zdt_true_pf(1, 50)
    front_a = true_pf[::5, :]
    front_b = true_pf[2::5, :]

    t, a, b = prepare_pareto_plot_data(true_pf, front_a, front_b, n_plot_a=8, n_plot_b=8)

    out_dir = Path(__file__).resolve().parent / "_tmp_plot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "front.png"
    fig, ax = plot_pareto_front(t, a, b, title="demo", out_png=out_png)

    assert out_png.exists()

    # Ensure plotted positions equal input positions.
    line_true = ax.lines[0].get_xydata()
    line_a = ax.lines[1].get_xydata()
    line_b = ax.lines[2].get_xydata()

    assert np.max(np.abs(line_true - t)) <= 1e-12
    assert np.max(np.abs(line_a - a)) <= 1e-12
    assert np.max(np.abs(line_b - b)) <= 1e-12

    fig.clf()


def test_zdt_hv_reference_matches_matlab_script() -> None:
    assert np.max(np.abs(zdt_hv_reference(1) - np.array([1.1, 1.1], dtype=np.float64))) <= 0.0
    assert np.max(np.abs(zdt_hv_reference(2) - np.array([1.1, 1.1], dtype=np.float64))) <= 0.0
    assert np.max(np.abs(zdt_hv_reference(3) - np.array([1.1, 1.1], dtype=np.float64))) <= 0.0
    assert np.max(np.abs(zdt_hv_reference(4) - np.array([1.1, 150.0], dtype=np.float64))) <= 0.0
