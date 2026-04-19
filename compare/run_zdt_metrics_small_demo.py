from __future__ import annotations

from pathlib import Path

import numpy as np

from zdt_metrics_plot import (
    calc_coverage,
    calc_hv_2d,
    plot_pareto_front,
    prepare_pareto_plot_data,
    zdt_cost,
    zdt_hv_reference,
    zdt_true_pf,
)


def main() -> None:
    out_dir = Path(__file__).resolve().parent

    # Small manually verifiable examples
    x = np.zeros(30, dtype=np.float64)
    f_zdt1 = zdt_cost(x, 1)

    p_hv = np.array([[0.2, 0.8], [0.4, 0.6]], dtype=np.float64)
    ref_hv = np.array([1.0, 1.0], dtype=np.float64)
    hv = calc_hv_2d(p_hv, ref_hv)

    a = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
    b = np.array([[1.5, 1.5], [0.5, 2.5]], dtype=np.float64)
    c_ab = calc_coverage(a, b)
    c_ba = calc_coverage(b, a)

    true_pf = zdt_true_pf(1, 200)
    front_a = true_pf[::8, :]
    front_b = true_pf[2::8, :]
    t, fa, fb = prepare_pareto_plot_data(true_pf, front_a, front_b, n_plot_a=20, n_plot_b=20)

    _, _ = plot_pareto_front(t, fa, fb, title="ZDT1 Pareto Front Demo", out_png=out_dir / "zdt1_pareto_demo.png")

    np.savetxt(out_dir / "zdt_small_demo_metrics.csv", np.array([[f_zdt1[0], f_zdt1[1], hv, c_ab, c_ba]], dtype=np.float64), delimiter=",", fmt="%.17g", header="zdt1_f1,zdt1_f2,hv_example,C(A,B),C(B,A)", comments="")

    refs = np.vstack([zdt_hv_reference(i) for i in [1, 2, 3, 4]])
    np.savetxt(out_dir / "zdt_hv_refs.csv", refs, delimiter=",", fmt="%.17g", header="ref_f1,ref_f2", comments="")

    print(f"ZDT1(x=0) = {f_zdt1}")
    print(f"HV example = {hv:.12f}")
    print(f"Coverage C(A,B) = {c_ab:.12f}, C(B,A) = {c_ba:.12f}")
    print(f"Saved demo files in: {out_dir}")


if __name__ == "__main__":
    main()
