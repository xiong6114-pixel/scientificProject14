from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mobka_metrics import coverage, hv_cir_like
from mobka_stage1 import run_mobka_stage1
from mobka_stage2 import run_mobka_stage2_cross_mutation
from mobka_stage3 import run_mobka_stage3_levy_archive


def _save_metrics(path: Path, rows: list[list[float]]) -> None:
    arr = np.asarray(rows, dtype=np.float64)
    np.savetxt(path, arr, delimiter=",", fmt="%.17g", header="stage,nd_count,hv,cov_vs_prev,cov_prev_vs_stage", comments="")


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = Path(__file__).resolve().parent

    settings = {
        "data1": str(root / "demand_points_info_10.csv"),
        "data2": str(root / "charge_points_info_10.csv"),
        "maxgen": 15,
        "popnum": 30,
    }

    seed = 2

    rng1 = np.random.RandomState(seed)
    f1, t1 = run_mobka_stage1(settings, rng=rng1, return_trace=True)

    rng2 = np.random.RandomState(seed)
    f2, t2 = run_mobka_stage2_cross_mutation(settings, rng=rng2, return_trace=True)

    rng3 = np.random.RandomState(seed)
    f3, t3 = run_mobka_stage3_levy_archive(settings, rng=rng3, return_trace=True)

    nd1, nd2, nd3 = float(f1.shape[0]), float(f2.shape[0]), float(f3.shape[0])
    hv1, hv2, hv3 = float(hv_cir_like(f1)), float(hv_cir_like(f2)), float(hv_cir_like(f3))

    c21 = float(coverage(f2, f1))
    c12 = float(coverage(f1, f2))
    c32 = float(coverage(f3, f2))
    c23 = float(coverage(f2, f3))

    _save_metrics(
        out_dir / "py_mobka_stage_metrics.csv",
        [
            [1, nd1, hv1, np.nan, np.nan],
            [2, nd2, hv2, c21, c12],
            [3, nd3, hv3, c32, c23],
        ],
    )

    np.savetxt(out_dir / "py_mobka_stage1_front.csv", f1, delimiter=",", fmt="%.17g")
    np.savetxt(out_dir / "py_mobka_stage2_front.csv", f2, delimiter=",", fmt="%.17g")
    np.savetxt(out_dir / "py_mobka_stage3_front.csv", f3, delimiter=",", fmt="%.17g")

    np.savetxt(out_dir / "py_mobka_stage1_archive_size_history.csv", t1.archive_size_history, delimiter=",", fmt="%.17g")
    np.savetxt(out_dir / "py_mobka_stage2_archive_size_history.csv", t2.archive_size_history, delimiter=",", fmt="%.17g")
    np.savetxt(out_dir / "py_mobka_stage3_archive_size_history.csv", t3.archive_size_history, delimiter=",", fmt="%.17g")
    np.savetxt(out_dir / "py_mobka_stage3_ext_archive_size_history.csv", t3.ext_archive_size_history, delimiter=",", fmt="%.17g")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(["S1", "S2", "S3"], [hv1, hv2, hv3])
    axes[0].set_title("HV by Stage")

    g = np.arange(1, settings["maxgen"] + 1)
    axes[1].plot(g, t1.archive_size_history, label="S1 rep")
    axes[1].plot(g, t2.archive_size_history, label="S2 rep")
    axes[1].plot(g, t3.archive_size_history, label="S3 rep")
    axes[1].plot(g, t3.ext_archive_size_history, label="S3 ext")
    axes[1].set_title("Archive Size by Generation")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "py_mobka_stage_compare.png", dpi=180)

    print(f"Stage1 nd={nd1:.0f}, hv={hv1:.6f}")
    print(f"Stage2 nd={nd2:.0f}, hv={hv2:.6f}, C(S2,S1)={c21:.6f}, C(S1,S2)={c12:.6f}")
    print(f"Stage3 nd={nd3:.0f}, hv={hv3:.6f}, C(S3,S2)={c32:.6f}, C(S2,S3)={c23:.6f}")


if __name__ == "__main__":
    main()
