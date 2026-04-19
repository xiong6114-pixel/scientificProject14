from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nsga2_matlab_aligned import NSGA2_funciton, build_initial_population_for_nsga2


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = Path(__file__).resolve().parent

    # Small-scale instance and fixed seed as required.
    settings = {
        "data1": str(root / "demand_points_info_10.csv"),
        "data2": str(root / "charge_points_info_10.csv"),
        "maxgen": 15,
        "popnum": 30,
    }
    seed = 2

    rng = np.random.RandomState(seed)
    settings = build_initial_population_for_nsga2(settings=settings, rng=rng)

    final_obj, trace = NSGA2_funciton(settings=settings, rng=rng, return_trace=True)

    np.savetxt(out_dir / "py_nsga2_small_final_obj.csv", final_obj, delimiter=",", fmt="%.17g")
    np.savetxt(out_dir / "py_nsga2_small_front_sorted.csv", trace.final_front_sorted, delimiter=",", fmt="%.17g")
    np.savetxt(
        out_dir / "py_nsga2_small_generation_metrics.csv",
        trace.generation_metrics,
        delimiter=",",
        fmt="%.17g",
        header="gen,best_obj1,best_obj2,mean_obj1,mean_obj2,nd_count",
        comments="",
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if final_obj.shape[0] > 0:
        axes[0].scatter(final_obj[:, 0], final_obj[:, 1], s=20)
    axes[0].set_xlabel("obj1")
    axes[0].set_ylabel("obj2")
    axes[0].set_title("NSGA-II Pareto Front (Python)")

    if trace.generation_metrics.shape[0] > 0:
        g = trace.generation_metrics[:, 0]
        axes[1].plot(g, trace.generation_metrics[:, 1], label="best_obj1")
        axes[1].plot(g, trace.generation_metrics[:, 3], label="mean_obj1")
        axes[1].plot(g, trace.generation_metrics[:, 2], label="best_obj2")
        axes[1].plot(g, trace.generation_metrics[:, 4], label="mean_obj2")
    axes[1].set_xlabel("generation")
    axes[1].set_title("Best/Mean Objective Trends")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "py_nsga2_small_experiment.png", dpi=180)

    nd_count = int(final_obj.shape[0])
    print(f"Python NSGA-II non-dominated count: {nd_count}")
    print("Saved:")
    print(f"  - {out_dir / 'py_nsga2_small_final_obj.csv'}")
    print(f"  - {out_dir / 'py_nsga2_small_front_sorted.csv'}")
    print(f"  - {out_dir / 'py_nsga2_small_generation_metrics.csv'}")
    print(f"  - {out_dir / 'py_nsga2_small_experiment.png'}")


if __name__ == "__main__":
    main()
