from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from imobka_matlab_aligned import IMOBKA_funciton
from moead_matlab_aligned import MOEAD_function
from nsga2_matlab_aligned import build_initial_population_for_nsga2
from nsga3_matlab_aligned import NSGA3_funciton


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = Path(__file__).resolve().parent

    settings = {
        "data1": str(root / "demand_points_info_10.csv"),
        "data2": str(root / "charge_points_info_10.csv"),
        "maxgen": 8,
        "popnum": 20,
    }

    rng_init = np.random.RandomState(2)
    settings_init = build_initial_population_for_nsga2(settings, rng_init)

    moead_obj, moead_trace = MOEAD_function(settings_init, rng=np.random.RandomState(2), return_trace=True)
    imobka_obj, imobka_trace = IMOBKA_funciton(settings_init, rng=np.random.RandomState(2), return_trace=True)
    nsga3_obj, nsga3_trace = NSGA3_funciton(settings, rng=np.random.RandomState(2), return_trace=True)

    np.savetxt(out_dir / "py_moead_small_final_obj.csv", moead_obj, delimiter=",", fmt="%.17g")
    np.savetxt(out_dir / "py_imobka_small_final_obj.csv", imobka_obj, delimiter=",", fmt="%.17g")
    np.savetxt(out_dir / "py_nsga3_small_final_obj.csv", nsga3_obj, delimiter=",", fmt="%.17g")

    np.savetxt(out_dir / "py_moead_small_generation_metrics.csv", moead_trace.generation_metrics, delimiter=",", fmt="%.17g")
    np.savetxt(out_dir / "py_nsga3_small_generation_metrics.csv", nsga3_trace.generation_metrics, delimiter=",", fmt="%.17g")

    fig, ax = plt.subplots(figsize=(6, 5))
    if moead_obj.size > 0:
        ax.scatter(moead_obj[:, 0], moead_obj[:, 1], s=20, label="MOEAD")
    if imobka_obj.size > 0:
        ax.scatter(imobka_obj[:, 0], imobka_obj[:, 1], s=20, label="IMOBKA")
    if nsga3_obj.size > 0:
        ax.scatter(nsga3_obj[:, 0], nsga3_obj[:, 1], s=20, label="NSGA3")
    ax.set_xlabel("obj1")
    ax.set_ylabel("obj2")
    ax.legend(loc="best")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "py_moead_imobka_nsga3_small_fronts.png", dpi=180)

    print(f"MOEAD nd={moead_obj.shape[0]}")
    print(f"IMOBKA nd={imobka_obj.shape[0]}")
    print(f"NSGA3 nd={nsga3_obj.shape[0]}")


if __name__ == "__main__":
    main()
