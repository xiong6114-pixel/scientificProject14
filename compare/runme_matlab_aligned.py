from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from cal_obj_1_debug import cal_obj_1_with_debug
from get_parameter import get_parameter
from imobka_matlab_aligned import IMOBKA_funciton
from init_encoding import init_sol
from mobka_metrics import coverage, hv_cir_like
from mobkaga_matlab_aligned import MOBKAGA_function
from moead_matlab_aligned import MOEAD_function
from nsga2_matlab_aligned import NSGA2_funciton
from nsga3_matlab_aligned import NSGA3_funciton


def _eval_obj(sol: np.ndarray, demand_points_info: np.ndarray, charge_points_info: np.ndarray, parameter: np.ndarray) -> Tuple[np.float64, np.float64]:
    f, _ = cal_obj_1_with_debug(sol, demand_points_info, charge_points_info, parameter)
    obj1 = np.float64(f[0, 0])
    obj2 = np.float64(f[1, 0])
    if (not np.isfinite(obj1)) or (not np.isfinite(obj2)):
        return np.float64(-1.0), np.float64(-1.0)
    return obj1, obj2


def build_initial_population_runme(
    settings: Dict[str, Any],
    rng: np.random.RandomState,
) -> Dict[str, Any]:
    demand_points_info = np.asarray(np.loadtxt(settings["data1"], delimiter=","), dtype=np.float64)
    charge_points_info = np.asarray(np.loadtxt(settings["data2"], delimiter=","), dtype=np.float64)

    demand_points_num = int(demand_points_info.shape[0])
    charge_points_num = int(charge_points_info.shape[0])

    parameter = get_parameter()
    pop_num = int(settings["popnum"])

    obj_manager = np.empty((0, 2), dtype=np.float64)
    sol_manager: list[np.ndarray] = []

    for _ in range(pop_num):
        curr_sol = init_sol(charge_points_num, demand_points_num, rng)
        obj_1, obj_2 = _eval_obj(curr_sol, demand_points_info, charge_points_info, parameter)

        while obj_1 == -1.0 and obj_2 == -1.0:
            curr_sol = init_sol(charge_points_num, demand_points_num, rng)
            obj_1, obj_2 = _eval_obj(curr_sol, demand_points_info, charge_points_info, parameter)

        obj_manager = np.vstack([obj_manager, np.array([[obj_1, obj_2]], dtype=np.float64)])
        sol_manager.append(curr_sol)

    out = dict(settings)
    out["obj_manager"] = obj_manager
    out["sol_manager"] = sol_manager
    return out


def _sort_front(obj: np.ndarray) -> np.ndarray:
    if obj.size == 0:
        return obj
    return obj[np.argsort(obj[:, 0]), :]


def run_runme_matlab_aligned(
    settings: Dict[str, Any],
    seed: int | None = None,
    output_dir: str | Path | None = None,
    make_plots: bool = True,
    show_plots: bool = False,
) -> Dict[str, Any]:
    # MATLAB runme.m uses rng('shuffle') unless manually changed.
    rng = np.random.RandomState(seed)

    demand_points_info = np.asarray(np.loadtxt(settings["data1"], delimiter=","), dtype=np.float64)
    charge_points_info = np.asarray(np.loadtxt(settings["data2"], delimiter=","), dtype=np.float64)

    demand_points_num = int(demand_points_info.shape[0])
    _charge_points_num = int(charge_points_info.shape[0])
    _all_demand = int(np.round(np.sum(demand_points_info[:, 2])))
    _parameter = get_parameter()

    settings_with_pop = build_initial_population_runme(settings, rng)

    print("正在执行MOBKAGA算法")
    final_obj_mobkaga, final_sol_mobkaga = MOBKAGA_function(settings_with_pop, rng=rng)

    print("正在执行mobka算法")
    final_obj_imobka = IMOBKA_funciton(settings_with_pop, rng=rng)

    print("正在执行NSGA2算法")
    final_obj_nsga2 = NSGA2_funciton(settings_with_pop, rng=rng)

    print("正在执行NSGA3算法")
    final_obj_nsga3 = NSGA3_funciton(settings_with_pop, rng=rng)

    print("正在执行moead")
    final_obj_moead = MOEAD_function(settings_with_pop, rng=rng)

    if final_obj_mobkaga.size > 0:
        rows_to_remove = np.where(final_obj_mobkaga[:, 0] < 1.0)[0]
        if rows_to_remove.size > 0:
            keep = np.ones(final_obj_mobkaga.shape[0], dtype=bool)
            keep[rows_to_remove] = False
            final_obj_mobkaga = final_obj_mobkaga[keep, :]
            if final_sol_mobkaga.shape[0] == keep.size:
                final_sol_mobkaga = final_sol_mobkaga[keep, :]

    print("更新后的 final_obj_MOBKAGA:")
    print(final_obj_mobkaga)
    print("更新后的 final_sol_MOBKAGA:")
    print(final_sol_mobkaga)

    hv_mobkaga = hv_cir_like(final_obj_mobkaga)
    hv_imobka = hv_cir_like(final_obj_imobka)
    hv_nsga2 = hv_cir_like(final_obj_nsga2)
    hv_nsga3 = hv_cir_like(final_obj_nsga3)
    hv_moead = hv_cir_like(final_obj_moead)

    print(f"mobkaga算法的HV={float(hv_mobkaga):.6f}")
    print(f"mobka算法的HV={float(hv_imobka):.6f}")
    print(f"nsga2算法的HV={float(hv_nsga2):.6f}")
    print(f"nsga3算法的HV={float(hv_nsga3):.6f}")
    print(f"moead算法的HV={float(hv_moead):.6f}")

    coverage_score1 = coverage(final_obj_mobkaga, final_obj_nsga2)
    coverage_score2 = coverage(final_obj_mobkaga, final_obj_nsga3)
    coverage_score4 = coverage(final_obj_mobkaga, final_obj_moead)
    coverage_score6 = coverage(final_obj_mobkaga, final_obj_imobka)

    out: Dict[str, Any] = {
        "seed": seed,
        "demand_points_num": demand_points_num,
        "settings": settings_with_pop,
        "final_obj_MOBKAGA": final_obj_mobkaga,
        "final_sol_MOBKAGA": final_sol_mobkaga,
        "final_obj_imobka": final_obj_imobka,
        "final_obj_nsga2": final_obj_nsga2,
        "final_obj_nsga3": final_obj_nsga3,
        "final_obj_moead": final_obj_moead,
        "hv": np.array([hv_mobkaga, hv_imobka, hv_nsga2, hv_nsga3, hv_moead], dtype=np.float64),
        "coverage": np.array([coverage_score6, coverage_score1, coverage_score2, coverage_score4], dtype=np.float64),
    }

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(output_dir / "py_runme_front_mobkaga.csv", final_obj_mobkaga, delimiter=",", fmt="%.17g")
    np.savetxt(output_dir / "py_runme_front_imobka.csv", final_obj_imobka, delimiter=",", fmt="%.17g")
    np.savetxt(output_dir / "py_runme_front_nsga2.csv", final_obj_nsga2, delimiter=",", fmt="%.17g")
    np.savetxt(output_dir / "py_runme_front_nsga3.csv", final_obj_nsga3, delimiter=",", fmt="%.17g")
    np.savetxt(output_dir / "py_runme_front_moead.csv", final_obj_moead, delimiter=",", fmt="%.17g")
    np.savetxt(output_dir / "py_runme_hv.csv", out["hv"].reshape(1, -1), delimiter=",", fmt="%.17g")
    np.savetxt(output_dir / "py_runme_coverage.csv", out["coverage"].reshape(1, -1), delimiter=",", fmt="%.17g")

    if make_plots:
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        mobkaga_sorted = _sort_front(final_obj_mobkaga)
        imobka_sorted = _sort_front(final_obj_imobka)
        nsga2_sorted = _sort_front(final_obj_nsga2)
        nsga3_sorted = _sort_front(final_obj_nsga3)
        moead_sorted = _sort_front(final_obj_moead)

        if mobkaga_sorted.size > 0:
            ax1.plot(mobkaga_sorted[:, 0], mobkaga_sorted[:, 1], "-*", label="IMOBKA")
        if imobka_sorted.size > 0:
            ax1.plot(imobka_sorted[:, 0], imobka_sorted[:, 1], "-d", label="MOBKA")
        if nsga2_sorted.size > 0:
            ax1.plot(nsga2_sorted[:, 0], nsga2_sorted[:, 1], "-.", label="NSGAII")
        if nsga3_sorted.size > 0:
            ax1.plot(nsga3_sorted[:, 0], nsga3_sorted[:, 1], "-s", label="NSGAIII")
        if moead_sorted.size > 0:
            ax1.plot(moead_sorted[:, 0], moead_sorted[:, 1], "-x", label="MOEAD")

        ax1.set_xlabel("建设充电站总成本（元）")
        ax1.set_ylabel("用户总时间成本（小时）")
        ax1.legend()
        fig1.tight_layout()
        fig1.savefig(output_dir / "py_runme_front_compare.png", dpi=180)

        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.bar(["mobkaga", "mobka", "nsga2", "nsga3", "moead"], out["hv"])
        ax2.set_ylabel("HV")
        fig2.tight_layout()
        fig2.savefig(output_dir / "py_runme_hv_bar.png", dpi=180)

        fig3, ax3 = plt.subplots(figsize=(7, 5))
        ax3.bar(["mobkaga-mobka", "mobkaga-nsga2", "mobkaga-nsga3", "mobkaga-moead"], out["coverage"])
        ax3.set_ylabel("mobkaga覆盖率")
        fig3.tight_layout()
        fig3.savefig(output_dir / "py_runme_coverage_bar.png", dpi=180)

        if show_plots:
            plt.show()
        else:
            plt.close(fig1)
            plt.close(fig2)
            plt.close(fig3)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="MATLAB runme.m aligned Python runner")
    parser.add_argument("--data1", type=str, default=None, help="Path to demand_points_info.csv")
    parser.add_argument("--data2", type=str, default=None, help="Path to charge_points_info.csv")
    parser.add_argument("--maxgen", type=int, default=200)
    parser.add_argument("--popnum", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None, help="Default None = rng('shuffle')-like behavior")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    data1 = args.data1 if args.data1 is not None else str(root / "demand_points_info.csv")
    data2 = args.data2 if args.data2 is not None else str(root / "charge_points_info.csv")

    settings = {
        "maxgen": int(args.maxgen),
        "popnum": int(args.popnum),
        "data1": data1,
        "data2": data2,
    }

    run_runme_matlab_aligned(
        settings=settings,
        seed=args.seed,
        output_dir=args.output_dir,
        make_plots=not args.no_plot,
        show_plots=args.show,
    )


if __name__ == "__main__":
    main()

