from __future__ import annotations

import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


ROOT_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_PROJECT_DIR))

from gd import GD
from hv import HV
from igd import IGD
from mogabka_seeded import MOGABKA
from compare.mobka_common import (
    Particle,
    cgrid,
    delete_one_rep_member,
    deter_dom,
    determine_domination,
    dominates_particle,
    fgrid,
    rep_to_obj_matrix,
    select_leader,
)
from compare.zdt_metrics_plot import zdt_true_pf
from compare.run_zdt_public_metrics_compare import _problem_spec, _zdt_cost


OUT_DIR = Path(__file__).resolve().parent


def _safe_hv(front: np.ndarray, true_pf: np.ndarray) -> float:
    front = np.asarray(front, dtype=np.float64)
    true_pf = np.asarray(true_pf, dtype=np.float64)
    if front.size == 0 or true_pf.size == 0:
        return 0.0
    return float(HV(front, true_pf)[0])


def _safe_igd(front: np.ndarray, true_pf: np.ndarray) -> float:
    front = np.asarray(front, dtype=np.float64)
    true_pf = np.asarray(true_pf, dtype=np.float64)
    if front.size == 0 or true_pf.size == 0:
        return float("inf")
    return float(IGD(front, true_pf))


def _safe_gd(front: np.ndarray, true_pf: np.ndarray) -> float:
    front = np.asarray(front, dtype=np.float64)
    true_pf = np.asarray(true_pf, dtype=np.float64)
    if front.size == 0 or true_pf.size == 0:
        return float("inf")
    return float(GD(front, true_pf))


def _dedup_front(front: np.ndarray) -> np.ndarray:
    front = np.asarray(front, dtype=np.float64)
    if front.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if front.ndim == 1:
        front = front.reshape(1, -1)
    uniq = np.unique(front, axis=0)
    return uniq[np.argsort(uniq[:, 0]), :]


def _run_mogabka_trace(prob_id: int, popnum: int, max_iter: int, seed: int) -> dict[str, np.ndarray]:
    dim, lb, ub = _problem_spec(prob_id)
    _, _, true_pf, result = MOGABKA(
        Max_iter=max_iter,
        SearchAgents_no=popnum,
        FUN=f"ZDT{prob_id}",
        dim=dim,
        numObj=2,
        lb=lb,
        ub=ub,
        seed=seed,
    )

    front = np.asarray(result.get("archive_pf_fitness", true_pf), dtype=np.float64)
    return {
        "front": _dedup_front(front),
        "hv": np.asarray(result["HV"], dtype=np.float64),
        "igd": np.asarray(result["IGD"], dtype=np.float64),
        "gd": np.asarray(result["GD"], dtype=np.float64),
    }


def _run_mobka_trace(prob_id: int, popnum: int, max_iter: int, rng: np.random.RandomState) -> dict[str, np.ndarray]:
    dim, lb, ub = _problem_spec(prob_id)
    true_pf = zdt_true_pf(prob_id, 1000)

    init_pop = rng.rand(popnum, dim) * (ub - lb).reshape(1, -1) + lb.reshape(1, -1)
    pop = []
    for i in range(popnum):
        pos = np.asarray(init_pop[i, :], dtype=np.float64).copy()
        cost = _zdt_cost(pos, prob_id)
        pop.append(
            Particle(
                Position=pos.copy(),
                Cost=cost.copy(),
                BestPosition=pos.copy(),
                BestCost=cost.copy(),
            )
        )

    n_rep = 200
    n_grid = 6
    alpha = 0.1
    beta = 1.5
    gamma = 2
    p = 0.8

    pop = deter_dom(pop)
    rep = [pp for pp in pop if not pp.IsDominated]
    grid = cgrid(rep, n_grid, alpha)
    for i in range(len(rep)):
        rep[i] = fgrid(rep[i], grid)

    hv_hist = []
    igd_hist = []
    gd_hist = []

    for it in range(1, max_iter + 1):
        for i in range(popnum):
            leader = select_leader(rep, beta, rng)
            r = np.float64(rng.rand())
            n = np.float64(0.05 * np.exp(-2 * (it / max_iter) ** 2))

            if p < r:
                candidate = pop[i].Position + n * (1 + np.sin(r)) * pop[i].Position
            else:
                candidate = pop[i].Position * (n * (2 * rng.rand(dim) - 1) + 1)

            m = np.float64(2 * np.sin(r + np.pi / 2))
            s_idx = int(rng.randint(0, popnum))
            cauchy_value = np.tan((rng.rand(dim) - 0.5) * np.pi)

            if dominates_particle(pop[i], pop[s_idx]):
                candidate = candidate + cauchy_value * (pop[i].BestPosition - leader.Position)
            else:
                candidate = candidate + cauchy_value * (leader.Position - m * pop[i].BestPosition)

            candidate = np.clip(candidate, lb, ub)
            pop[i].Position = candidate
            pop[i].Cost = _zdt_cost(candidate, prob_id)

        pop = deter_dom(pop)
        rep = rep + [pp for pp in pop if not pp.IsDominated]
        rep = determine_domination(rep)
        rep = [pp for pp in rep if not pp.IsDominated]

        grid = cgrid(rep, n_grid, alpha)
        for i in range(len(rep)):
            rep[i] = fgrid(rep[i], grid)

        if len(rep) > n_rep:
            extra = len(rep) - n_rep
            for _ in range(extra):
                rep = delete_one_rep_member(rep, gamma, rng)

        rep_front = _dedup_front(rep_to_obj_matrix(rep))
        hv_hist.append(_safe_hv(rep_front, true_pf))
        igd_hist.append(_safe_igd(rep_front, true_pf))
        gd_hist.append(_safe_gd(rep_front, true_pf))

    final_front = _dedup_front(rep_to_obj_matrix(rep))
    return {
        "front": final_front,
        "hv": np.asarray(hv_hist, dtype=np.float64),
        "igd": np.asarray(igd_hist, dtype=np.float64),
        "gd": np.asarray(gd_hist, dtype=np.float64),
    }


def _plot_histories(ax, x: np.ndarray, a: np.ndarray, b: np.ndarray, title: str, ylabel: str, logy: bool = False) -> None:
    if logy:
        ax.semilogy(x, np.maximum(a, 1e-12), color="#c0392b", linewidth=2.0, label="MOGABKA")
        ax.semilogy(x, np.maximum(b, 1e-12), color="#2980b9", linewidth=2.0, label="MOBKA")
    else:
        ax.plot(x, a, color="#c0392b", linewidth=2.0, label="MOGABKA")
        ax.plot(x, b, color="#2980b9", linewidth=2.0, label="MOBKA")

    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)


def _save_history_csv(path: Path, hv: np.ndarray, igd: np.ndarray, gd: np.ndarray) -> None:
    iters = np.arange(1, hv.size + 1, dtype=np.int64)
    with path.open("w", encoding="utf-8") as f:
        f.write("iter,hv,igd,gd\n")
        for i in range(hv.size):
            f.write(f"{iters[i]},{float(hv[i]):.17g},{float(igd[i]):.17g},{float(gd[i]):.17g}\n")


def main() -> None:
    max_iter = int(os.environ.get("MAX_ITER", "30"))
    popnum = int(os.environ.get("SEARCH_AGENTS_NO", "50"))
    base_seed = int(os.environ.get("COMPARE_SEED", "42"))
    problems = [1, 2, 3, 4]

    histories: dict[int, dict[str, dict[str, np.ndarray]]] = {}

    for prob_id in problems:
        print(f"running convergence trace for ZDT{prob_id}...", flush=True)
        mogabka_trace = _run_mogabka_trace(prob_id, popnum, max_iter, base_seed + prob_id * 100)
        mobka_trace = _run_mobka_trace(prob_id, popnum, max_iter, np.random.RandomState(base_seed + prob_id * 100 + 1))
        histories[prob_id] = {"mogabka": mogabka_trace, "mobka": mobka_trace}

        _save_history_csv(OUT_DIR / f"zdt{prob_id}_convergence_mogabka.csv", mogabka_trace["hv"], mogabka_trace["igd"], mogabka_trace["gd"])
        _save_history_csv(OUT_DIR / f"zdt{prob_id}_convergence_mobka.csv", mobka_trace["hv"], mobka_trace["igd"], mobka_trace["gd"])

    x = np.arange(1, max_iter + 1, dtype=np.int64)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.5))
    for col, prob_id in enumerate(problems):
        _plot_histories(
            axes[0, col],
            x,
            histories[prob_id]["mogabka"]["igd"],
            histories[prob_id]["mobka"]["igd"],
            title=f"ZDT{prob_id} IGD Convergence",
            ylabel="IGD (lower is better)",
            logy=True,
        )
        _plot_histories(
            axes[1, col],
            x,
            histories[prob_id]["mogabka"]["hv"],
            histories[prob_id]["mobka"]["hv"],
            title=f"ZDT{prob_id} HV Convergence",
            ylabel="HV (higher is better)",
            logy=False,
        )

        axes[0, col].text(
            0.03,
            0.03,
            f"final IGD\nMOGABKA={histories[prob_id]['mogabka']['igd'][-1]:.3g}\nMOBKA={histories[prob_id]['mobka']['igd'][-1]:.3g}",
            transform=axes[0, col].transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
        )
        axes[1, col].text(
            0.03,
            0.03,
            f"final HV\nMOGABKA={histories[prob_id]['mogabka']['hv'][-1]:.3f}\nMOBKA={histories[prob_id]['mobka']['hv'][-1]:.3f}",
            transform=axes[1, col].transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
        )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("MOGABKA vs MOBKA Convergence on ZDT1-ZDT4", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_png = OUT_DIR / "plot_zdt_mogabka_vs_mobka_convergence.png"
    fig.savefig(out_png, dpi=220)
    plt.close(fig)

    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
