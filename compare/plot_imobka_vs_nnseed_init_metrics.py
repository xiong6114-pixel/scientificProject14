from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT_PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_PROJECT_DIR))

from hv import HV
from mogabka_seeded import MOGABKA
from typed_compare_utils import (
    activated_ev_context,
    build_mogabka_seed_config,
    build_typed_compare_problem,
    merge_fronts_and_extract_reference,
)


OUT_DIR = Path(__file__).resolve().parent
POI_KMEANS_DIR = ROOT_PROJECT_DIR.parent / "poi kmeans"
DEFAULT_DEBUG_MAT_PATH = POI_KMEANS_DIR / "dataset_v3_50_debug.mat"


def _frontify(front: np.ndarray, obj_dim: int = 2) -> np.ndarray:
    arr = np.asarray(front, dtype=float)
    if arr.size == 0:
        return np.empty((0, obj_dim), dtype=float)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _safe_hv(front: np.ndarray, ref: np.ndarray) -> float:
    front = _frontify(front)
    ref = _frontify(ref)
    if front.size == 0 or ref.size == 0:
        return 0.0
    return float(HV(front, ref)[0])


def _feasible_mask(objs: np.ndarray, invalid_penalty: float) -> np.ndarray:
    objs = _frontify(objs)
    if objs.size == 0:
        return np.zeros(0, dtype=bool)
    return np.all(np.isfinite(objs), axis=1) & np.all(objs < float(invalid_penalty), axis=1)


def _collect_variant(
    problem,
    label: str,
    init_nn_seed_count: int,
    max_iter: int,
    popnum: int,
    base_seed: int,
    cross_rate: float,
    variate_rate: float,
    archive_size: int,
):
    seed_cfg = build_mogabka_seed_config(problem, init_nn_seed_count)
    problem_context = dict(problem.problem_context)
    problem_context.update(
        {
            "collect_history": True,
            "cross_rate": float(cross_rate),
            "variate_rate": float(variate_rate),
            "archive_size": int(archive_size),
        }
    )

    with activated_ev_context(problem):
        _, _, _, result = MOGABKA(
            Max_iter=max_iter,
            SearchAgents_no=popnum,
            FUN="EV_TYPED_CS",
            dim=problem.dim,
            numObj=2,
            lb=problem.lb_vec,
            ub=problem.ub_vec,
            seed=base_seed,
            seed_injection_config=seed_cfg,
            problem_context=problem_context,
        )

    init_fit = _frontify(result.get("initial_population_fitness", np.empty((0, 2), dtype=float)))
    init_pf = _frontify(result.get("initial_pf_fitness", np.empty((0, 2), dtype=float)))
    history = [init_pf]
    history.extend(_frontify(front) for front in result.get("archive_pf_fitness_history", []))

    feasible_ratio = float(np.mean(_feasible_mask(init_fit, problem.invalid_penalty))) if init_fit.shape[0] > 0 else 0.0
    return {
        "label": label,
        "init_nn_seed_count": int(init_nn_seed_count),
        "initial_population_size": int(init_fit.shape[0]),
        "initial_feasible_ratio": feasible_ratio,
        "initial_nd_count": int(init_pf.shape[0]),
        "history_fronts": history,
        "result": result,
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _annotate_bars(ax, bars, fmt: str) -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=10,
        )


def main() -> None:
    sample_idx = int(os.environ.get("SAMPLE_INDEX", "1286"))
    max_iter = int(os.environ.get("MAX_ITER", "50"))
    popnum = int(os.environ.get("SEARCH_AGENTS_NO", "50"))
    base_seed = int(os.environ.get("COMPARE_SEED", "42"))
    seed_device = os.environ.get("SEED_DEVICE", "cpu")
    early_gen_count = int(os.environ.get("EARLY_GEN_COUNT", "15"))
    cross_rate = float(os.environ.get("CROSS_RATE", "0.3"))
    variate_rate = float(os.environ.get("VARIATE_RATE", "0.8"))
    archive_size = int(os.environ.get("ARCHIVE_SIZE", "100"))

    os.environ["RUNTIME_DEBUG_MAT_PATH"] = str(Path(os.environ.get("RUNTIME_DEBUG_MAT_PATH", str(DEFAULT_DEBUG_MAT_PATH))))
    os.environ["RUNTIME_SAMPLE_INDEX"] = str(sample_idx)

    problem = build_typed_compare_problem(seed_device=seed_device)

    variants = [
        _collect_variant(
            problem=problem,
            label="IMOBKA (random init)",
            init_nn_seed_count=0,
            max_iter=max_iter,
            popnum=popnum,
            base_seed=base_seed,
            cross_rate=cross_rate,
            variate_rate=variate_rate,
            archive_size=archive_size,
        ),
        _collect_variant(
            problem=problem,
            label="NN-seed-IMOBKA",
            init_nn_seed_count=12,
            max_iter=max_iter,
            popnum=popnum,
            base_seed=base_seed,
            cross_rate=cross_rate,
            variate_rate=variate_rate,
            archive_size=archive_size,
        ),
    ]

    all_fronts: list[np.ndarray] = []
    for variant in variants:
        all_fronts.extend(variant["history_fronts"])
    ref_front = merge_fronts_and_extract_reference(all_fronts, invalid_penalty=problem.invalid_penalty)

    hv_rows: list[dict[str, float | int | str]] = []
    init_rows: list[dict[str, float | int | str]] = []
    for variant in variants:
        hv_curve = [_safe_hv(front, ref_front) for front in variant["history_fronts"]]
        variant["hv_curve"] = hv_curve
        for gen, hv in enumerate(hv_curve):
            hv_rows.append(
                {
                    "sample_idx": sample_idx,
                    "variant": variant["label"],
                    "generation": gen,
                    "hv": float(hv),
                }
            )
        init_rows.append(
            {
                "sample_idx": sample_idx,
                "variant": variant["label"],
                "init_nn_seed_count": variant["init_nn_seed_count"],
                "initial_population_size": variant["initial_population_size"],
                "initial_feasible_ratio": variant["initial_feasible_ratio"],
                "initial_nd_count": variant["initial_nd_count"],
            }
        )

    max_points = min(len(variant["hv_curve"]) for variant in variants)
    max_points = min(max_points, early_gen_count + 1)
    generations = np.arange(max_points)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    colors = {
        "IMOBKA (random init)": "#d95f02",
        "NN-seed-IMOBKA": "#1b9e77",
    }

    for variant in variants:
        axes[0].plot(
            generations,
            variant["hv_curve"][:max_points],
            marker="o",
            linewidth=2.2,
            markersize=4.5,
            color=colors[variant["label"]],
            label=variant["label"],
        )
    axes[0].set_title("Early-Generation Convergence")
    axes[0].set_xlabel("Generation (0 = initial)")
    axes[0].set_ylabel("HV")
    axes[0].grid(alpha=0.28, linestyle="--")
    axes[0].legend(frameon=False, fontsize=10)

    x = np.arange(len(variants))
    width = 0.55
    feasible_vals = [variant["initial_feasible_ratio"] for variant in variants]
    nd_vals = [variant["initial_nd_count"] for variant in variants]
    labels = [variant["label"] for variant in variants]

    bars1 = axes[1].bar(x, feasible_vals, width=width, color=[colors[label] for label in labels], alpha=0.9)
    axes[1].set_xticks(x, labels, rotation=10)
    axes[1].set_ylim(0.0, 1.08)
    axes[1].set_title("Initial Feasible Ratio")
    axes[1].set_ylabel("Ratio")
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")
    _annotate_bars(axes[1], bars1, "{:.1%}")

    bars2 = axes[2].bar(x, nd_vals, width=width, color=[colors[label] for label in labels], alpha=0.9)
    axes[2].set_xticks(x, labels, rotation=10)
    axes[2].set_title("Initial Non-dominated Count")
    axes[2].set_ylabel("Count")
    axes[2].grid(axis="y", alpha=0.25, linestyle="--")
    _annotate_bars(axes[2], bars2, "{:.0f}")

    mu = float(problem.parameter[1]) if problem.parameter.size >= 2 else float("nan")
    v = float(problem.parameter[2]) if problem.parameter.size >= 3 else float("nan")
    fig.suptitle(
        (
            f"sample_idx={sample_idx} | pop={popnum}, iter={max_iter}, seed={base_seed}, "
            f"cross={cross_rate}, mutate={variate_rate}, archive={archive_size}, "
            f"mu={mu:g}, V={v:g}"
        ),
        fontsize=11,
    )

    stem = f"sample_{sample_idx}_imobka_vs_nnseed_init_metrics"
    png_path = OUT_DIR / f"{stem}.png"
    hv_csv_path = OUT_DIR / f"{stem}_hv_curve.csv"
    init_csv_path = OUT_DIR / f"{stem}_initial_metrics.csv"
    md_path = OUT_DIR / f"{stem}.md"

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    _write_csv(hv_csv_path, hv_rows, ["sample_idx", "variant", "generation", "hv"])
    _write_csv(
        init_csv_path,
        init_rows,
        [
            "sample_idx",
            "variant",
            "init_nn_seed_count",
            "initial_population_size",
            "initial_feasible_ratio",
            "initial_nd_count",
        ],
    )

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# sample_idx={sample_idx} init comparison\n\n")
        f.write("This figure compares the same MOGABKA core under two initialization modes:\n")
        f.write("- `IMOBKA (random init)`: `init_nn_seed_count=0`\n")
        f.write("- `NN-seed-IMOBKA`: `init_nn_seed_count=12`\n\n")
        f.write("Settings used:\n")
        f.write(f"- `popnum={popnum}`, `max_iter={max_iter}`, `seed={base_seed}`\n")
        f.write(f"- `cross_rate={cross_rate}`, `variate_rate={variate_rate}`, `archive_size={archive_size}`\n")
        f.write(f"- `mu={mu:g}`, `V={v:g}`\n\n")
        f.write("Initial metrics:\n")
        for row in init_rows:
            f.write(
                f"- {row['variant']}: feasible_ratio={float(row['initial_feasible_ratio']):.2%}, "
                f"initial_nd_count={int(row['initial_nd_count'])}, "
                f"initial_population_size={int(row['initial_population_size'])}\n"
            )

    print(str(png_path))
    print(str(hv_csv_path))
    print(str(init_csv_path))
    print(str(md_path))


if __name__ == "__main__":
    main()
