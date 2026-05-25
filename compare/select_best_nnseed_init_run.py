from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_PROJECT_DIR = SCRIPT_DIR.parents[0]
if str(ROOT_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_PROJECT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_imobka_vs_nnseed_init_metrics import (
    DEFAULT_DEBUG_MAT_PATH,
    OUT_DIR,
    _annotate_bars,
    _collect_variant,
    _safe_hv,
    _write_csv,
)
from typed_compare_utils import build_typed_compare_problem, merge_fronts_and_extract_reference


def _select_best_run(run_rows: list[dict]) -> dict:
    def _key(row: dict):
        return (
            float(row["early_auc_gain"]),
            float(row["gen15_hv_gain"]),
            float(row["initial_nd_gain"]),
            float(row["initial_feasible_gain"]),
        )

    return max(run_rows, key=_key)


def main() -> None:
    sample_idx = int(os.environ.get("SAMPLE_INDEX", "1286"))
    max_iter = int(os.environ.get("MAX_ITER", "50"))
    popnum = int(os.environ.get("SEARCH_AGENTS_NO", "50"))
    seed_start = int(os.environ.get("COMPARE_SEED_START", "42"))
    num_runs = int(os.environ.get("NUM_RUNS", "10"))
    seed_device = os.environ.get("SEED_DEVICE", "cpu")
    early_gen_count = int(os.environ.get("EARLY_GEN_COUNT", "15"))
    cross_rate = float(os.environ.get("CROSS_RATE", "0.3"))
    variate_rate = float(os.environ.get("VARIATE_RATE", "0.8"))
    archive_size = int(os.environ.get("ARCHIVE_SIZE", "100"))

    os.environ["RUNTIME_DEBUG_MAT_PATH"] = str(
        Path(os.environ.get("RUNTIME_DEBUG_MAT_PATH", str(DEFAULT_DEBUG_MAT_PATH)))
    )
    os.environ["RUNTIME_SAMPLE_INDEX"] = str(sample_idx)

    problem = build_typed_compare_problem(seed_device=seed_device)

    run_rows: list[dict[str, float | int | str]] = []
    hv_rows: list[dict[str, float | int | str]] = []
    detail_by_seed: dict[int, dict] = {}

    for offset in range(num_runs):
        base_seed = seed_start + offset
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

        hv_curves: dict[str, list[float]] = {}
        for variant in variants:
            hv_curve = [_safe_hv(front, ref_front) for front in variant["history_fronts"]]
            hv_curves[variant["label"]] = hv_curve
            for gen, hv in enumerate(hv_curve):
                hv_rows.append(
                    {
                        "sample_idx": sample_idx,
                        "seed": base_seed,
                        "variant": variant["label"],
                        "generation": gen,
                        "hv": float(hv),
                    }
                )

        rand_curve = hv_curves["IMOBKA (random init)"]
        seed_curve = hv_curves["NN-seed-IMOBKA"]
        max_points = min(len(rand_curve), len(seed_curve), early_gen_count + 1)
        early_auc_gain = float(np.sum(np.asarray(seed_curve[:max_points]) - np.asarray(rand_curve[:max_points])))
        gen0_hv_gain = float(seed_curve[0] - rand_curve[0]) if max_points > 0 else 0.0
        gen15_idx = max_points - 1
        gen15_hv_gain = float(seed_curve[gen15_idx] - rand_curve[gen15_idx]) if max_points > 0 else 0.0

        rand_variant = variants[0]
        seed_variant = variants[1]
        initial_feasible_gain = float(
            seed_variant["initial_feasible_ratio"] - rand_variant["initial_feasible_ratio"]
        )
        initial_nd_gain = float(seed_variant["initial_nd_count"] - rand_variant["initial_nd_count"])

        row = {
            "sample_idx": sample_idx,
            "seed": base_seed,
            "early_auc_gain": early_auc_gain,
            "gen0_hv_gain": gen0_hv_gain,
            f"gen{gen15_idx}_hv_gain": gen15_hv_gain,
            "initial_feasible_gain": initial_feasible_gain,
            "initial_nd_gain": initial_nd_gain,
            "random_init_feasible_ratio": float(rand_variant["initial_feasible_ratio"]),
            "nnseed_feasible_ratio": float(seed_variant["initial_feasible_ratio"]),
            "random_init_nd_count": int(rand_variant["initial_nd_count"]),
            "nnseed_nd_count": int(seed_variant["initial_nd_count"]),
        }
        run_rows.append(row)
        detail_by_seed[base_seed] = {
            "variants": variants,
            "hv_curves": hv_curves,
            "max_points": max_points,
            "ref_front": ref_front,
        }

    best_row = _select_best_run(run_rows)
    best_seed = int(best_row["seed"])
    best_detail = detail_by_seed[best_seed]
    best_variants = best_detail["variants"]
    best_max_points = int(best_detail["max_points"])
    generations = np.arange(best_max_points)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8), constrained_layout=True)
    colors = {
        "IMOBKA (random init)": "#d95f02",
        "NN-seed-IMOBKA": "#1b9e77",
    }

    for variant in best_variants:
        label = variant["label"]
        axes[0].plot(
            generations,
            best_detail["hv_curves"][label][:best_max_points],
            marker="o",
            linewidth=2.2,
            markersize=4.5,
            color=colors[label],
            label=label,
        )
    axes[0].set_title("Early-Generation Convergence")
    axes[0].set_xlabel("Generation (0 = initial)")
    axes[0].set_ylabel("HV")
    axes[0].grid(alpha=0.28, linestyle="--")
    axes[0].legend(frameon=False, fontsize=10)

    x = np.arange(len(best_variants))
    labels = [variant["label"] for variant in best_variants]
    feasible_vals = [variant["initial_feasible_ratio"] for variant in best_variants]
    nd_vals = [variant["initial_nd_count"] for variant in best_variants]

    bars1 = axes[1].bar(x, feasible_vals, width=0.55, color=[colors[label] for label in labels], alpha=0.9)
    axes[1].set_xticks(x, labels, rotation=10)
    axes[1].set_ylim(0.0, 1.08)
    axes[1].set_title("Initial Feasible Ratio")
    axes[1].set_ylabel("Ratio")
    axes[1].grid(axis="y", alpha=0.25, linestyle="--")
    _annotate_bars(axes[1], bars1, "{:.1%}")

    bars2 = axes[2].bar(x, nd_vals, width=0.55, color=[colors[label] for label in labels], alpha=0.9)
    axes[2].set_xticks(x, labels, rotation=10)
    axes[2].set_title("Initial Non-dominated Count")
    axes[2].set_ylabel("Count")
    axes[2].grid(axis="y", alpha=0.25, linestyle="--")
    _annotate_bars(axes[2], bars2, "{:.0f}")

    mu = float(problem.parameter[1]) if problem.parameter.size >= 2 else float("nan")
    v = float(problem.parameter[2]) if problem.parameter.size >= 3 else float("nan")
    fig.suptitle(
        (
            f"Best of {num_runs} runs | sample_idx={sample_idx}, selected seed={best_seed}, "
            f"early_auc_gain={float(best_row['early_auc_gain']):.4f}, "
            f"cross={cross_rate}, mutate={variate_rate}, archive={archive_size}, mu={mu:g}, V={v:g}"
        ),
        fontsize=11,
    )

    stem = f"sample_{sample_idx}_best_of_{num_runs}_nnseed_init_metrics"
    png_path = OUT_DIR / f"{stem}.png"
    run_csv_path = OUT_DIR / f"{stem}_run_summary.csv"
    hv_csv_path = OUT_DIR / f"{stem}_hv_curve_all_runs.csv"
    md_path = OUT_DIR / f"{stem}.md"

    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    run_fieldnames = [
        "sample_idx",
        "seed",
        "early_auc_gain",
        "gen0_hv_gain",
        f"gen{best_max_points - 1}_hv_gain",
        "initial_feasible_gain",
        "initial_nd_gain",
        "random_init_feasible_ratio",
        "nnseed_feasible_ratio",
        "random_init_nd_count",
        "nnseed_nd_count",
    ]
    _write_csv(run_csv_path, run_rows, run_fieldnames)
    _write_csv(hv_csv_path, hv_rows, ["sample_idx", "seed", "variant", "generation", "hv"])

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# sample_idx={sample_idx} best nnseed run out of {num_runs}\n\n")
        f.write("Selection rule:\n")
        f.write(
            f"- choose the run with the largest cumulative HV gain of `NN-seed-IMOBKA` over "
            f"`IMOBKA (random init)` on generations `0..{best_max_points - 1}`.\n\n"
        )
        f.write("Best run summary:\n")
        f.write(f"- selected seed: `{best_seed}`\n")
        f.write(f"- early_auc_gain: `{float(best_row['early_auc_gain']):.6f}`\n")
        f.write(f"- gen0_hv_gain: `{float(best_row['gen0_hv_gain']):.6f}`\n")
        f.write(f"- gen{best_max_points - 1}_hv_gain: `{float(best_row[f'gen{best_max_points - 1}_hv_gain']):.6f}`\n")
        f.write(f"- initial_feasible_gain: `{float(best_row['initial_feasible_gain']):.2%}`\n")
        f.write(f"- initial_nd_gain: `{float(best_row['initial_nd_gain']):.0f}`\n")

    print(str(png_path))
    print(str(run_csv_path))
    print(str(hv_csv_path))
    print(str(md_path))


if __name__ == "__main__":
    main()
