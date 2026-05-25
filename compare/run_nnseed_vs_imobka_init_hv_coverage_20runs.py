from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_PROJECT_DIR = SCRIPT_DIR.parents[0]
if str(ROOT_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_PROJECT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from coverage import Coverage
from plot_imobka_vs_nnseed_init_metrics import (
    DEFAULT_DEBUG_MAT_PATH,
    OUT_DIR,
    _collect_variant,
    _safe_hv,
    _write_csv,
)
from typed_compare_utils import build_typed_compare_problem, merge_fronts_and_extract_reference


def _safe_coverage(front_a: np.ndarray, front_b: np.ndarray) -> float:
    front_a = np.asarray(front_a, dtype=float)
    front_b = np.asarray(front_b, dtype=float)
    if front_a.size == 0 or front_b.size == 0:
        return 0.0
    return float(Coverage(front_b, front_a))


def _first_gen_reaching_threshold(hv_curve: list[float], threshold: float) -> int | None:
    for gen, hv in enumerate(hv_curve):
        if float(hv) >= float(threshold):
            return int(gen)
    return None


def _variant_by_label(variants: list[dict[str, Any]], label: str) -> dict[str, Any]:
    for variant in variants:
        if str(variant["label"]) == label:
            return variant
    raise KeyError(f"missing variant: {label}")


def _is_nnseed_better(row: dict[str, Any]) -> bool:
    return bool(
        float(row["final_hv_gain"]) > 0.0
        and float(row["initial_hv_gain"]) > 0.0
        and float(row["initial_coverage_gain"]) > 0.0
        and int(row["gen_gain_to_same_hv_threshold"]) > 0
    )


def _fmt_pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def main() -> None:
    sample_idx = int(os.environ.get("SAMPLE_INDEX", "1286"))
    max_iter = int(os.environ.get("MAX_ITER", "50"))
    popnum = int(os.environ.get("SEARCH_AGENTS_NO", "50"))
    seed_start = int(os.environ.get("COMPARE_SEED_START", "42"))
    num_runs = int(os.environ.get("NUM_RUNS", "20"))
    seed_device = os.environ.get("SEED_DEVICE", "cpu")
    cross_rate = float(os.environ.get("CROSS_RATE", "0.3"))
    variate_rate = float(os.environ.get("VARIATE_RATE", "0.8"))
    archive_size = int(os.environ.get("ARCHIVE_SIZE", "100"))

    os.environ["RUNTIME_DEBUG_MAT_PATH"] = str(
        Path(os.environ.get("RUNTIME_DEBUG_MAT_PATH", str(DEFAULT_DEBUG_MAT_PATH)))
    )
    os.environ["RUNTIME_SAMPLE_INDEX"] = str(sample_idx)

    problem = build_typed_compare_problem(seed_device=seed_device)

    run_rows: list[dict[str, Any]] = []
    hv_rows: list[dict[str, Any]] = []

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

        for variant in variants:
            hv_curve = [_safe_hv(front, ref_front) for front in variant["history_fronts"]]
            variant["hv_curve"] = hv_curve
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

        imobka_variant = _variant_by_label(variants, "IMOBKA (random init)")
        nnseed_variant = _variant_by_label(variants, "NN-seed-IMOBKA")

        init_front_imobka = np.asarray(imobka_variant["history_fronts"][0], dtype=float)
        init_front_nnseed = np.asarray(nnseed_variant["history_fronts"][0], dtype=float)

        init_hv_imobka = float(imobka_variant["hv_curve"][0])
        init_hv_nnseed = float(nnseed_variant["hv_curve"][0])
        final_hv_imobka = float(imobka_variant["hv_curve"][-1])
        final_hv_nnseed = float(nnseed_variant["hv_curve"][-1])

        init_cov_nnseed_to_imobka = _safe_coverage(init_front_nnseed, init_front_imobka)
        init_cov_imobka_to_nnseed = _safe_coverage(init_front_imobka, init_front_nnseed)

        same_hv_threshold = final_hv_imobka
        gen_imobka_to_threshold = _first_gen_reaching_threshold(imobka_variant["hv_curve"], same_hv_threshold)
        gen_nnseed_to_threshold = _first_gen_reaching_threshold(nnseed_variant["hv_curve"], same_hv_threshold)

        row = {
            "sample_idx": sample_idx,
            "seed": base_seed,
            "same_hv_threshold": same_hv_threshold,
            "initial_hv_imobka": init_hv_imobka,
            "initial_hv_nnseed": init_hv_nnseed,
            "initial_hv_gain": init_hv_nnseed - init_hv_imobka,
            "initial_coverage_nnseed_to_imobka": init_cov_nnseed_to_imobka,
            "initial_coverage_imobka_to_nnseed": init_cov_imobka_to_nnseed,
            "initial_coverage_gain": init_cov_nnseed_to_imobka - init_cov_imobka_to_nnseed,
            "initial_feasible_ratio_imobka": float(imobka_variant["initial_feasible_ratio"]),
            "initial_feasible_ratio_nnseed": float(nnseed_variant["initial_feasible_ratio"]),
            "initial_nd_count_imobka": int(imobka_variant["initial_nd_count"]),
            "initial_nd_count_nnseed": int(nnseed_variant["initial_nd_count"]),
            "final_hv_imobka": final_hv_imobka,
            "final_hv_nnseed": final_hv_nnseed,
            "final_hv_gain": final_hv_nnseed - final_hv_imobka,
            "gen_to_same_hv_threshold_imobka": "" if gen_imobka_to_threshold is None else int(gen_imobka_to_threshold),
            "gen_to_same_hv_threshold_nnseed": "" if gen_nnseed_to_threshold is None else int(gen_nnseed_to_threshold),
            "gen_gain_to_same_hv_threshold": (
                ""
                if gen_imobka_to_threshold is None or gen_nnseed_to_threshold is None
                else int(gen_imobka_to_threshold - gen_nnseed_to_threshold)
            ),
        }
        row["nnseed_better"] = "yes" if _is_nnseed_better(row) else "no"
        run_rows.append(row)

    better_rows = [row for row in run_rows if row["nnseed_better"] == "yes"]
    better_rows_sorted = sorted(
        better_rows,
        key=lambda row: (
            float(row["gen_gain_to_same_hv_threshold"]),
            float(row["initial_hv_gain"]),
            float(row["initial_coverage_gain"]),
            float(row["final_hv_gain"]),
        ),
        reverse=True,
    )

    stem = f"sample_{sample_idx}_nnseed_vs_imobka_init_hv_coverage_{num_runs}runs"
    all_csv_path = OUT_DIR / f"{stem}_all.csv"
    better_csv_path = OUT_DIR / f"{stem}_better_only.csv"
    hv_csv_path = OUT_DIR / f"{stem}_hv_curve_all_runs.csv"
    md_path = OUT_DIR / f"{stem}_summary.md"

    fieldnames = [
        "sample_idx",
        "seed",
        "same_hv_threshold",
        "initial_hv_imobka",
        "initial_hv_nnseed",
        "initial_hv_gain",
        "initial_coverage_nnseed_to_imobka",
        "initial_coverage_imobka_to_nnseed",
        "initial_coverage_gain",
        "initial_feasible_ratio_imobka",
        "initial_feasible_ratio_nnseed",
        "initial_nd_count_imobka",
        "initial_nd_count_nnseed",
        "final_hv_imobka",
        "final_hv_nnseed",
        "final_hv_gain",
        "gen_to_same_hv_threshold_imobka",
        "gen_to_same_hv_threshold_nnseed",
        "gen_gain_to_same_hv_threshold",
        "nnseed_better",
    ]
    _write_csv(all_csv_path, run_rows, fieldnames)
    _write_csv(better_csv_path, better_rows_sorted, fieldnames)
    _write_csv(hv_csv_path, hv_rows, ["sample_idx", "seed", "variant", "generation", "hv"])

    avg_initial_hv_gain = float(np.mean([float(row["initial_hv_gain"]) for row in better_rows_sorted])) if better_rows_sorted else 0.0
    avg_initial_cov_gain = float(np.mean([float(row["initial_coverage_gain"]) for row in better_rows_sorted])) if better_rows_sorted else 0.0
    avg_gen_gain = float(np.mean([float(row["gen_gain_to_same_hv_threshold"]) for row in better_rows_sorted])) if better_rows_sorted else 0.0
    avg_final_hv_gain = float(np.mean([float(row["final_hv_gain"]) for row in better_rows_sorted])) if better_rows_sorted else 0.0

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# sample_idx={sample_idx} nnseed vs imobka ({num_runs} runs)\n\n")
        f.write("## Metric definition\n\n")
        f.write("- `IMOBKA` means the same MOGABKA core with random initialization (`init_nn_seed_count=0`).\n")
        f.write("- `NN-seed-IMOBKA` means the same core with NN-seed initialization (`init_nn_seed_count=12`).\n")
        f.write("- `initial HV` and `initial Coverage` are computed on generation `0` fronts.\n")
        f.write("- `Coverage(A->B)` means the fraction of front `B` dominated by front `A`.\n")
        f.write("- `same_hv_threshold` is defined as the final HV reached by `IMOBKA` in that run.\n")
        f.write("- `nnseed_better = yes` requires four conditions at once:\n")
        f.write("  final_hv_gain > 0, initial_hv_gain > 0, initial_coverage_gain > 0, gen_gain_to_same_hv_threshold > 0.\n\n")

        f.write("## Run settings\n\n")
        f.write(f"- sample_idx: `{sample_idx}`\n")
        f.write(f"- num_runs: `{num_runs}`\n")
        f.write(f"- seed range: `{seed_start}` to `{seed_start + num_runs - 1}`\n")
        f.write(f"- popnum: `{popnum}`\n")
        f.write(f"- max_iter: `{max_iter}`\n")
        f.write(f"- cross_rate: `{cross_rate}`\n")
        f.write(f"- variate_rate: `{variate_rate}`\n")
        f.write(f"- archive_size: `{archive_size}`\n\n")

        f.write("## Summary\n\n")
        f.write(f"- nnseed better runs: `{len(better_rows_sorted)}/{num_runs}`\n")
        if better_rows_sorted:
            f.write(f"- average initial HV gain on better runs: `{avg_initial_hv_gain:.6f}`\n")
            f.write(f"- average initial Coverage gain on better runs: `{avg_initial_cov_gain:.6f}`\n")
            f.write(f"- average generation lead to same HV threshold: `{avg_gen_gain:.2f}`\n")
            f.write(f"- average final HV gain on better runs: `{avg_final_hv_gain:.6f}`\n\n")

            f.write("## Better runs only\n\n")
            f.write("| seed | init HV gain | init C(nnseed->imobka) | init C(imobka->nnseed) | gen(imobka) | gen(nnseed) | gen lead | final HV gain |\n")
            f.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
            for row in better_rows_sorted:
                f.write(
                    "| "
                    f"{int(row['seed'])} | "
                    f"{float(row['initial_hv_gain']):.6f} | "
                    f"{float(row['initial_coverage_nnseed_to_imobka']):.6f} | "
                    f"{float(row['initial_coverage_imobka_to_nnseed']):.6f} | "
                    f"{int(row['gen_to_same_hv_threshold_imobka'])} | "
                    f"{int(row['gen_to_same_hv_threshold_nnseed'])} | "
                    f"{int(row['gen_gain_to_same_hv_threshold'])} | "
                    f"{float(row['final_hv_gain']):.6f} |\n"
                )
        else:
            f.write("- no run satisfies the strict `nnseed_better` rule.\n")

    print(str(all_csv_path))
    print(str(better_csv_path))
    print(str(hv_csv_path))
    print(str(md_path))


if __name__ == "__main__":
    main()
