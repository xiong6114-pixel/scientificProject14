from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent

REAL_HV_CSV = OUT_DIR / "real_compare_hv.csv"
REAL_COVERAGE_CSV = OUT_DIR / "real_compare_coverage.csv"
ZDT_METRICS_CSV = OUT_DIR / "zdt_public_metrics.csv"
ZDT_MEAN_METRICS_CSV = OUT_DIR / "zdt_public_metrics_mean.csv"


ALGO_COLORS: Dict[str, str] = {
    "IMOGABKA-seed": "#c0392b",
    "IMOGABKA": "#c0392b",
    "MOBKA": "#7f8c8d",
    "NSGA2": "#2e86de",
    "NSGA-II": "#2e86de",
    "NSGA3": "#27ae60",
    "NSGA-III": "#27ae60",
    "MOEAD": "#f39c12",
    "MOEA/D": "#f39c12",
}


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")


def load_real_hv(path: Path) -> tuple[List[str], np.ndarray, np.ndarray]:
    _require_file(path)
    labels: List[str] = []
    nd_count: List[int] = []
    hv_values: List[float] = []

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels.append(row["algorithm"])
            nd_count.append(int(float(row["nd_count"])))
            hv_values.append(float(row["hv"]))

    return labels, np.asarray(nd_count, dtype=np.int64), np.asarray(hv_values, dtype=np.float64)


def load_coverage(path: Path) -> tuple[List[str], np.ndarray]:
    _require_file(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    labels = rows[0][1:]
    matrix = []
    for row in rows[1:]:
        matrix.append([float(x) for x in row[1:]])
    return labels, np.asarray(matrix, dtype=np.float64)


def load_zdt_metrics(path: Path) -> List[dict]:
    _require_file(path)
    rows: List[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "problem": row["problem"],
                    "algorithm": row["algorithm"],
                    "nd_count": int(float(row["nd_count"])),
                    "gd": float(row["gd"]),
                    "igd": float(row["igd"]),
                    "spacing": float(row["spacing"]),
                }
            )
    return rows


def load_zdt_mean_metrics(path: Path) -> List[dict]:
    _require_file(path)
    rows: List[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "algorithm": row["algorithm"],
                    "mean_gd": float(row["mean_gd"]),
                    "mean_igd": float(row["mean_igd"]),
                    "mean_spacing": float(row["mean_spacing"]),
                }
            )
    return rows


def plot_real_hv(labels: List[str], nd_count: np.ndarray, hv_values: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(labels))
    colors = [ALGO_COLORS.get(label, "#34495e") for label in labels]
    bars = ax.bar(x, hv_values, color=colors, edgecolor="black", linewidth=0.8)

    ax.set_title("Real 50-Point Problem: HV Comparison")
    ax.set_ylabel("Normalized HV")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylim(0, max(1.0, float(np.max(hv_values)) * 1.12))
    ax.grid(axis="y", alpha=0.25)

    for bar, hv, nd in zip(bars, hv_values, nd_count):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"HV={hv:.3f}\nND={nd}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    out_path = OUT_DIR / "plot_real_hv_comparison.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_real_coverage(labels: List[str], matrix: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0)

    ax.set_title("Real 50-Point Problem: Coverage Heatmap")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_yticklabels(labels)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            color = "white" if val >= 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Coverage C(A,B)")
    fig.tight_layout()
    out_path = OUT_DIR / "plot_real_coverage_heatmap.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_zdt_mean_metrics(rows: List[dict]) -> Path:
    metric_specs = [
        ("mean_gd", "Mean GD"),
        ("mean_igd", "Mean IGD"),
        ("mean_spacing", "Mean Spacing"),
    ]
    labels = [row["algorithm"] for row in rows]
    y = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2))
    for ax, (metric_key, title) in zip(axes, metric_specs):
        values = np.asarray([row[metric_key] for row in rows], dtype=np.float64)
        colors = [ALGO_COLORS.get(label, "#34495e") for label in labels]
        ax.barh(y, values, color=colors, edgecolor="black", linewidth=0.7)
        ax.set_title(title)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xscale("log")
        ax.grid(axis="x", alpha=0.25)
        for idx, val in enumerate(values):
            ax.text(val * 1.05 if val > 0 else 1e-6, idx, f"{val:.3g}", va="center", fontsize=8)

    fig.suptitle("ZDT Public Benchmarks: Mean Metrics", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = OUT_DIR / "plot_zdt_mean_metrics.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_zdt_problem_heatmaps(rows: List[dict]) -> Path:
    problems = sorted({row["problem"] for row in rows}, key=lambda x: int(x.replace("ZDT", "")))
    algorithms = ["IMOGABKA", "MOBKA", "NSGA-II", "NSGA-III", "MOEA/D"]
    metric_specs = [("gd", "GD"), ("igd", "IGD"), ("spacing", "Spacing")]

    tables: Dict[str, np.ndarray] = {}
    for metric_key, _ in metric_specs:
        table = np.zeros((len(algorithms), len(problems)), dtype=np.float64)
        for i, algo in enumerate(algorithms):
            for j, prob in enumerate(problems):
                matched = next(row for row in rows if row["algorithm"] == algo and row["problem"] == prob)
                table[i, j] = float(matched[metric_key])
        tables[metric_key] = table

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    for ax, (metric_key, title) in zip(axes, metric_specs):
        raw = tables[metric_key]
        shown = np.log10(np.maximum(raw, 1e-6))
        im = ax.imshow(shown, cmap="viridis")
        ax.set_title(f"{title} by Problem")
        ax.set_xticks(np.arange(len(problems)))
        ax.set_yticks(np.arange(len(algorithms)))
        ax.set_xticklabels(problems)
        ax.set_yticklabels(algorithms)
        for i in range(raw.shape[0]):
            for j in range(raw.shape[1]):
                txt = f"{raw[i, j]:.2g}"
                ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("log10(value)")

    fig.suptitle("ZDT Public Benchmarks: Per-Problem Metrics", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = OUT_DIR / "plot_zdt_problem_metrics.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_summary_dashboard(
    real_labels: List[str],
    hv_values: np.ndarray,
    coverage_labels: List[str],
    coverage_matrix: np.ndarray,
    zdt_mean_rows: List[dict],
) -> Path:
    fig = plt.figure(figsize=(13.5, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1.05, 1])

    ax_hv = fig.add_subplot(gs[0, 0])
    x = np.arange(len(real_labels))
    ax_hv.bar(x, hv_values, color=[ALGO_COLORS.get(label, "#34495e") for label in real_labels], edgecolor="black")
    ax_hv.set_title("Real Problem HV")
    ax_hv.set_xticks(x)
    ax_hv.set_xticklabels(real_labels, rotation=20)
    ax_hv.set_ylim(0, max(1.0, float(np.max(hv_values)) * 1.12))
    ax_hv.grid(axis="y", alpha=0.25)

    ax_cov = fig.add_subplot(gs[0, 1])
    im = ax_cov.imshow(coverage_matrix, cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax_cov.set_title("Real Problem Coverage")
    ax_cov.set_xticks(np.arange(len(coverage_labels)))
    ax_cov.set_yticks(np.arange(len(coverage_labels)))
    ax_cov.set_xticklabels(coverage_labels, rotation=25, ha="right")
    ax_cov.set_yticklabels(coverage_labels)
    for i in range(coverage_matrix.shape[0]):
        for j in range(coverage_matrix.shape[1]):
            val = coverage_matrix[i, j]
            ax_cov.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val >= 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax_cov, fraction=0.046, pad=0.04)

    metric_specs = [("mean_gd", "Mean GD"), ("mean_igd", "Mean IGD"), ("mean_spacing", "Mean Spacing")]
    sub_gs = gs[1, :].subgridspec(1, 3)
    for idx, (metric_key, title) in enumerate(metric_specs):
        ax = fig.add_subplot(sub_gs[0, idx])
        labels = [row["algorithm"] for row in zdt_mean_rows]
        values = np.asarray([row[metric_key] for row in zdt_mean_rows], dtype=np.float64)
        y = np.arange(len(labels))
        ax.barh(y, values, color=[ALGO_COLORS.get(label, "#34495e") for label in labels], edgecolor="black", linewidth=0.7)
        ax.set_title(title)
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xscale("log")
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle("Experiment Comparison Dashboard", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = OUT_DIR / "plot_experiment_dashboard.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def main() -> None:
    plt.style.use("default")
    plt.rcParams["axes.unicode_minus"] = False

    real_labels, nd_count, hv_values = load_real_hv(REAL_HV_CSV)
    coverage_labels, coverage_matrix = load_coverage(REAL_COVERAGE_CSV)
    zdt_rows = load_zdt_metrics(ZDT_METRICS_CSV)
    zdt_mean_rows = load_zdt_mean_metrics(ZDT_MEAN_METRICS_CSV)

    saved = [
        plot_real_hv(real_labels, nd_count, hv_values),
        plot_real_coverage(coverage_labels, coverage_matrix),
        plot_zdt_mean_metrics(zdt_mean_rows),
        plot_zdt_problem_heatmaps(zdt_rows),
        plot_summary_dashboard(real_labels, hv_values, coverage_labels, coverage_matrix, zdt_mean_rows),
    ]

    print("Saved comparison plots:")
    for path in saved:
        print(path)


if __name__ == "__main__":
    main()
