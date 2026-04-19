from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from zdt_metrics_plot import zdt_true_pf


OUT_DIR = Path(__file__).resolve().parent


def _load_front(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing front file: {path}")
    arr = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return np.asarray(arr, dtype=np.float64)


def _sort_front(front: np.ndarray) -> np.ndarray:
    if front.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return front[np.argsort(front[:, 0]), :]


def _plot_one(ax, prob_id: int) -> None:
    true_pf = zdt_true_pf(prob_id, 1000)
    mogabka_front = _sort_front(_load_front(OUT_DIR / f"zdt{prob_id}_front_imogabka.csv"))
    mobka_front = _sort_front(_load_front(OUT_DIR / f"zdt{prob_id}_front_mobka.csv"))

    ax.plot(true_pf[:, 0], true_pf[:, 1], color="black", linewidth=1.0, alpha=0.9, label="True PF")
    ax.scatter(
        mogabka_front[:, 0],
        mogabka_front[:, 1],
        s=18,
        c="#c0392b",
        marker="o",
        edgecolors="none",
        alpha=0.85,
        label="NN-seed MOGABKA",
    )
    ax.scatter(
        mobka_front[:, 0],
        mobka_front[:, 1],
        s=24,
        c="#2980b9",
        marker="^",
        edgecolors="none",
        alpha=0.85,
        label="MOBKA",
    )

    ax.set_title(f"ZDT{prob_id}")
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.grid(True, alpha=0.25)
    ax.text(
        0.03,
        0.97,
        f"MOGABKA: {mogabka_front.shape[0]} pts\nMOBKA: {mobka_front.shape[0]} pts",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    axes = axes.reshape(-1)

    for ax, prob_id in zip(axes, [1, 2, 3, 4]):
        _plot_one(ax, prob_id)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.suptitle("NN-seed MOGABKA vs MOBKA on ZDT1-ZDT4", fontsize=16, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_path = OUT_DIR / "plot_zdt_mogabka_vs_mobka_fronts.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
