"""Figures with explicit metrics and consistent display ranges."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

METHODS = ("clean", "degraded", "wiener", "tv_admm")
TITLES = ("Clean reference", "Blur + noise", "Wiener-form filter", "TV + ADMM")
COLORS = ("#596579", "#da7356", "#4b87bd", "#32a28a")


def image_comparison(images, metrics, path, predictions=None, truth=None, names=None, threshold=0.3, categories=()):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.7), constrained_layout=True)
    for axis, method, title in zip(axes, METHODS, TITLES):
        axis.imshow(np.clip(images[method], 0, 1))
        if predictions is not None:
            pred = predictions[method]
            for box, label, score in zip(pred["boxes"], pred["labels"], pred["scores"]):
                if score < threshold or int(label) not in categories:
                    continue
                x0, y0, x1, y1 = box
                axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#ff6348", linewidth=1.4))
                axis.text(x0 + 1, y0 + 3, f"{names[int(label)]} {score:.2f}", fontsize=7, color="white", va="top", clip_on=True, bbox={"facecolor":"#323a46", "alpha":0.85, "pad":1})
            for box in truth["boxes"]:
                x0, y0, x1, y1 = box
                axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#35e1cc", linestyle="--", linewidth=1.0))
        q = metrics[method]
        subtitle = "Reference image" if method == "clean" else f"PSNR {q['psnr']:.2f} dB | SSIM {q['ssim']:.3f}"
        axis.set_title(f"{title}\n{subtitle}", fontsize=11)
        axis.axis("off")
    if predictions is not None:
        fig.suptitle("Dashed teal: ground truth | Solid orange: predictions (selected categories only)", fontsize=12)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summary_plot(summary, path):
    levels = sorted({row["blur_length"] for row in summary})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for method, title, color in zip(METHODS[1:], TITLES[1:], COLORS[1:]):
        rows = sorted((r for r in summary if r["method"] == method), key=lambda r: r["blur_length"])
        axes[0].plot(levels, [r["psnr"] for r in rows], "o-", label=title, color=color)
    for method, title, color in zip(METHODS, TITLES, COLORS):
        rows = sorted((r for r in summary if r["method"] == method), key=lambda r: r["blur_length"])
        if rows and rows[0].get("recall") is not None:
            axes[1].plot(levels, [r["recall"] for r in rows], "o-", label=title, color=color)
    axes[0].set_ylabel("Mean PSNR (dB; higher is better)")
    axes[1].set_ylabel("Micro recall at fixed score / IoU threshold")
    axes[1].set_ylim(0, 1.05)
    for ax in axes:
        ax.set_xlabel("Motion length (pixels after resizing)")
        ax.set_xticks(levels)
        ax.grid(alpha=0.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def convergence_plot(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    iterations = [r["iteration"] for r in history]
    axes[0].plot(iterations, [r["objective"] for r in history], color=COLORS[-1])
    axes[0].set_ylabel("Objective (not required to be monotone)")
    for key in ("primal_rms", "dual_rms"):
        axes[1].semilogy(iterations, [max(1e-12, r[key]) for r in history], label=key)
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("ADMM iteration")
        ax.grid(alpha=0.2)
    fig.savefig(path, dpi=150)
    plt.close(fig)
