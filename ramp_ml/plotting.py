from typing import Optional
import numpy as np


def plot_series_with_events(
    T: np.ndarray,
    pred_idx: np.ndarray,
    gt_idx: Optional[np.ndarray] = None,
    title: str = "",
    out_png: Optional[str] = None,
    max_points: Optional[int] = None,
    # NEW: font controls (you can override if needed)
    base_fontsize: int = 14,
    title_fontsize: int = 16,
):
    import matplotlib.pyplot as plt

    # Make all plot text bold + larger by default
    plt.rcParams.update({
        "font.size": base_fontsize,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.titlesize": title_fontsize,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize,
    })

    n = len(T)
    x = np.arange(n)

    if max_points is not None and n > max_points:
        step = int(np.ceil(n / max_points))
        x_plot = x[::step]
        T_plot = T[::step]
    else:
        x_plot = x
        T_plot = T

    fig, ax = plt.subplots(figsize=(18, 3))
    ax.plot(x_plot, T_plot, linewidth=1)

    ymin = float(np.min(T_plot))
    ymax = float(np.max(T_plot))

    # predicted: dashed black (keep your style)
    if pred_idx is not None and len(pred_idx) > 0:
        ax.vlines(pred_idx, ymin, ymax, linestyles="--", colors="k")

    # ground truth: dotted red (keep your style)
    if gt_idx is not None and len(gt_idx) > 0:
        ax.vlines(gt_idx, ymin, ymax, linestyles=":", colors="r")

    ax.set_xlabel("time index (1 Hz)", fontweight="bold")
    ax.set_ylabel("T", fontweight="bold")
    ax.set_title(title, fontweight="bold", fontsize=title_fontsize)

    # Ensure tick labels are bold too
    for tick in ax.get_xticklabels():
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontweight("bold")

    fig.tight_layout()

    if out_png:
        fig.savefig(out_png, dpi=200)
        print(f"[saved] {out_png}")
    else:
        plt.show()

    plt.close(fig)