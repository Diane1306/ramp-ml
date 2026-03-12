from typing import Optional
import numpy as np


def plot_series_with_events(
    T: np.ndarray,
    pred_idx: np.ndarray,
    gt_idx: Optional[np.ndarray] = None,
    title: str = "",
    out_png: Optional[str] = None,
    max_points: Optional[int] = None,
):
    import matplotlib.pyplot as plt

    n = len(T)
    x = np.arange(n)

    if max_points is not None and n > max_points:
        step = int(np.ceil(n / max_points))
        x_plot = x[::step]
        T_plot = T[::step]
    else:
        x_plot = x
        T_plot = T

    plt.figure(figsize=(18, 3))
    plt.plot(x_plot, T_plot, linewidth=1)

    ymin = float(np.min(T_plot))
    ymax = float(np.max(T_plot))

    # predicted: dashed black (keep your style)
    if pred_idx is not None and len(pred_idx) > 0:
        plt.vlines(pred_idx, ymin, ymax, linestyles="--", colors="k")

    # ground truth: dotted red (keep your style)
    if gt_idx is not None and len(gt_idx) > 0:
        plt.vlines(gt_idx, ymin, ymax, linestyles=":", colors="r")

    plt.xlabel("time index (1 Hz)")
    plt.ylabel("T")
    plt.title(title)
    plt.tight_layout()

    if out_png:
        plt.savefig(out_png, dpi=200)
        print(f"[saved] {out_png}")
    else:
        plt.show()
    plt.close()