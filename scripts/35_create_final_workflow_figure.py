"""Create the final two-row QAFUE workflow figure.

Outputs:
    results/figures/final_workflow_figure.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PNG = PROJECT_ROOT / "results" / "figures" / "final_workflow_figure.png"


def add_box(
    ax: plt.Axes,
    center: tuple[float, float],
    width: float,
    height: float,
    title: str,
    subtitle: str,
    color: str,
) -> FancyBboxPatch:
    x = center[0] - width / 2
    y = center[1] - height / 2
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.4,
        edgecolor="#1F2937",
        facecolor=color,
    )
    ax.add_patch(patch)
    ax.text(
        center[0],
        center[1] + 0.023,
        title,
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#111827",
    )
    if subtitle:
        ax.text(
            center[0],
            center[1] - 0.038,
            subtitle,
            ha="center",
            va="center",
            fontsize=8.8,
            color="#374151",
            linespacing=1.15,
        )
    return patch


def add_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#374151",
    rad: float = 0.0,
    lw: float = 1.7,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=15,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=7,
            shrinkB=7,
        )
    )


def main() -> int:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.dpi": 300,
        }
    )

    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.94,
        "Quality-Aware Fetal Ultrasound Enhancement (QAFUE)",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color="#111827",
    )

    row1_y = 0.66
    row2_y = 0.27
    w = 0.125
    h = 0.145
    xs = [0.095, 0.255, 0.415, 0.575, 0.735, 0.895]

    row1 = [
        ("HR", "clean fetal\nultrasound target", "#EFF6FF"),
        ("Degradation", "blur, noise,\ncontrast loss", "#FEF3C7"),
        ("LQ", "low-quality\ninput frame", "#F3F4F6"),
        ("SwinIR-Light", "QAFUE Phase A\nseed 42", "#DCFCE7"),
        ("Restored", "enhanced\n256 x 256 image", "#E0F2FE"),
        ("U-Net Dice", "frozen HC18\nutility model", "#FCE7F3"),
    ]

    for x, (title, subtitle, color) in zip(xs, row1):
        add_box(ax, (x, row1_y), w, h, title, subtitle, color)
    for start_x, end_x in zip(xs[:-1], xs[1:]):
        add_arrow(ax, (start_x + w / 2, row1_y), (end_x - w / 2, row1_y))

    gate_w = 0.16
    row2 = [
        (0.21, "ACOUSLIC\nSweep Frames", "raw acquisition\nstream", "#F0FDFA"),
        (0.43, "Proxy Quality\nMLP Gate", "frame quality\nscore", "#EDE9FE"),
        (0.65, "Adaptive\nAcquisition", "accept, reacquire,\nslow sweep", "#FFE4E6"),
    ]
    for x, title, subtitle, color in row2:
        add_box(ax, (x, row2_y), gate_w, h, title, subtitle, color)
    add_arrow(ax, (0.21 + gate_w / 2, row2_y), (0.43 - gate_w / 2, row2_y), color="#0F766E")
    add_arrow(ax, (0.43 + gate_w / 2, row2_y), (0.65 - gate_w / 2, row2_y), color="#0F766E")
    add_arrow(ax, (0.65, row2_y + h / 2), (0.415, row1_y - h / 2), color="#B45309", rad=-0.25)
    add_arrow(ax, (0.65 + gate_w / 2, row2_y), (0.21 - gate_w / 2, row2_y), color="#B45309", rad=0.38)

    ax.text(
        0.5,
        0.48,
        "HC18 restoration and downstream utility",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#374151",
    )
    ax.text(
        0.5,
        0.085,
        "ACOUSLIC quality gate feeds acquisition feedback and conditions which frames enter enhancement.",
        ha="center",
        va="center",
        fontsize=10,
        color="#374151",
    )

    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"Wrote {OUTPUT_PNG.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
