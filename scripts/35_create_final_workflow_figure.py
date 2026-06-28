"""
Create a paper-ready workflow figure for the proposed framework.

Outputs:
    results/final_workflow_figure.png
    results/final_workflow_figure.pdf
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PNG = Path("results/final_workflow_figure.png")
OUTPUT_PDF = Path("results/final_workflow_figure.pdf")

TITLE = "Proxy Acquisition-Aware Fetal Ultrasound Enhancement Framework"


def resolve_project_path(path_value: Path | str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def wrap_text(text: str, width: int = 26) -> str:
    lines: list[str] = []
    for raw_line in text.splitlines():
        if not raw_line.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw_line, width=width, break_long_words=False))
    return "\n".join(lines)


def add_box(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str = "#2F3A45",
    fontsize: int = 9,
):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        linewidth=1.3,
        edgecolor=edgecolor,
        facecolor=facecolor,
        mutation_aspect=1,
    )
    ax.add_patch(patch)

    ax.text(
        x + width / 2,
        y + height - 0.025,
        wrap_text(title, width=24),
        ha="center",
        va="top",
        fontsize=fontsize + 1,
        fontweight="bold",
        color="#17212B",
    )
    ax.text(
        x + width / 2,
        y + height / 2 - 0.01,
        wrap_text(body, width=28),
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#17212B",
        linespacing=1.15,
    )
    return patch


def box_anchor(
    xy: tuple[float, float],
    width: float,
    height: float,
    side: str,
) -> tuple[float, float]:
    x, y = xy
    if side == "left":
        return x, y + height / 2
    if side == "right":
        return x + width, y + height / 2
    if side == "top":
        return x + width / 2, y + height
    if side == "bottom":
        return x + width / 2, y
    raise ValueError(f"Unsupported side: {side}")


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = "#394B59",
    rad: float = 0.0,
    linewidth: float = 1.5,
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=linewidth,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(arrow)
    return arrow


def add_group_label(
    ax,
    xy: tuple[float, float],
    width: float,
    height: float,
    label: str,
    edgecolor: str,
):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.1,
        edgecolor=edgecolor,
        facecolor="none",
        linestyle="--",
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.012,
        y + height - 0.018,
        label,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=edgecolor,
    )


def create_workflow_figure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(17, 10), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.965,
        TITLE,
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="#111827",
    )

    main_y = 0.58
    main_h = 0.24
    box_w = 0.122
    gap = 0.018
    start_x = 0.035

    main_boxes = [
        (
            (start_x + 0 * (box_w + gap), main_y),
            "Input",
            "Low-quality fetal ultrasound image / sweep",
            "#F8FAFC",
        ),
        (
            (start_x + 1 * (box_w + gap), main_y),
            "1. Proxy acquisition-quality estimation",
            "blur\ncontrast\nnoise\nedge visibility\noptical flow / sweep motion",
            "#E8F1FF",
        ),
        (
            (start_x + 2 * (box_w + gap), main_y),
            "2. Quality score / learned embedding",
            "MLP proxy quality model\nframe label prediction",
            "#EEF7ED",
        ),
        (
            (start_x + 3 * (box_w + gap), main_y),
            "3. Adaptive enhancement decision",
            "good_frame: keep original\nblurry_frame: enhance\nlow_contrast_frame: enhance contrast\nnoisy_frame: denoise cautiously\npoor_frame: repeat / reacquire feedback",
            "#FFF4DE",
        ),
        (
            (start_x + 4 * (box_w + gap), main_y),
            "4. SR / diffusion enhancement",
            "Quality-aware CNN\nCNN-residual diffusion refinement",
            "#FDECEC",
        ),
        (
            (start_x + 5 * (box_w + gap), main_y),
            "5. Re-assessment",
            "recompute proxy quality\nupdate frame quality\nupdate sweep feedback",
            "#F3ECFF",
        ),
        (
            (start_x + 6 * (box_w + gap), main_y),
            "6. Final output",
            "enhanced fetal ultrasound\nfeedback decision:\naccept / repeat_sweep / move_slower / keep_probe_steady / improve_visibility",
            "#E9F8F4",
        ),
    ]

    for xy, title, body, facecolor in main_boxes:
        add_box(ax, xy, box_w, main_h, title, body, facecolor=facecolor, fontsize=8)

    for index in range(len(main_boxes) - 1):
        start = box_anchor(main_boxes[index][0], box_w, main_h, "right")
        end = box_anchor(main_boxes[index + 1][0], box_w, main_h, "left")
        add_arrow(ax, start, end)

    # Reassessment loop back to the adaptive decision.
    reassess_xy = main_boxes[5][0]
    decision_xy = main_boxes[3][0]
    add_arrow(
        ax,
        box_anchor(reassess_xy, box_w, main_h, "bottom"),
        (decision_xy[0] + box_w * 0.65, decision_xy[1] - 0.005),
        color="#7C3AED",
        rad=-0.35,
        linewidth=1.4,
    )
    ax.text(
        0.57,
        0.505,
        "feedback reassessment loop",
        ha="center",
        va="center",
        fontsize=9,
        color="#5B21B6",
        fontweight="bold",
    )

    # Validation branches.
    branch_y = 0.15
    branch_h = 0.23
    branch_w = 0.095
    branch_gap = 0.011

    add_group_label(
        ax,
        (0.035, 0.105),
        0.445,
        0.335,
        "A. HC18 validation branch",
        edgecolor="#1D4ED8",
    )
    hc18_boxes = [
        ((0.055, branch_y), "HC18", "reference fetal ultrasound images", "#EFF6FF"),
        (
            (0.055 + 1 * (branch_w + branch_gap), branch_y),
            "Degradation",
            "synthetic low-quality inputs",
            "#EFF6FF",
        ),
        (
            (0.055 + 2 * (branch_w + branch_gap), branch_y),
            "SR / diffusion evaluation",
            "CNN SR\nquality-aware CNN\nconditional diffusion\nCNN-residual diffusion",
            "#EFF6FF",
        ),
        (
            (0.055 + 3 * (branch_w + branch_gap), branch_y),
            "Metrics",
            "PSNR\nSSIM\nMAE\nedge score\nanatomy preservation",
            "#EFF6FF",
        ),
    ]
    for xy, title, body, facecolor in hc18_boxes:
        add_box(ax, xy, branch_w, branch_h, title, body, facecolor, fontsize=8)
    for index in range(len(hc18_boxes) - 1):
        add_arrow(
            ax,
            box_anchor(hc18_boxes[index][0], branch_w, branch_h, "right"),
            box_anchor(hc18_boxes[index + 1][0], branch_w, branch_h, "left"),
            color="#1D4ED8",
            linewidth=1.2,
        )

    add_group_label(
        ax,
        (0.52, 0.105),
        0.445,
        0.335,
        "B. ACOUSLIC-AI validation branch",
        edgecolor="#047857",
    )
    acouslic_boxes = [
        (
            (0.54, branch_y),
            "ACOUSLIC-AI .mha sweeps",
            "blind sweep volumes",
            "#ECFDF5",
        ),
        (
            (0.54 + 1 * (branch_w + branch_gap), branch_y),
            "Frame extraction",
            "sampled ultrasound frames",
            "#ECFDF5",
        ),
        (
            (0.54 + 2 * (branch_w + branch_gap), branch_y),
            "Proxy quality labels",
            "frame labels\nsweep feedback labels",
            "#ECFDF5",
        ),
        (
            (0.54 + 3 * (branch_w + branch_gap), branch_y),
            "Feedback report",
            "adaptive feedback change\nproxy improvement analysis",
            "#ECFDF5",
        ),
    ]
    for xy, title, body, facecolor in acouslic_boxes:
        add_box(ax, xy, branch_w, branch_h, title, body, facecolor, fontsize=8)
    for index in range(len(acouslic_boxes) - 1):
        add_arrow(
            ax,
            box_anchor(acouslic_boxes[index][0], branch_w, branch_h, "right"),
            box_anchor(acouslic_boxes[index + 1][0], branch_w, branch_h, "left"),
            color="#047857",
            linewidth=1.2,
        )

    # Link validation branches to the main pipeline.
    add_arrow(
        ax,
        (0.30, 0.44),
        box_anchor(main_boxes[4][0], box_w, main_h, "bottom"),
        color="#1D4ED8",
        rad=0.10,
        linewidth=1.1,
    )
    add_arrow(
        ax,
        (0.735, 0.44),
        box_anchor(main_boxes[1][0], box_w, main_h, "bottom"),
        color="#047857",
        rad=-0.18,
        linewidth=1.1,
    )

    ax.text(
        0.5,
        0.055,
        "Proxy feedback and quality labels are used for offline analysis; this figure does not imply real-time clinical deployment.",
        ha="center",
        va="center",
        fontsize=9,
        color="#4B5563",
    )

    output_png = resolve_project_path(OUTPUT_PNG)
    output_pdf = resolve_project_path(OUTPUT_PDF)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved workflow figure PNG: {OUTPUT_PNG}")
    print(f"Saved workflow figure PDF: {OUTPUT_PDF}")


def main() -> int:
    create_workflow_figure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
