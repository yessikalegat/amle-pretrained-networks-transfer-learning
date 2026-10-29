"""Draw assets/transfer_learning.png for notebook 02: the three ways to reuse a pretrained network.

A. Feature extraction: the frozen convolutional base turns every image into a feature vector once; a new
   classifier (logistic regression or a small dense head) is trained on the vectors.
B. Fine-tuning: after the new head has been trained, the top blocks of the base are unfrozen and trained
   together with the head, with a low learning rate.

Run from the repository root: uv run python assets/diagrams/transfer_learning.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent.parent / "transfer_learning.png"
INK, GREY, BLUE, AMBER, GREEN, PALE_BLUE = "#222222", "#6b6b6b", "#2f6fb0", "#c47612", "#3a8a4f", "#e3edf7"

fig, ax = plt.subplots(figsize=(20, 8.4), dpi=200)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 30)
ax.set_ylim(-11.0, 1.4)
ax.set_aspect("equal")
ax.axis("off")


def box(x, y, w, h, text, edge, face="white", size=10.5, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", facecolor=face,
                                edgecolor=edge, linewidth=1.6))
    ax.text(x + w / 2, y - h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.35,
            fontweight=weight)


def arrow(x0, y0, x1, y1, colour=GREY, text=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=5,head_width=3", color=colour,
                                 linewidth=1.4))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.25, text, ha="center", va="bottom", fontsize=9, color=colour)


def base_blocks(x, y, frozen_top=True, label="convolutional base\npretrained on ImageNet"):
    """Draw the base as five stacked blocks; the top two are coloured differently when they are trained."""
    for i in range(5):
        trained = (not frozen_top) and i >= 3
        box(x + i * 1.25, y - 0.25 * i, 1.05, 2.6 - 0.35 * i, f"block\n{i + 1}", AMBER if trained else BLUE,
            face="#fdf1e2" if trained else PALE_BLUE, size=8.5)
    ax.text(x + 3.0, y + 0.45, label, ha="center", va="bottom", fontsize=10, color=BLUE, linespacing=1.3)


# Panel A
ax.text(0, 1.05, "A. Feature extraction: the base stays frozen", fontsize=13.5, color=INK, va="center")
box(0.2, -1.0, 2.2, 1.6, "image\n160 × 160 × 3", INK)
arrow(2.5, -1.8, 3.4, -1.8)
base_blocks(3.5, -0.6)
ax.text(6.5, -3.85, "frozen: weights are not updated", ha="center", fontsize=9.5, color=BLUE)
arrow(9.9, -1.8, 11.4, -1.8, text="average\npooling")
box(11.5, -0.8, 2.6, 2.0, "feature vector\n1,280 values", GREY)
arrow(14.2, -1.4, 16.3, -0.55)
arrow(14.2, -2.2, 16.3, -3.05)
box(16.4, 0.15, 6.2, 1.4, "logistic regression (scikit-learn)", GREEN)
box(16.4, -2.35, 6.2, 1.4, "dense head (Keras): Dense, Dropout,\nsoftmax output", GREEN)
arrow(22.7, -0.55, 24.3, -1.6)
arrow(22.7, -3.05, 24.3, -2.0)
box(24.4, -1.0, 3.6, 1.6, "3 classes:\nhealthy, rust,\nleaf spot", INK, size=9.5)
ax.text(19.5, -4.1, "only the new head is trained", ha="center", fontsize=9.5, color=GREEN)

# Panel B
y0 = -6.4
ax.text(0, y0 + 0.9, "B. Fine-tuning: the top blocks of the base are trained as well", fontsize=13.5, color=INK,
        va="center")
box(0.2, y0 - 1.0, 2.2, 1.6, "image\n160 × 160 × 3", INK)
arrow(2.5, y0 - 1.8, 3.4, y0 - 1.8)
base_blocks(3.5, y0 - 0.6, frozen_top=False, label="")
ax.text(6.5, y0 - 3.85, "blocks 1 to 3 frozen, blocks 4 and 5 trained", ha="center", fontsize=9.5, color=AMBER)
arrow(9.9, y0 - 1.8, 11.4, y0 - 1.8)
box(11.5, y0 - 0.9, 4.6, 1.8, "dense head\n(already trained in A)", GREEN)
arrow(16.2, y0 - 1.8, 17.8, y0 - 1.8)
box(17.9, y0 - 1.0, 3.6, 1.6, "3 classes", INK)
ax.text(23.0, y0 - 0.6, "low learning rate, for example 0.00001:\nthe pretrained filters only change a little,\n"
        "so what they learnt on ImageNet is not lost", ha="left", va="top", fontsize=10, color=AMBER, linespacing=1.45)
fig.savefig(OUT, facecolor="white", bbox_inches="tight")
