"""Draw assets/mobilenet_blocks.png and assets/mobilenetv2.png for notebook 01: the ideas and the architecture of MobileNetV2.

A. A standard convolution against a depthwise separable convolution, with their parameter counts for 96 input and
   96 output channels and 3 x 3 filters.
B. The inverted residual block: expand with a 1 x 1 convolution, filter each channel with a 3 x 3 depthwise
   convolution, project back with a 1 x 1 convolution, and add the input (shortcut).
C. The whole network: a first convolution, 17 inverted residual blocks in 7 stages, a 1 x 1 convolution, global
   average pooling and one dense layer. Shapes as printed by keras.applications.MobileNetV2().summary().

Run from the repository root: uv run python assets/diagrams/mobilenetv2.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

ASSETS = Path(__file__).parent.parent
INK, GREY, BLUE, AMBER, GREEN, RED = "#222222", "#6b6b6b", "#2f6fb0", "#c47612", "#3a8a4f", "#b03a2e"
PALE_BLUE, PALE_AMBER = "#e3edf7", "#fdf1e2"



def new_axes(width, height, xlim, ylim):
    global fig, ax
    fig, ax = plt.subplots(figsize=(width, height), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")


def box(x, y, w, h, text, edge, face="white", size=10):
    ax.add_patch(FancyBboxPatch((x, y - h), w, h, boxstyle="round,pad=0.02,rounding_size=0.2", facecolor=face,
                                edgecolor=edge, linewidth=1.5))
    ax.text(x + w / 2, y - h / 2, text, ha="center", va="center", fontsize=size, color=INK, linespacing=1.3)


def arrow(x0, y0, x1, y1, colour=GREY, text=None, size=9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>,head_length=4,head_width=2.5", color=colour,
                                 linewidth=1.3))
    if text:
        ax.text((x0 + x1) / 2, max(y0, y1) + 0.2, text, ha="center", va="bottom", fontsize=size, color=colour)


def stack(x, y, n, colour, label, w=0.9, h=0.9, step=0.12):
    """A stack of n squares (one per channel shown), front square at (x, y - h)."""
    for i in reversed(range(n)):
        ax.add_patch(Rectangle((x + i * step, y - h + i * step), w, h, facecolor="white", edgecolor=colour,
                               linewidth=1.1))
    ax.text(x + (w + n * step) / 2, y - h - 0.25, label, ha="center", va="top", fontsize=9, color=INK)


# ---------------- Figure 1, panel A: standard vs depthwise separable convolution ----------------
new_axes(15, 10.5, (0, 26), (-17.6, 1.4))
ax.text(0, 1.0, "A. Standard and depthwise separable convolution (96 channels in, 96 out, 3 × 3 filters)",
        fontsize=13, color=INK, va="center")
ax.text(0.2, -0.4, "standard convolution", fontsize=11.5, color=RED, va="center")
stack(0.4, -1.2, 6, GREY, "input\n96 channels")
arrow(2.2, -1.9, 8.0, -1.9, RED, "96 filters of 3 × 3 × 96")
stack(8.2, -1.2, 6, RED, "output\n96 channels")
ax.text(16.6, -1.9, "3 · 3 · 96 · 96\n= 82,944 weights", fontsize=11, color=RED, va="center")

ax.text(0.2, -4.6, "depthwise separable convolution", fontsize=11.5, color=GREEN, va="center")
stack(0.4, -5.4, 6, GREY, "input\n96 channels")
arrow(2.2, -6.1, 8.0, -6.1, GREEN, "1. depthwise: one 3 × 3 filter\nper channel, no mixing", size=9)
stack(8.2, -5.4, 6, GREEN, "96 channels,\nfiltered separately")
arrow(10.0, -6.1, 15.8, -6.1, GREEN, "2. pointwise: 96 filters of\n1 × 1 × 96 mix the channels", size=9)
stack(16.0, -5.4, 6, GREEN, "output\n96 channels")
ax.text(18.4, -6.1, "3 · 3 · 96 + 96 · 96\n= 864 + 9,216\n= 10,080 weights", fontsize=11, color=GREEN,
        va="center")
ax.text(12.0, -8.4, "same input and output shape, about 8 times fewer weights and computations", fontsize=11,
        color=INK, ha="center")

# ---------------- Figure 1, panel B: inverted residual block ----------------
yb = -10.2
ax.text(0, yb, "B. Inverted residual block (17 of them in MobileNetV2)", fontsize=13, color=INK, va="center")
box(0.2, yb - 1.4, 3.6, 1.6, "input\n56 × 56 × 24", GREY)
arrow(3.9, yb - 2.2, 4.9, yb - 2.2)
box(5.0, yb - 1.4, 4.2, 1.6, "1 × 1 conv\nexpand to 144 channels", BLUE, PALE_BLUE)
arrow(9.3, yb - 2.2, 10.3, yb - 2.2)
box(10.4, yb - 1.4, 4.2, 1.6, "3 × 3 depthwise conv\n144 channels", GREEN, "#e7f2ea")
arrow(14.7, yb - 2.2, 15.7, yb - 2.2)
box(15.8, yb - 1.4, 4.2, 1.6, "1 × 1 conv\nproject to 24 channels", BLUE, PALE_BLUE)
arrow(20.1, yb - 2.2, 21.1, yb - 2.2)
box(21.2, yb - 1.6, 2.4, 1.2, "add", AMBER, PALE_AMBER)
ax.add_patch(FancyArrowPatch((2.0, yb - 3.1), (22.4, yb - 2.9), connectionstyle="arc3,rad=0.18",
                             arrowstyle="-|>,head_length=4,head_width=2.5", color=AMBER, linewidth=1.3))
ax.text(12.0, yb - 4.3, "shortcut: the input is added to the output of the block", fontsize=10, color=AMBER,
        ha="center")
ax.text(0.2, yb - 5.2, "Narrow (24 channels) at the input and output, wide (144) in the middle, where the depthwise "
        "convolution needs few weights:\nthe reverse of earlier designs, hence \"inverted\". The shortcut lets the "
        "signal and the gradients pass directly through very deep networks.", fontsize=10.5, color=INK, va="top",
        linespacing=1.45)
fig.savefig(ASSETS / "mobilenet_blocks.png", facecolor="white", bbox_inches="tight")
plt.close(fig)

# ---------------- Figure 2: the whole network ----------------
new_axes(18, 6.4, (0, 36), (-21.8, -9.6))
yc = -11.0
ax.text(0, yc + 0.7, "MobileNetV2 for 224 × 224 colour images: output shape after each stage", fontsize=13,
        color=INK, va="center")
STAGES = [  # (label above, shape below, side of the drawn square, depth)
    ("input", "224 × 224 × 3", 3.2, 0.25),
    ("3 × 3 conv\nstride 2", "112 × 112 × 32", 2.6, 0.6),
    ("1 block", "112 × 112 × 16", 2.6, 0.45),
    ("2 blocks", "56 × 56 × 24", 2.0, 0.6),
    ("3 blocks", "28 × 28 × 32", 1.5, 0.75),
    ("4 blocks", "14 × 14 × 64", 1.1, 1.1),
    ("3 blocks", "14 × 14 × 96", 1.1, 1.4),
    ("3 blocks", "7 × 7 × 160", 0.8, 1.9),
    ("1 block", "7 × 7 × 320", 0.8, 2.6),
    ("1 × 1 conv", "7 × 7 × 1,280", 0.8, 3.6),
]
x = 0.2
for i, (top, bottom, side, depth) in enumerate(STAGES):
    dx, dy = depth * 0.5, depth * 0.35
    y = yc - 4.0 + side / 2
    colour = INK if i == 0 else (BLUE if "conv" in top else GREEN)
    ax.add_patch(Polygon([(x, y - side), (x + side, y - side), (x + side, y), (x, y)], facecolor="white",
                         edgecolor=colour, linewidth=1.3))
    ax.add_patch(Polygon([(x, y), (x + dx, y + dy), (x + side + dx, y + dy), (x + side, y)], facecolor="#f4f4f4",
                         edgecolor=colour, linewidth=1.3))
    ax.add_patch(Polygon([(x + side, y - side), (x + side + dx, y - side + dy), (x + side + dx, y + dy),
                          (x + side, y)], facecolor="#e8e8e8", edgecolor=colour, linewidth=1.3))
    ax.text(x + (side + dx) / 2, yc - 1.1, top, ha="center", va="bottom", fontsize=9, color=colour, linespacing=1.2)
    ax.text(x + (side + dx) / 2, yc - 6.6, bottom, ha="center", va="top", fontsize=9, color=INK)
    x_end = x + side + dx
    x = x_end + 0.75
    arrow(x_end + 0.08, yc - 4.0, x - 0.08, yc - 4.0)
ax.add_patch(Rectangle((x, yc - 5.6), 0.45, 3.2, facecolor="white", edgecolor=GREY, linewidth=1.4))
ax.text(x + 0.22, yc - 6.6, "global\naverage\npooling\n1,280", ha="center", va="top", fontsize=9, color=INK,
        linespacing=1.2)
arrow(x + 0.5, yc - 4.0, x + 1.4, yc - 4.0)
ax.add_patch(Rectangle((x + 1.5, yc - 5.0), 0.45, 2.0, facecolor="white", edgecolor=GREEN, linewidth=1.4))
ax.text(x + 1.72, yc - 6.6, "dense\nsoftmax\n1,000", ha="center", va="top", fontsize=9, color=INK, linespacing=1.2)
ax.text(17.5, yc - 9.6, "17 inverted residual blocks in 7 stages (green). Only one dense layer, after global average "
        "pooling: 3.5 million parameters in total, against 138 million for VGG16.", fontsize=10.5, color=INK,
        ha="center")
fig.savefig(ASSETS / "mobilenetv2.png", facecolor="white", bbox_inches="tight")
