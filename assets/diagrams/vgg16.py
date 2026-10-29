"""Draw assets/vgg16.png: the VGG16 architecture with the output shape after every block, for notebook 01.

Five convolutional blocks (3 x 3 filters, "same" padding, ReLU), each followed by 2 x 2 max pooling, then three
dense layers. Shapes as printed by keras.applications.VGG16().summary().

Run from the repository root: uv run python assets/diagrams/vgg16.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

OUT = Path(__file__).parent.parent / "vgg16.png"
INK, GREY, BLUE, AMBER, GREEN = "#222222", "#6b6b6b", "#2f6fb0", "#c47612", "#3a8a4f"

fig, ax = plt.subplots(figsize=(17, 5.8), dpi=200)
fig.patch.set_facecolor("white")
ax.set_xlim(0, 36.5)
ax.set_ylim(-4.2, 5.3)
ax.set_aspect("equal")
ax.axis("off")


def block(x, side, depth, colour, top_label, bottom_label):
    """Draw a feature-map box: a square front face of the given side and a receding depth."""
    dx, dy = depth * 0.55, depth * 0.4
    y = 0.3 - side / 2
    ax.add_patch(Polygon([(x, y), (x + side, y), (x + side, y + side), (x, y + side)], facecolor="white",
                         edgecolor=colour, linewidth=1.4))
    ax.add_patch(Polygon([(x, y + side), (x + dx, y + side + dy), (x + side + dx, y + side + dy),
                          (x + side, y + side)], facecolor="#f4f4f4", edgecolor=colour, linewidth=1.4))
    ax.add_patch(Polygon([(x + side, y), (x + side + dx, y + dy), (x + side + dx, y + side + dy),
                          (x + side, y + side)], facecolor="#e8e8e8", edgecolor=colour, linewidth=1.4))
    ax.text(x + (side + dx) / 2, 3.3, top_label, ha="center", va="bottom", fontsize=9.5, color=colour,
            linespacing=1.3)
    ax.text(x + (side + dx) / 2, -2.9, bottom_label, ha="center", va="top", fontsize=9.5, color=INK,
            linespacing=1.3)
    return x + side + dx


def arrow(x0, x1, colour=GREY):
    ax.add_patch(FancyArrowPatch((x0 + 0.1, 0.3), (x1 - 0.1, 0.3), arrowstyle="-|>,head_length=4,head_width=2.5",
                                 color=colour, linewidth=1.2))


x = 0.3
end = block(x, 3.4, 0.35, INK, "input", "224 × 224 × 3")
blocks = [  # (side of the drawn square, depth, label above, shape after pooling)
    (3.4, 1.0, "block 1\n2 conv, 64", "112 × 112 × 64"),
    (2.6, 1.5, "block 2\n2 conv, 128", "56 × 56 × 128"),
    (1.9, 2.2, "block 3\n3 conv, 256", "28 × 28 × 256"),
    (1.3, 3.0, "block 4\n3 conv, 512", "14 × 14 × 512"),
    (0.8, 3.0, "block 5\n3 conv, 512", "7 × 7 × 512"),
]
for side, depth, top, bottom in blocks:
    x = end + 1.0
    arrow(end, x, BLUE)
    end = block(x, side, depth, BLUE, top, bottom)
x = end + 1.0
arrow(end, x)
for units, height, label in [(25088, 4.6, "flatten\n25,088"), (4096, 3.4, "dense\n4,096"), (4096, 3.4, "dense\n4,096"),
                             (1000, 2.0, "softmax\n1,000")]:
    ax.add_patch(Rectangle((x, 0.3 - height / 2), 0.45, height, facecolor="white",
                           edgecolor=GREY if units == 25088 else GREEN, linewidth=1.5))
    ax.text(x + 0.22, -2.9, label, ha="center", va="top", fontsize=9.5, color=INK, linespacing=1.3)
    end = x + 0.45
    x = end + 1.7
    if units != 1000:
        arrow(end, x, GREEN)

ax.text(11.0, 4.95, "convolutional base: 13 convolutional layers with 3 × 3 filters, each block ends with 2 × 2 max pooling",
        ha="center", fontsize=11, color=BLUE)
ax.text(31.2, 4.95, "classifier: 3 dense layers", ha="center", fontsize=11, color=GREEN)
fig.savefig(OUT, facecolor="white", bbox_inches="tight")
