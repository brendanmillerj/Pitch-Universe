"""Shared figure style -- a thin wrapper over figure_aesthetics/cell_typography.py.

Figures are drawn at print size (Cell full width, 174 mm), so the canvas scale
R = 1 and every font/stroke size is the printed size. Production rules live in
figure_aesthetics/cell_figure_guidelines.md: Arial (falls back with a warning
where it isn't installed), 6-8 pt text, capital panel letters, no red + green.

`save` writes the Cell production PDF plus a 300-dpi PNG for quick review.
"""
import sys

from matplotlib.colors import LinearSegmentedColormap

from mlbpitch import paths

sys.path.insert(0, str(paths.ROOT / "figure_aesthetics"))
from cell_typography import cell_lw, cell_save, cell_sizes, cell_tag, style_axes  # noqa: E402

PRINT_W = 174 / 25.4
CMAP_DIVERGING = LinearSegmentedColormap.from_list("z", ["#4575b4", "white", "#d73027"])

# Fastballs warm, breaking balls cool, offspeed neutral/gold -- no green, so the
# palette never puts red and green together.
PITCH_COLORS = {
    "FF": "#d73027", "SI": "#f46d43", "FC": "#8c510a",
    "SL": "#4575b4", "ST": "#74add1", "CU": "#542788",
    "CH": "#7f7f7f", "FS": "#e0a800",
}
PITCH_NAMES = {
    "FF": "Four-seam", "SI": "Sinker", "FC": "Cutter", "SL": "Slider",
    "ST": "Sweeper", "CU": "Curveball", "CH": "Changeup", "FS": "Splitter",
}


def apply_style():
    return cell_sizes(canvas_width=PRINT_W, print_layout="full")


def save(fig, stem):
    import matplotlib as mpl
    # cell_save writes the PDF without a dpi, so rasterized layers would embed at
    # the 100-dpi default (~72 ppi at print); Cell needs >= 300 ppi.
    with mpl.rc_context({"savefig.dpi": 300}):
        cell_save(fig, f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=300)
    import matplotlib.pyplot as plt
    plt.close(fig)


__all__ = ["apply_style", "save", "cell_tag", "cell_lw", "style_axes",
           "CMAP_DIVERGING", "PITCH_COLORS", "PITCH_NAMES", "PRINT_W"]
