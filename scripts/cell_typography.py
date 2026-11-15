"""
cell_typography.py — Auto-optimized typography for Cell journal figures (matplotlib)
==============================================================================
Python equivalent of cell_typography.R. Same canvas widths, same scaling,
same targets, so R and Python panels match when combined.

Usage:
    from cell_typography import *
    sz = cell_sizes(max_panels_per_row=3)     # sets rcParams + module SZ_* values
    fig, axes = plt.subplots(1, 3, figsize=(sz.canvas_w, 4.5))
    for ax, tag in zip(axes, "ABC"):
        style_axes(ax)
        cell_tag(ax, tag)
    ax.plot(x, y, lw=cell_lw(0.75))           # 0.75 pt at print
    ax.text(x, y, "n.s.", fontsize=sz.annot)  # 6 pt at print
    cell_save(fig, "Figure 1.pdf")
    cell_save(fig, "Figure 1.tif")            # LZW TIFF, dpi auto

The figure is drawn on a canvas (canvas_w inches) larger than the printed
width. Sizes are multiplied by R = canvas / print so that, after the journal
shrinks the figure to print width, text lands at the target point sizes.

Cell production rules enforced or checked here:
  - Arial, embedded as TrueType (pdf.fonttype = 42)
  - Text ~6-8 pt at print size (warns if a target is outside)
  - Strokes 0.5-1.5 pt at print size (warns if outside)
  - Print widths 85 / 114 / 174 mm (55 mm for 3-column formats)
  - Print height <= 20 cm (warns)
  - TIFF: LZW, >= 300 / 500 / 1000 ppi at print size
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

__all__ = [
    "cell_sizes", "theme_cell", "style_axes", "cell_tag", "cell_lw",
    "cell_text_size", "cell_save", "CellSizes",
    "REVIEWED_COLOR", "UNREVIEWED_COLOR", "REVIEWED_DARK", "UNREVIEWED_DARK",
    "DB_FILL", "DB_LINE", "DB_LEVELS",
]

PRINT_MM = {"full": 174, "1.5col": 114, "single": 85, "preview_1col": 55}
MAX_PRINT_H_IN = 200 / 25.4  # 20 cm
MIN_PPI = {"color": 300, "bw": 500, "lineart": 1000}
WIDTH_MAP = {1: 7, 2: 10, 3: 14, 4: 16}

DEFAULT_TARGETS = dict(tag=8, title=7.8, axis_ttl=7, axis_text=6, legend=6, annot_pt=6)


@dataclass
class CellSizes:
    tag: float
    title: float
    axis_ttl: float
    axis_text: float
    legend: float
    annot: float      # matplotlib fontsize (pt on canvas); same as other text here
    line: float       # matplotlib linewidth (pt on canvas)
    canvas_w: float
    print_w: float
    R: float


# Module-level state, mirroring the R globals.
_SZ: CellSizes | None = None


def _need_sizes() -> CellSizes:
    if _SZ is None:
        raise RuntimeError("Call cell_sizes() first.")
    return _SZ


def _pick_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    if "Arial" in available:
        return "Arial"
    warnings.warn("Arial not found. Falling back to a metric-compatible font. "
                  "Install Arial before exporting production figures.")
    for alt in ("Liberation Sans", "Arimo", "Helvetica", "DejaVu Sans"):
        if alt in available:
            return alt
    return "sans-serif"


def cell_sizes(
    max_panels_per_row: int = 1,
    canvas_width: float | None = None,
    has_facets: bool = False,
    print_layout: str = "full",
    target_pts: dict | None = None,
    line_pt: float = 0.75,
    apply_theme: bool = True,
) -> CellSizes:
    """Compute canvas-scaled sizes and (by default) apply theme_cell()."""
    global _SZ

    if print_layout not in PRINT_MM:
        raise ValueError(f"print_layout must be one of: {', '.join(PRINT_MM)}")

    tp = {**DEFAULT_TARGETS, **(target_pts or {})}
    bad = {k: v for k, v in tp.items() if v < 6 or v > 8}
    if bad:
        warnings.warn(f"Outside Cell's 6-8 pt text range at print: {bad}")
    if not 0.5 <= line_pt <= 1.5:
        warnings.warn(f"line_pt = {line_pt} is outside Cell's 0.5-1.5 pt stroke range")

    if canvas_width is None:
        n = min(max(int(max_panels_per_row), 1), 4)
        canvas_width = WIDTH_MAP[n] + (1 if has_facets else 0)

    print_w = PRINT_MM[print_layout] / 25.4
    R = canvas_width / print_w

    # No integer rounding: rounding at canvas scale can push print size below 6 pt.
    _SZ = CellSizes(
        tag=tp["tag"] * R,
        title=tp["title"] * R,
        axis_ttl=tp["axis_ttl"] * R,
        axis_text=tp["axis_text"] * R,
        legend=tp["legend"] * R,
        annot=tp["annot_pt"] * R,
        line=line_pt * R,
        canvas_w=canvas_width,
        print_w=print_w,
        R=R,
    )

    # Expose R-style names at module level.
    g = globals()
    g.update(SZ_TAG=_SZ.tag, SZ_TITLE=_SZ.title, SZ_AXIS_TTL=_SZ.axis_ttl,
             SZ_AXIS_TEXT=_SZ.axis_text, SZ_LEGEND=_SZ.legend, SZ_ANNOT=_SZ.annot,
             SZ_LINE=_SZ.line, CANVAS_W=canvas_width, CELL_R=R, PRINT_W_IN=print_w)

    if apply_theme:
        theme_cell()
    return _SZ


def theme_cell() -> None:
    """Set matplotlib rcParams to the Cell style (equivalent of theme_cell() in R)."""
    sz = _need_sizes()
    font = _pick_font()
    mpl.rcParams.update({
        # Fonts: Arial, embedded as TrueType (Type 42), editable text in SVG
        "font.family": "sans-serif",
        "font.sans-serif": [font, "Liberation Sans", "Arimo", "DejaVu Sans"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "mathtext.default": "regular",
        # Text sizes
        "font.size": sz.axis_text,
        "axes.titlesize": sz.title,
        "axes.titleweight": "normal",
        "axes.labelsize": sz.axis_ttl,
        "xtick.labelsize": sz.axis_text,
        "ytick.labelsize": sz.axis_text,
        "legend.fontsize": sz.legend,
        "legend.title_fontsize": sz.legend,
        "legend.frameon": False,
        # Colors
        "text.color": "black",
        "axes.labelcolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "axes.edgecolor": "black",
        # Full black border, no grid (ggplot panel.border + blank grid)
        "axes.grid": False,
        "axes.spines.top": True,
        "axes.spines.right": True,
        "axes.linewidth": sz.line,
        "lines.linewidth": sz.line,
        "patch.linewidth": sz.line,
        # theme_minimal has no ticks
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.minor.size": 0,
        "ytick.minor.size": 0,
        "xtick.major.width": sz.line,
        "ytick.major.width": sz.line,
        # Output
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    })


def style_axes(ax, legend: bool = False) -> None:
    """Per-axes touches rcParams can't cover (legend off by default, like R)."""
    sz = _need_sizes()
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(sz.line)
        s.set_color("black")
    if not legend and ax.get_legend() is not None:
        ax.get_legend().remove()


def cell_tag(ax, label: str, x: float = -0.12, y: float = 1.05) -> None:
    """Panel label. Cell requires capital letters."""
    sz = _need_sizes()
    if not label.isupper():
        warnings.warn(f"Panel label '{label}' is not capitalized; Cell requires capitals.")
    ax.text(x, y, label, transform=ax.transAxes, fontsize=sz.tag,
            fontweight="bold", va="bottom", ha="right")


def cell_lw(pt: float = 0.75) -> float:
    """Point size at print -> matplotlib linewidth on the canvas."""
    sz = _need_sizes()
    if not 0.5 <= pt <= 1.5:
        warnings.warn(f"Stroke {pt} pt is outside Cell's 0.5-1.5 pt range")
    return pt * sz.R


def cell_text_size(pt: float = 6) -> float:
    """Point size at print -> matplotlib fontsize on the canvas."""
    sz = _need_sizes()
    if not 6 <= pt <= 8:
        warnings.warn(f"Text {pt} pt is outside Cell's 6-8 pt range")
    return pt * sz.R


def cell_save(fig, filename: str | Path, image_type: str = "color") -> Path:
    """Save as PDF (fonts embedded) or LZW TIFF at the dpi needed for print ppi.

    Size is taken from the figure (set figsize=(sz.canvas_w, height)).
    """
    sz = _need_sizes()
    if image_type not in MIN_PPI:
        raise ValueError(f"image_type must be one of: {', '.join(MIN_PPI)}")

    filename = Path(filename)
    w, h = fig.get_size_inches()
    print_w, print_h = w / sz.R, h / sz.R
    if print_w > sz.print_w + 1e-6:
        warnings.warn(f"Print width {print_w:.2f} in exceeds the chosen layout ({sz.print_w:.2f} in)")
    if print_h > MAX_PRINT_H_IN:
        warnings.warn(f"Print height {print_h:.2f} in exceeds Cell's 20 cm ({MAX_PRINT_H_IN:.2f} in) limit")

    ext = filename.suffix.lower()
    if ext == ".pdf":
        fig.savefig(filename, format="pdf")
    elif ext in (".tif", ".tiff"):
        dpi = math.ceil(MIN_PPI[image_type] / sz.R)  # canvas dpi -> print ppi
        fig.savefig(filename, format="tiff", dpi=dpi,
                    pil_kwargs={"compression": "tiff_lzw"})
    else:
        raise ValueError("Use .pdf or .tif/.tiff (Cell's preferred production formats)")

    print(f"Saved {filename} | canvas {w:.2f} x {h:.2f} in | "
          f"print {print_w * 2.54:.1f} x {print_h * 2.54:.1f} cm")
    return filename


# Database identity colors (Reviewed / Unreviewed)
REVIEWED_COLOR = "#dbe8f3"
UNREVIEWED_COLOR = "#4f86b8"
REVIEWED_DARK = "#9cb7cf"
UNREVIEWED_DARK = "#355a7a"

DB_FILL = {"Reviewed": REVIEWED_COLOR, "Unreviewed": UNREVIEWED_COLOR}
DB_LINE = {"Reviewed": REVIEWED_DARK, "Unreviewed": UNREVIEWED_DARK}
DB_LEVELS = ["Unreviewed", "Reviewed"]
