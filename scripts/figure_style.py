"""Shared figure style -- a thin wrapper over cell_typography.py (vendored from figure_aesthetics/).

Figures are drawn at print size (Cell full width, 174 mm), so the canvas scale
R = 1 and every font/stroke size is the printed size. Production rules live in
figure_aesthetics/cell_figure_guidelines.md: Arial (falls back with a warning
where it isn't installed), 6-8 pt text, capital panel letters, no red + green.

`save` writes the Cell production PDF plus a 300-dpi PNG for quick review.
"""
import sys

from matplotlib.colors import LinearSegmentedColormap

from mlbpitch import paths

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


# Arial is not installed on CARC; cell_typography then falls back to DejaVu Sans, which looks nothing like Arial.
# First choice: real Arial in figure_aesthetics/fonts/ (Microsoft core fonts, copied from the riboseq revision;
# gitignored, since the EULA allows use but not redistribution). Second: ArcGIS's Arial Unicode MS under /apps
# (the same Monotype letterforms, regular and bold).
ARIAL_LOCAL = sorted((paths.ROOT / "figure_aesthetics" / "fonts").glob("Arial*.ttf"))
ARIAL_FALLBACK = ["/apps/generic/arcgis/server/fonts/ARIALUNI.TTF", "/apps/generic/arcgis/server/fonts/ArialUnicode-Bold.ttf"]


def _arial():
    import matplotlib as mpl
    from matplotlib import font_manager
    for path in ARIAL_LOCAL:
        font_manager.fontManager.addfont(str(path))
    have = {f.name for f in font_manager.fontManager.ttflist}
    if "Arial" in have:
        mpl.rcParams["font.family"] = "sans-serif"
        mpl.rcParams["font.sans-serif"] = ["Arial"] + [f for f in mpl.rcParams["font.sans-serif"] if f != "Arial"]
        return "Arial"
    for path in ARIAL_FALLBACK:
        try:
            font_manager.fontManager.addfont(path)
        except (FileNotFoundError, RuntimeError):
            pass
    if "Arial Unicode MS" in {f.name for f in font_manager.fontManager.ttflist}:
        mpl.rcParams["font.sans-serif"] = ["Arial Unicode MS"] + list(mpl.rcParams["font.sans-serif"])
        return "Arial Unicode MS"
    return None


def apply_style():
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Arial not found")
        sz = cell_sizes(canvas_width=PRINT_W, print_layout="full")
    if _arial() is None:
        warnings.warn("Neither Arial nor Arial Unicode MS is available; figures use a fallback font")
    return sz


def save(fig, stem):
    import matplotlib as mpl
    # cell_save writes the PDF without a dpi, so rasterized layers would embed at
    # the 100-dpi default (~72 ppi at print); Cell needs >= 300 ppi.
    with mpl.rc_context({"savefig.dpi": 300}):
        cell_save(fig, f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=300)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ====================================================================== paper figures (figures/figureN/)
# The look of the riboseq revision figures (panda_riboseq_revision/figures/): every axes placed by hand in inches
# from the top-left of a 174 mm canvas, a bold letter with a short title beside it, open L axes for bars and lines,
# boxed maps and heatmaps, rounded flowchart boxes, legends beside their panels, 0.6 pt lines.
W = PRINT_W                                   # 6.85 in
H_MAX = 200 / 25.4                            # Cell: at most 20 cm tall
# Text sizes: every role at the top of Cell's 6-8 pt range (user 2026-11-06, twice: smaller text read too small);
# the panel letter is set apart by weight, not size.
PAPER_TARGETS = dict(tag=8, title=8, axis_ttl=8, axis_text=8, legend=8, annot_pt=8)
T_TAG, T_TTL, T_AX, T_TXT = 8.0, 8.0, 8.0, 8.0
LW = 0.6
HILITE, HILITE_DARK = "#a4c8ff", "#3d5a8a"    # flowchart key box (black / white text)
GREY, LIGHT = "#7f7f7f", "#d9d9d9"
DASH, DOTTED = (0, (2, 1.5)), (0, (0.6, 1.2))


def paper_style():
    """cell_typography at PAPER_TARGETS plus the riboseq marks: 2 pt ticks, 0.6 pt lines, black text."""
    import warnings
    import matplotlib as mpl
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Arial not found")
        sz = cell_sizes(canvas_width=PRINT_W, print_layout="full", target_pts=PAPER_TARGETS)
    if _arial() is None:
        warnings.warn("Neither Arial nor Arial Unicode MS is available; figures use a fallback font")
    assert (sz.tag, sz.title, sz.axis_ttl, sz.axis_text, sz.legend) == (T_TAG, T_TTL, T_AX, T_TXT, T_TXT), \
        "figure_style sizes drifted from PAPER_TARGETS"
    mpl.rcParams.update({
        "font.size": T_TXT, "axes.linewidth": LW, "axes.labelsize": T_AX, "axes.titlesize": T_TTL,
        "xtick.labelsize": T_TXT, "ytick.labelsize": T_TXT, "legend.fontsize": T_TXT,
        "xtick.major.width": LW, "ytick.major.width": LW, "xtick.major.size": 2, "ytick.major.size": 2,
        "xtick.minor.width": LW * 0.8, "ytick.minor.width": LW * 0.8, "xtick.minor.size": 1.2, "ytick.minor.size": 1.2,
        "xtick.major.pad": 1.5, "ytick.major.pad": 1.5, "xtick.direction": "out", "ytick.direction": "out",
        "axes.labelpad": 1.5, "axes.grid": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "text.color": "#000000", "axes.labelcolor": "#000000", "xtick.color": "#000000", "ytick.color": "#000000",
        "axes.edgecolor": "#000000", "mathtext.default": "regular", "axes.unicode_minus": True,
        "hatch.linewidth": 0.5, "legend.frameon": False, "savefig.facecolor": "white",
    })
    return sz


def canvas(h):
    import matplotlib.pyplot as plt
    assert h <= H_MAX + 1e-6, f"figure height {h:.2f} in exceeds 20 cm"
    # 300 dpi from the start: at 100 dpi 6 pt text is ~8 px and hinting makes it measure ~15% narrower than it prints,
    # so anything placed by measured extents (legend_strip, title runs) would collide in the PNG/TIF/PDF.
    return plt.figure(figsize=(W, h), dpi=300)


def iax(fig, x, y, w, h, **kw):
    """Axes at (x, y) inches from the figure's top-left corner, w x h inches."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x / fw, 1 - (y + h) / fh, w / fw, h / fh], **kw)


def ftext(fig, x, y, s, size=T_TXT, **kw):
    """Figure text at (x, y) inches from the top-left; va defaults to top, ha to left."""
    fw, fh = fig.get_size_inches()
    kw.setdefault("va", "top")
    kw.setdefault("ha", "left")
    return fig.text(x / fw, 1 - y / fh, s, fontsize=size, **kw)


def tag(fig, x, y, letter, title=None, subtitle=None):
    """Bold 8 pt panel letter at (x, y); 7.8 pt title 0.17 in to its right on the same top; 6 pt subtitle below."""
    ftext(fig, x, y, letter, size=T_TAG, fontweight="bold")
    if title:
        ftext(fig, x + 0.17, y, title, size=T_TTL)
    if subtitle:
        ftext(fig, x + 0.17, y + 0.165, subtitle, size=T_TXT)


def open_axes(ax, nbins=4, x=True, y=True):
    """Bars, lines, boxplots: left and bottom spines only."""
    from matplotlib.ticker import MaxNLocator
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(LW)
    ax.tick_params(length=2, width=LW, pad=1.5)
    if x and ax.get_xscale() == "linear":
        ax.xaxis.set_major_locator(MaxNLocator(nbins))
    if y and ax.get_yscale() == "linear":
        ax.yaxis.set_major_locator(MaxNLocator(nbins))


def box_axes(ax, ticks=True):
    """Maps, heatmaps, scatters: full 0.6 pt box."""
    for s in ax.spines.values():
        s.set_visible(True)
        s.set_linewidth(LW)
    if ticks:
        ax.tick_params(length=2, width=LW, pad=1.5)
    else:
        ax.set_xticks([])
        ax.set_yticks([])


def schematic_ax(fig, x, y, w, h):
    """Invisible axes whose data units are inches from its own top-left, for flow boxes and arrows."""
    ax = iax(fig, x, y, w, h)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    return ax


def flow_box(ax, x, y, w, h, title, lines=(), fill="white", ec="black", text_color=None, ls="-",
             rounding=0.03, title_size=T_AX, line_size=T_TXT, italic_lines=False):
    """Rounded box (inches, on a schematic_ax), 7 pt title over 6 pt lines, centred."""
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}", fc=fill, ec=ec,
                                lw=0.8, ls=ls, zorder=2))
    if text_color is None:
        text_color = "white" if fill == HILITE_DARK else "black"
    rows = [(title, title_size, "normal")] + [(s, line_size, "italic" if italic_lines else "normal") for s in lines]
    rows = [r for r in rows if r[0]]
    lh = [s * 1.22 / 72 for _t, s, _st in rows]
    y0 = y + h / 2 - sum(lh) / 2
    for (t, s, st), d in zip(rows, lh):
        ax.text(x + w / 2, y0 + d / 2, t, fontsize=s, style=st, ha="center", va="center", color=text_color, zorder=3)
        y0 += d


def flow_arrow(ax, p0, p1, ls="-", color="black"):
    from matplotlib.patches import FancyArrowPatch
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=6, lw=0.8, color=color, ls=ls,
                                 shrinkA=0, shrinkB=0, zorder=1))


def marker_handle(color, label, marker="o", ms=3, mfc=None, mec="none", mew=0.6):
    from matplotlib.lines import Line2D
    return Line2D([], [], ls="", marker=marker, ms=ms, mfc=color if mfc is None else mfc, mec=mec, mew=mew,
                  color=color, label=label)


def line_handle(color, label, ls="-", lw=0.8, marker=None, ms=2.5):
    from matplotlib.lines import Line2D
    return Line2D([], [], color=color, ls=ls, lw=lw, marker=marker, ms=ms, mec="none", label=label)


def patch_handle(color, label, ec="none", hatch=None, lw=0.6):
    from matplotlib.patches import Patch
    return Patch(fc=color, ec=ec, hatch=hatch, lw=lw, label=label)


def legend_strip(fig, groups, x, y, gap=0.3):
    """One line of legends at (x, y) inches from the top-left (top of the strip). `groups` is a list of handle
    lists; each becomes a frameless one-row legend chained 0.3 in after the previous one. Returns the x where the
    strip ends (inches)."""
    fw, fh = fig.get_size_inches()
    for handles in groups:
        patchy = any(type(h).__name__ == "Patch" for h in handles)
        leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(x / fw, 1 - y / fh), ncol=len(handles),
                         frameon=False, fontsize=T_TXT, borderaxespad=0, borderpad=0, columnspacing=1.0,
                         handlelength=1.4 if patchy else 1.6, handleheight=0.7, handletextpad=0.4)
        fig.canvas.draw()                      # the legend's text extents exist only after a draw
        x = leg.get_window_extent(fig.canvas.get_renderer()).x1 / fig.dpi + gap
    return x - gap


def footnote(fig, s, y_from_bottom=0.03):
    fw, fh = fig.get_size_inches()
    fig.text((fw - 0.06) / fw, y_from_bottom / fh, s, fontsize=T_TXT, ha="right", va="bottom")


def highlight(ax, x, y, label=None, xytext=None, size=14, italic=False, **kw):
    """Open black circle on a point, optionally with a leader line to a label (xytext in axes fraction)."""
    ax.scatter([x], [y], s=size, facecolors="none", edgecolors="k", linewidths=0.6, zorder=6)
    if label is not None:
        ax.annotate(label, (x, y), xytext=xytext, textcoords="axes fraction", fontsize=T_TXT,
                    style="italic" if italic else "normal",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="k", shrinkA=1, shrinkB=2.5), zorder=6, **kw)


def colorbar(fig, x, y, cmap, vmin, vmax, ticks, label, h=0.6, w=0.07):
    """Riboseq inline colorbar: a 0.07 in strip at (x, y) inches, ticks on the right, label titled above."""
    import numpy as np
    ax = iax(fig, x, y, w, h)
    ax.imshow(np.linspace(vmax, vmin, 100)[:, None], aspect="auto", cmap=cmap, extent=(0, 1, vmin, vmax))
    ax.set_xticks([])
    ax.yaxis.tick_right()
    ax.set_yticks(ticks)
    ax.tick_params(length=2, width=LW, pad=1)
    for s in ax.spines.values():
        s.set_linewidth(LW)
    ax.set_title(label, fontsize=T_TXT, pad=3)
    return ax


def cell_check(fig):
    """Cell print rules: visible text 6-8 pt, strokes at least 0.5 pt. Prints and returns the problems."""
    import matplotlib
    bad = []
    for t in fig.findobj(matplotlib.text.Text):
        if t.get_visible() and t.get_text().strip() and not 6 - 1e-6 <= t.get_fontsize() <= 8 + 1e-6:
            bad.append(f"text {t.get_fontsize():.2f} pt: {t.get_text()[:40]!r}")
    for ln in fig.findobj(matplotlib.lines.Line2D):
        if ln.get_visible() and ln.get_linestyle() not in ("None", "") and len(ln.get_xdata()) > 1 \
                and 0 < ln.get_linewidth() < 0.5 - 1e-6:
            bad.append(f"line {ln.get_linewidth():.2f} pt")
    for pa in fig.findobj(matplotlib.patches.Patch):
        ec = pa.get_edgecolor()
        if pa.get_visible() and ec[3] > 0 and 0 < pa.get_linewidth() < 0.5 - 1e-6:
            bad.append(f"stroke {pa.get_linewidth():.2f} pt on {type(pa).__name__}")
    bad = sorted(set(bad))
    for b in bad:
        print("CELL:", b)
    print(f"cell_check: {len(bad)} problem(s)")
    return bad


def save_paper(fig, stem):
    """cell_check, then PDF (fonts embedded), LZW RGB TIFF and a 300-dpi PNG, as the riboseq figures do."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from PIL import Image
    cell_check(fig)
    with mpl.rc_context({"savefig.dpi": 300}):
        cell_save(fig, f"{stem}.pdf")
    cell_save(fig, f"{stem}.tif")
    im = Image.open(f"{stem}.tif")
    dpi = im.info.get("dpi", (300, 300))
    rgb = Image.new("RGB", im.size, "white")
    rgb.paste(im, mask=im.split()[3] if im.mode == "RGBA" else None)
    rgb.save(f"{stem}.tif", compression="tiff_lzw", dpi=dpi)
    fig.savefig(f"{stem}.png", dpi=300)
    w, h = fig.get_size_inches()
    print(f"wrote {stem}.pdf / .tif / .png ({w * 2.54:.1f} x {h * 2.54:.1f} cm)")
    plt.close(fig)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------- PyMOL-style sphere rendering
def _rgb(c):
    from matplotlib.colors import to_rgb
    import numpy as np
    return np.array(to_rgb(c), dtype="float64")


def project3d(U, view=(20, -58), persp=3.5):
    """Rotate (n, 3) points by view = (elevation, azimuth) degrees and project in perspective.
    Returns screen x, screen y, depth (larger = farther) and the perspective factor."""
    import numpy as np
    e, a = np.deg2rad(view[0]), np.deg2rad(view[1])
    x1 = U[:, 0] * np.cos(a) - U[:, 1] * np.sin(a)
    y1 = U[:, 0] * np.sin(a) + U[:, 1] * np.cos(a)
    z1 = U[:, 2]
    sy = z1 * np.cos(e) - y1 * np.sin(e)
    depth = y1 * np.cos(e) + z1 * np.sin(e)
    f = persp / (persp + depth)
    return x1 * f, sy * f, depth, f


def draw_box(ax, view, U, persp=10.0):
    """The box around the 3D points U shown, in perspective (project3d): front edges solid, back edges light and
    dashed. Returns the projected corner x, y (to include in the axis limits). As drawn in Figures 1-3."""
    import numpy as np
    lo, hi = U.min(axis=0), U.max(axis=0)
    C = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])], float)
    cx, cy, cd, _f = project3d(C, view, persp)
    far = int(np.argmax(cd))
    for i in range(8):
        for j in range(i + 1, 8):
            if np.sum(~np.isclose(C[i], C[j])) != 1:
                continue
            back = far in (i, j)
            ax.plot([cx[i], cx[j]], [cy[i], cy[j]], color="#b0b0b0" if back else "#6e6e6e", lw=0.5,
                    ls=DASH if back else "-", zorder=0.5)
    return cx, cy


def best_view(U, idx, persp=3.5, elevs=(10, 20, 30, 40), azims=range(0, 360, 10)):
    """The (elevation, azimuth) whose projection spreads the rows `idx` most on screen (largest mean pairwise
    distance), so highlighted points are not drawn stacked on one another."""
    import numpy as np
    idx = np.asarray(idx)
    best, arg = -1.0, (20, -58)
    for e in elevs:
        for a in azims:
            sx, sy, _d, _f = project3d(U[idx], (e, a), persp)
            P = np.c_[sx, sy]
            score = float(np.mean(np.linalg.norm(P[:, None] - P[None], axis=-1)))
            if score > best:
                best, arg = score, (e, a)
    return arg


def render_spheres(ax, X, colors, radius_pt=1.0, alpha=1.0, view=(20, -58), persp=3.5, fog=0.55, dpi=600,
                   center=None, half=None, pad=0.06, light=(-0.5, 0.6, 0.62), rim=0.22, order=None, layers=None,
                   sizes=None, bg="white"):
    """Draw points as ray-traced-looking spheres (PyMOL style) into an image on `ax`, which is turned off.

    X: (n, 2) or (n, 3) coordinates; 3D is rotated by `view` = (elevation, azimuth) in degrees and drawn in
    perspective with depth fog. Each sphere is Blinn-Phong shaded (ambient + diffuse + specular, light from the
    upper left and front) with a darkened rim, anti-aliased, and composited far-to-near (painter's algorithm).
    colors: one colour or n colours; alpha: scalar or n values (0-1); sizes: n radius multipliers.
    `layers` (n ints) draws higher layers after
    lower ones regardless of depth, so highlighted points are never hidden. `center` / `half`: the data window
    (default: the data's bounding box), mapped to the axes with equal aspect. 2D input: draw order is `order`
    (default: as given). Returns a dict with each point's pixel x, y, radius (pixels) and the canvas dpi, for
    vector overlays in the same coordinates (ax data units = pixels, y down)."""
    import numpy as np
    X = np.asarray(X, dtype="float64")
    n = len(X)
    three = X.shape[1] == 3
    single = isinstance(colors, str) or (np.ndim(colors) == 1 and len(colors) in (3, 4) and np.isscalar(colors[0]))
    cols = np.tile(_rgb(colors), (n, 1)) if single else np.array([_rgb(c) for c in colors])
    al = np.broadcast_to(np.asarray(alpha, dtype="float64"), (n,)).copy()
    lay = np.zeros(n) if layers is None else np.asarray(layers, dtype="float64")
    if center is None:
        center = (X.max(0) + X.min(0)) / 2
    if half is None:
        half = float(np.max(X.max(0) - X.min(0)) / 2)
    U = (X - center) / half                          # window -> [-1, 1]
    if three:
        sx, sy, depth, f = project3d(U, view, persp)
        lim = float(np.max(np.abs(np.r_[sx, sy]))) if n else 1.0
    else:
        sx, sy, depth, f = U[:, 0], U[:, 1], np.zeros(n), np.ones(n)
        lim = 1.0
    fig = ax.figure
    bb = ax.get_position()
    fw, fh = fig.get_size_inches()
    Wp, Hp = int(round(bb.width * fw * dpi)), int(round(bb.height * fh * dpi))
    r0 = radius_pt / 72 * dpi
    scale = (min(Wp, Hp) / 2 - r0 * 1.5) / (lim * (1 + pad))
    px = Wp / 2 + sx * scale
    py = Hp / 2 - sy * scale
    rr = r0 * f * (1.0 if sizes is None else np.asarray(sizes, dtype="float64"))
    dn = (depth - depth.min()) / (np.ptp(depth) or 1.0)          # 0 near .. 1 far
    bgc = _rgb(bg)
    canvas = np.ones((Hp, Wp, 3)) * bgc
    L = np.asarray(light, dtype="float64")
    L = L / np.linalg.norm(L)
    Hv = L + np.array([0, 0, 1.0])
    Hv = Hv / np.linalg.norm(Hv)
    if order is None:
        order = np.lexsort((-depth, lay)) if three else np.lexsort((np.arange(n), lay))
    for i in order:
        r = rr[i]
        x0, x1_ = int(np.floor(px[i] - r - 1)), int(np.ceil(px[i] + r + 1))
        y0, y1_ = int(np.floor(py[i] - r - 1)), int(np.ceil(py[i] + r + 1))
        if x1_ < 0 or y1_ < 0 or x0 >= Wp or y0 >= Hp:
            continue
        xs = np.arange(max(x0, 0), min(x1_, Wp)) + 0.5
        ys = np.arange(max(y0, 0), min(y1_, Hp)) + 0.5
        DX = (xs[None, :] - px[i]) / r
        DY = (ys[:, None] - py[i]) / r
        d2 = DX ** 2 + DY ** 2
        d = np.sqrt(d2)
        cov = np.clip((1 - d) * r + 0.5, 0, 1) * al[i]            # anti-aliased coverage
        if not cov.any():
            continue
        nz = np.sqrt(np.clip(1 - d2, 0, 1))
        ndl = np.clip(DX * L[0] - DY * L[1] + nz * L[2], 0, 1)    # image y runs down
        ndh = np.clip(DX * Hv[0] - DY * Hv[1] + nz * Hv[2], 0, 1)
        shade = 0.36 + 0.70 * ndl
        spec = 0.55 * ndh ** 40
        rimk = np.clip((d - (1 - rim)) / rim, 0, 1)
        c = cols[i][None, None, :] * shade[..., None] + spec[..., None]
        c = c * (1 - 0.55 * rimk[..., None])
        if three and fog:
            t = fog * dn[i]
            c = c * (1 - t) + t * bgc
        c = np.clip(c, 0, 1)
        sl = canvas[max(y0, 0):min(y1_, Hp), max(x0, 0):min(x1_, Wp)]
        sl[:] = sl * (1 - cov[..., None]) + c * cov[..., None]
    ax.imshow(canvas, extent=(0, Wp, Hp, 0), interpolation="antialiased", zorder=1)
    ax.set_xlim(0, Wp)
    ax.set_ylim(Hp, 0)
    ax.set_aspect("auto")
    ax.axis("off")
    def to_px(Y):
        """Data coordinates -> this image's pixel coordinates (for overlays such as ellipses)."""
        V = (np.atleast_2d(np.asarray(Y, dtype="float64")) - center) / half
        if three:
            qx, qy, _dq, _fq = project3d(V, view, persp)
        else:
            qx, qy = V[:, 0], V[:, 1]
        return Wp / 2 + qx * scale, Hp / 2 - qy * scale

    return {"x": px, "y": py, "r": rr, "dpi": dpi, "to_px": to_px}


def enclosing_ellipse(ax, R, Y, pad=1.03, fc=HILITE, ec=HILITE_DARK, alpha=0.28, lw=0.8, zorder=4):
    """Covariance ellipse (2D) or ellipsoid (3D, drawn as its projected outline) of the points Y, scaled so every
    point of Y is inside (largest Mahalanobis distance x pad), on an axes drawn by render_spheres (R = its return
    value). Returns inside(Z): which rows of Z lie within it (in the data space, so 3D is tested in 3D)."""
    import numpy as np
    from matplotlib.patches import Polygon
    from scipy.spatial import ConvexHull
    Y = np.asarray(Y, dtype="float64")
    d = Y.shape[1]
    mu = Y.mean(0)
    S = np.cov(Y.T) + 1e-9 * np.eye(d)
    Li = np.linalg.inv(np.linalg.cholesky(S))
    m = np.sqrt((((Y - mu) @ Li.T) ** 2).sum(1))
    s = float(m.max() * pad)
    Lc = np.linalg.cholesky(S)
    if d == 2:
        t = np.linspace(0, 2 * np.pi, 241)
        surf = mu + s * (np.c_[np.cos(t), np.sin(t)] @ Lc.T)
        qx, qy = R["to_px"](surf)
        poly = np.c_[qx, qy]
    else:
        u, v = np.meshgrid(np.linspace(0, 2 * np.pi, 72), np.linspace(0, np.pi, 36))
        sph = np.c_[(np.cos(u) * np.sin(v)).ravel(), (np.sin(u) * np.sin(v)).ravel(), np.cos(v).ravel()]
        qx, qy = R["to_px"](mu + s * (sph @ Lc.T))
        pts = np.c_[qx, qy]
        poly = pts[ConvexHull(pts).vertices]
    ax.add_patch(Polygon(poly, closed=True, fc=fc, ec="none", alpha=alpha, zorder=zorder))
    ax.add_patch(Polygon(poly, closed=True, fill=False, ec=ec, lw=lw, zorder=zorder + 0.1))

    def inside(Z):
        Z = np.atleast_2d(np.asarray(Z, dtype="float64"))
        return np.sqrt((((Z - mu) @ Li.T) ** 2).sum(1)) <= s
    return inside


def ring_overlay(ax, R, idx, color="black", lw=0.6, grow=1.7):
    """Vector rings (open circles) around the spheres `idx` drawn by render_spheres (R = its return value)."""
    import numpy as np
    from matplotlib.patches import Circle
    for i in np.atleast_1d(idx):
        ax.add_patch(Circle((R["x"][i], R["y"][i]), R["r"][i] * grow, fill=False, ec=color, lw=lw, zorder=5))


__all__ = ["apply_style", "save", "cell_tag", "cell_lw", "style_axes",
           "CMAP_DIVERGING", "PITCH_COLORS", "PITCH_NAMES", "PRINT_W",
           "paper_style", "canvas", "iax", "ftext", "tag", "open_axes", "box_axes", "schematic_ax", "flow_box",
           "flow_arrow", "marker_handle", "line_handle", "patch_handle", "legend_strip", "footnote", "highlight",
           "colorbar", "cell_check", "save_paper", "md_table", "render_spheres", "ring_overlay", "enclosing_ellipse",
           "project3d", "best_view", "PAPER_TARGETS",
           "W", "H_MAX", "T_TAG", "T_TTL", "T_AX", "T_TXT", "LW",
           "HILITE", "HILITE_DARK", "GREY", "LIGHT", "DASH", "DOTTED"]
