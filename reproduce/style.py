"""The look shared by the paper's figures: STIX serif, muted axes, one palette."""
INK, MUTED, GRID, PHOTO = "#0b0b0b", "#52514e", "#e6e5e0", "#898781"
COND, UNCOND, ACC = "#2a78d6", "#1baf7a", "#eb6834"
WCOL = {1.0: "#86b6ef", 4.5: "#3987e5", 8.0: "#1c5cab", 15.0: "#0d366b"}
SIGRAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]       # sigma: light = low, dark = high
W_COL = 3.3                                                             # one CVPR column, inches


def rc(size=7.5, small=6.5):
    """matplotlib with the paper's fonts; returns pyplot. PDFs embed TrueType (fonttype 42)."""
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                                "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": size, "axes.labelsize": size,
                                "xtick.labelsize": small, "ytick.labelsize": small, "legend.fontsize": small})
    import matplotlib.pyplot as plt
    return plt


def style(ax, grid=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
        ax.spines[s].set_linewidth(0.6)
    ax.tick_params(colors=MUTED, labelcolor=INK, width=0.6, length=2.5, which="both")
    if grid:
        ax.grid(True, color=GRID, lw=0.5, which="major")
    ax.set_axisbelow(True)


def title(ax, t):
    ax.text(0, 1.04, t, transform=ax.transAxes, fontsize=7.5, color=INK)


def sig_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("sig", SIGRAMP)
