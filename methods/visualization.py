from __future__ import annotations

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure

"""
Plotting helpers used throughout the tutorial notebook.

Functions accept data in the forms produced by the other `methods` modules and
return matplotlib Figure/Axes objects so the caller retains full control.
"""

# --- Layout constants -------------------------------------------------------

# Column widths commonly used in LaTeX documents (mm).
LATEX_SINGLE_COLUMN_WIDTH_MM = 85.0
LATEX_DOUBLE_COLUMN_WIDTH_MM = 175.0

# Sensible default height/width ratio for scientific plots.
DEFAULT_ASPECT_RATIO = 0.5

# Default font size for publication-ready figures.
DEFAULT_FONT_SIZE = 9.5

# --- Matplotlib style setup -------------------------------------------------

def setup_matplotlib(
    font_size: float = DEFAULT_FONT_SIZE,
    figure_width_mm: float = LATEX_DOUBLE_COLUMN_WIDTH_MM,
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
) -> None:
    """Set publication-ready matplotlib defaults.

    Parameters
    ----------
    font_size
        Base font size applied consistently across labels/ticks/legend.
    figure_width_mm
        Figure width in mm.
    aspect_ratio
        Figure height = figure_width * aspect_ratio.
    """
    figure_width_in = figure_width_mm / 25.4
    figure_height_in = figure_width_in * aspect_ratio

    mpl.rcParams.update(
        {
            # Font: STIX Two — journal-quality serif, condensed and crisp, full math support
            "font.family": "serif",
            "font.serif": ["STIX Two Text", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "text.usetex": False,
            # Consistent typography
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.titlesize": font_size,
            # Geometry (matplotlib expects inches)
            "figure.figsize": (figure_width_in, figure_height_in),
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "figure.autolayout": True,
            # Predictable export
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
            # Publication-friendly defaults
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.linewidth": 0.6,
            "axes.formatter.use_mathtext": True,
            "lines.linewidth": 0.9,
            "lines.markersize": 4.0,
            "legend.frameon": False,
            "legend.handlelength": 1.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "errorbar.capsize": 2.0,
        }
    )
