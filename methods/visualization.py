from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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

# Default colourmap used by all plotting functions; override via setup_matplotlib.
DEFAULT_CMAP = 'viridis'

# --- Matplotlib style setup -------------------------------------------------

def setup_matplotlib(
    font_size: float = DEFAULT_FONT_SIZE,
    figure_width_mm: float = LATEX_DOUBLE_COLUMN_WIDTH_MM,
    aspect_ratio: float = DEFAULT_ASPECT_RATIO,
    cmap: str = DEFAULT_CMAP,
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
    cmap
        Default colourmap stored in ``DEFAULT_CMAP`` and used by all
        plotting functions in this module.
    """
    global DEFAULT_CMAP
    DEFAULT_CMAP = cmap

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


# --- ODS animation ----------------------------------------------------------

def animate_ods(ref_3d, ods_3d, n_frames=40, title=None, cmap=None):
    """
    Animate one period of a 3-D Operating Deflection Shape.

    Point positions are fixed at their reference XY locations; out-of-plane (Z)
    displacement is encoded by colour.

    Parameters
    ----------
    ref_3d : ndarray of shape (n_pts, 3)
        Reference 3-D positions (Y, X, Z) in world units.
    ods_3d : ndarray of shape (n_pts, 3)
        Complex 3-D ODS vector at a single frequency.
    n_frames : int
        Frames per period.
    title : str, optional
    cmap : str, optional
        Colourmap; defaults to ``DEFAULT_CMAP``.

    Returns
    -------
    anim : FuncAnimation
    """
    cmap = cmap if cmap is not None else DEFAULT_CMAP
    phases = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    vmax = np.abs(ods_3d[:, 2]).max() or 1.0

    fig, ax = plt.subplots()
    z0 = np.real(ods_3d[:, 2] * np.exp(1j * phases[0]))
    sc = ax.scatter(ref_3d[:, 1], ref_3d[:, 0],
                    c=z0, cmap=cmap, vmin=-vmax, vmax=vmax, s=30)
    plt.colorbar(sc, ax=ax, label='Z [mm]')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    if title:
        ax.set_title(title)

    def _update(frame):
        sc.set_array(np.real(ods_3d[:, 2] * np.exp(1j * phases[frame])))
        return (sc,)

    anim = FuncAnimation(fig, _update, frames=n_frames, blit=True, interval=50)
    plt.close(fig)
    return anim


def animate_ods_3d(ref_3d, ods_3d, scale=1.0, n_frames=40, cmap=None):
    """
    Render one period of a 3-D ODS as an inline animated GIF.

    Uses pyvista off-screen rendering — no Qt window, safe inside Jupyter.
    Points are triangulated into an interpolated surface; colour encodes
    displacement magnitude.

    Parameters
    ----------
    ref_3d : ndarray of shape (n_pts, 3)
        Reference positions (Y, X, Z) in world units.
    ods_3d : ndarray of shape (n_pts, 3)
        Complex ODS vector at a single frequency.
    scale : float
        Visual amplification of 3-D displacement.
    n_frames : int
        Frames per period.
    cmap : str, optional
        Colourmap; defaults to ``DEFAULT_CMAP``.

    Returns
    -------
    IPython.display.HTML
        Animated GIF embedded inline in the notebook.
    """
    import base64
    import io
    import imageio.v3 as iio
    import pyvista as pv
    from IPython.display import HTML

    cmap = cmap if cmap is not None else DEFAULT_CMAP

    # ref_3d convention (Y, X, Z) → pyvista (X, Y, Z)
    ref_xyz = ref_3d[:, [1, 0, 2]].astype(float)
    ods_xyz = ods_3d[:, [1, 0, 2]]

    phases = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    # (n_frames, n_pts, 3) — absolute positions each frame
    history = ref_xyz[np.newaxis] + scale * np.real(
        ods_xyz[np.newaxis] * np.exp(1j * phases[:, np.newaxis, np.newaxis])
    )
    norms = np.linalg.norm(history - ref_xyz[np.newaxis], axis=2)  # (n_frames, n_pts)
    vmax = norms.max() or 1.0

    cloud = pv.PolyData(history[0])
    cloud['disp'] = norms[0]
    surf = cloud.delaunay_2d()

    pl = pv.Plotter(off_screen=True, window_size=(800, 600))
    pl.add_mesh(surf, scalars='disp', cmap=cmap, clim=(0, vmax),
                scalar_bar_args={'title': '|disp| [mm]', 'vertical': True})
    pl.render()

    frames = []
    for i in range(n_frames):
        surf.points = history[i]
        surf['disp'] = norms[i]
        pl.render()
        frames.append(pl.screenshot(return_img=True)[:, :, :3])

    pl.close()

    buf = io.BytesIO()
    iio.imwrite(buf, np.stack(frames), extension='.gif', loop=0, duration=50)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return HTML(f'<img src="data:image/gif;base64,{b64}" style="max-width:100%"/>')
