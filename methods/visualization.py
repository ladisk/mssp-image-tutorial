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
LATEX_SINGLE_COLUMN_WIDTH_MM = 80.0
LATEX_DOUBLE_COLUMN_WIDTH_MM = 175.0

# Sensible default height/width ratio for scientific plots.
DEFAULT_ASPECT_RATIO = 0.5

# Default font size for publication-ready figures.
DEFAULT_FONT_SIZE = 12.0

# Default colourmap used by all plotting functions; override via setup_matplotlib.
DEFAULT_CMAP = 'viridis'

def figure_width_px():
    """Return the current matplotlib figure width in pixels."""
    return mpl.rcParams['figure.figsize'][0] * mpl.rcParams['figure.dpi']


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

_animate_ods_cache: dict = {}


def animate_ods(ref_3d, ods_3d, n_frames=30, title=None, cmap=None):
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
    import matplotlib.tri as mtri

    cmap = cmap if cmap is not None else DEFAULT_CMAP
    key = (hash(ref_3d.tobytes()), hash(ods_3d.tobytes()), n_frames, title, cmap)
    if key in _animate_ods_cache:
        return _animate_ods_cache[key]
    phases = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    vmax = np.abs(ods_3d[:, 2]).max() or 1.0

    triang = mtri.Triangulation(ref_3d[:, 1], ref_3d[:, 0])
    disps = np.real(
        ods_3d[:, 2][np.newaxis] * np.exp(1j * phases[:, np.newaxis])
    )  # (n_frames, n_pts)

    # Pre-interpolate onto a regular grid; frame updates become fast imshow
    # data swaps and blit=True works
    x_grid = np.linspace(ref_3d[:, 1].min(), ref_3d[:, 1].max(), 300)
    y_grid = np.linspace(ref_3d[:, 0].min(), ref_3d[:, 0].max(), 200)
    Xi, Yi = np.meshgrid(x_grid, y_grid)
    grids = np.stack([
        mtri.LinearTriInterpolator(triang, d)(Xi, Yi)
        for d in disps
    ])  # (n_frames, H, W) masked array

    # extent=[left, right, bottom, top] with bottom > top → Y increases downward
    extent = [x_grid[0], x_grid[-1], y_grid[-1], y_grid[0]]

    fig, ax = plt.subplots(figsize=mpl.rcParams['figure.figsize'])
    im = ax.imshow(grids[0], cmap=cmap, vmin=-vmax, vmax=vmax,
                   extent=extent, aspect='equal', interpolation='bilinear')
    plt.colorbar(im, ax=ax, label='Z [mm]')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    if title:
        ax.set_title(title)

    def _update(frame):
        im.set_data(grids[frame])
        return [im]

    anim = FuncAnimation(fig, _update, frames=n_frames, blit=True, interval=75)
    plt.close(fig)
    _animate_ods_cache[key] = anim
    return anim


def anim_to_html(anim):
    """Wrap a FuncAnimation as width-constrained inline HTML."""
    from IPython.display import HTML
    return HTML(
        f'<div style="max-width:{figure_width_px():.0f}px">'
        f'{anim.to_jshtml()}</div>')


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
    disp_per_frame = np.real(
        ods_xyz[np.newaxis] * np.exp(1j * phases[:, np.newaxis, np.newaxis])
    )  # (n_frames, n_pts, 3) — physical (unscaled) displacement
    history = ref_xyz[np.newaxis] + scale * disp_per_frame
    norms_physical = np.linalg.norm(disp_per_frame, axis=2)  # (n_frames, n_pts)
    vmax = norms_physical.max() or 1.0

    cloud = pv.PolyData(history[0])
    cloud['disp'] = norms_physical[0]
    surf = cloud.delaunay_2d()

    pl = pv.Plotter(off_screen=True, window_size=(800, 600))
    pl.add_mesh(surf, scalars='disp', cmap=cmap, clim=(0, vmax),
                scalar_bar_args={'title': '|disp| [mm]', 'vertical': True})
    pl.render()

    frames = []
    for i in range(n_frames):
        surf.points = history[i]
        surf['disp'] = norms_physical[i]
        pl.render()
        frames.append(pl.screenshot(return_img=True)[:, :, :3])

    pl.close()

    buf = io.BytesIO()
    iio.imwrite(buf, np.stack(frames), extension='.gif', loop=0, duration=50)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return HTML(f'<img src="data:image/gif;base64,{b64}" style="max-width:100%"/>')


def plot_ods_frame(ref_3d, ods_3d, phase, scale=1.0, cmap=None,
                   colorbar=True, vmax=None, window_size=(1600, 1200),
                   save=None):
    """
    Render a single 3-D ODS frame at a given phase angle using pyvista.

    The displaced surface is rendered off-screen; displacement magnitude
    encodes colour. Returns the RGB image as a numpy array.

    Parameters
    ----------
    ref_3d : ndarray of shape (n_pts, 3)
        Reference positions (Y, X, Z) in world units.
    ods_3d : ndarray of shape (n_pts, 3)
        Complex ODS vector at a single frequency.
    phase : float
        Phase angle in radians.
    scale : float
        Visual amplification of displacement.
    cmap : str, optional
        Colourmap; defaults to ``DEFAULT_CMAP``.
    colorbar : bool
        Whether to show the displacement colorbar (default ``True``).
    vmax : float, optional
        Upper limit of the colour scale. Computed from the ODS amplitude
        if not given; pass a shared value across modes for consistent scaling.
    window_size : tuple of int
        Render resolution in pixels (default ``(1600, 1200)``).
    save : str or Path, optional
        If given, save the rendered frame as a PNG to this path.

    Returns
    -------
    img : ndarray of shape (H, W, 3)
        Rendered RGB image.
    """
    import pyvista as pv

    if ods_3d.ndim != 2 or ods_3d.shape[1] != 3:
        raise ValueError(
            f"ods_3d must have shape (n_pts, 3); got {ods_3d.shape}. "
            "Select a single frequency first, e.g. ods_3d[:, :, i].")

    cmap = cmap if cmap is not None else DEFAULT_CMAP

    ref_xyz = ref_3d[:, [1, 0, 2]].astype(float)
    ods_xyz = ods_3d[:, [1, 0, 2]]

    disp_physical = np.real(ods_xyz * np.exp(1j * phase))
    pts = ref_xyz + scale * disp_physical
    norms = np.linalg.norm(disp_physical, axis=1)
    if vmax is None:
        vmax = np.sqrt((np.abs(ods_xyz) ** 2).sum(axis=1)).max() or 1.0

    cloud = pv.PolyData(pts)
    cloud['disp'] = norms
    surf = cloud.delaunay_2d()

    pl = pv.Plotter(off_screen=True, window_size=list(window_size))
    pl.add_mesh(surf, scalars='disp', cmap=cmap, clim=(0, vmax),
                scalar_bar_args={'title': '|disp| [mm]', 'vertical': True},
                show_scalar_bar=colorbar)
    pl.render()
    img = pl.screenshot(save, return_img=True)[:, :, :3]
    pl.close()

    return img


def plot_ods_grid(ref_3d, ods_3d, phases, scale=1.0, cmap=None,
                  titles=None, shared_scale=False, autoscale=False,
                  window_size=(800, 600), figscale=1.0, save=None):
    """
    Render multiple ODS modes side by side as a single matplotlib figure.

    Each panel is a pyvista off-screen render of one mode; panels are arranged
    in a single row.

    Parameters
    ----------
    ref_3d : ndarray of shape (n_pts, 3)
        Reference positions (Y, X, Z) in world units.
    ods_3d : ndarray of shape (n_pts, 3, n_modes)
        Complex ODS vectors, one per mode.
    phases : array-like of float
        Phase angle in radians for each mode.
    scale : float
        Visual amplification of displacement.
    cmap : str, optional
        Colourmap; defaults to ``DEFAULT_CMAP``.
    titles : list of str, optional
        Per-panel titles (e.g. frequency labels).
    shared_scale : bool
        If ``True``, use a single vmax across all modes so amplitudes are
        comparable. Default ``False`` (each mode normalised independently).
    window_size : tuple of int
        Per-panel pyvista render resolution (default ``(800, 600)``).
    figscale : float
        Uniform scale factor applied to the figure dimensions (default ``1.0``).
    save : str or Path, optional
        If given, save the figure as a PNG to this path.

    Returns
    -------
    fig, axes : Figure, list of Axes
    """
    n_modes = ods_3d.shape[2]
    phases = np.asarray(phases)

    if autoscale:
        max_amps = [np.sqrt((np.abs(ods_3d[:, :, i]) ** 2).sum(axis=1)).max() or 1.0
                    for i in range(n_modes)]
        scales = [scale / a for a in max_amps]
    else:
        scales = [scale] * n_modes

    vmax = None
    if shared_scale:
        vmax = max(
            np.sqrt((np.abs(ods_3d[:, :, i]) ** 2).sum(axis=1)).max()
            for i in range(n_modes)) or 1.0

    imgs = [
        plot_ods_frame(ref_3d, ods_3d[:, :, i], phases[i],
                       scale=scales[i], cmap=cmap, colorbar=False,
                       vmax=vmax, window_size=window_size)
        for i in range(n_modes)
    ]

    h, w = imgs[0].shape[:2]
    fig_w = LATEX_DOUBLE_COLUMN_WIDTH_MM / 25.4 * figscale
    fig, axes = plt.subplots(1, n_modes,
                             figsize=(fig_w, fig_w / n_modes * h / w))
    if n_modes == 1:
        axes = [axes]

    for ax, img, title in zip(axes, imgs, titles or [''] * n_modes):
        ax.imshow(img)
        ax.set_axis_off()
        if title:
            ax.set_title(title)

    fig.tight_layout(pad=0.2)
    if save:
        fig.savefig(save, dpi=300, bbox_inches='tight')
    return fig, axes
