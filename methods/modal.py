"""
Experimental modal analysis using image-based measurements (paper Section 5).

Wraps sdypy-EMA for modal parameter identification and provides FRF estimation
and mode shape validation utilities.
"""
import numpy as np
import matplotlib.pyplot as plt
from methods.utils import load_hdf5

# --- FRF estimation ---------------------------------------------------------

def load_signals(sig_files, force_ch=0, response_ch=1):
    """
    Load force and response signals from a list of HDF5 measurement files.

    Parameters
    ----------
    sig_files : list of str
        Paths to HDF5 files, each containing an ``InputTask`` dataset with
        shape ``(n_samples, n_channels)``.
    force_ch : int
        Channel index for the force signal (default 0).
    response_ch : int
        Channel index for the response signal (default 1).

    Returns
    -------
    force : ndarray of shape (n_reps, n_samples)
    response : ndarray of shape (n_reps, n_samples)
    """
    data = [load_hdf5(f)['InputTask']['data'] for f in sig_files]
    force    = np.array([d[:, force_ch]    for d in data])
    response = np.array([d[:, response_ch] for d in data])
    return force, response

def compute_frf(force, response, fs=1.0):
    """
    Ensemble-averaged H₁, H₂ and coherence from one or more measurement repetitions.

    Parameters
    ----------
    force : ndarray of shape (n_samples,) or (n_reps, n_samples)
        Input (force) time series, mean-removed before DFT.
    response : ndarray matching force in shape
        Output (response) time series, mean-removed before DFT.
    fs : float
        Sampling frequency in Hz (default 1.0).

    Returns
    -------
    H1 : ndarray of shape (n_freq,)
        H₁ estimator — minimises output-noise bias.
    H2 : ndarray of shape (n_freq,)
        H₂ estimator — minimises input-noise bias.
    coherence : ndarray of shape (n_freq,)
    freq : ndarray of shape (n_freq,)
        Frequency axis in Hz.
    """
    force    = np.atleast_2d(force)    # (n_reps, n_samples)
    response = np.atleast_2d(response)

    Sff_list, Sfr_list, Srr_list, Srf_list = [], [], [], []
    for f, r in zip(force, response):
        F = np.fft.rfft(f - f.mean())
        R = np.fft.rfft(r - r.mean())
        Sff_list.append(np.abs(F) ** 2)
        Sfr_list.append(np.conj(F) * R)
        Srr_list.append(np.abs(R) ** 2)
        Srf_list.append(np.conj(R) * F)

    Sff = np.mean(Sff_list, axis=0)
    Sfr = np.mean(Sfr_list, axis=0)
    Srr = np.mean(Srr_list, axis=0)
    Srf = np.mean(Srf_list, axis=0)

    H1        = Sfr / Sff
    H2        = Srr / Srf
    coherence = H1 / H2
    freq      = np.fft.rfftfreq(force.shape[1], d=1.0 / fs)
    return H1, H2, coherence, freq


# --- Visualisation ----------------------------------------------------------

def plot_mode_shape(mode_shape, grid_coords, title=None, ax=None, cmap=None):
    """
    Plot a mode shape amplitude on the 2-D measurement grid using smooth
    tricontourf interpolation.

    Parameters
    ----------
    mode_shape : ndarray of shape (n_pts,) or (n_pts, n_directions)
        Complex or real mode shape vector.  If 2-D, the Euclidean norm across
        directions is plotted; if 1-D, the absolute value is plotted.
    grid_coords : ndarray of shape (n_pts, 3)
        World-frame grid coordinates ``(Y, X, Z)`` as returned by
        ``grid_coordinates_3d``; Y is the row direction, X the column direction.
    title : str, optional
        Axes title.
    ax : matplotlib Axes, optional
        Axes to plot into; a new figure is created if not given.
    cmap : str, optional
        Colourmap; defaults to ``'viridis'``.

    Returns
    -------
    ax : Axes
    cf : TriContourSet
        The contour artist (use for colourbar).
    """
    import matplotlib.tri as mtri

    if ax is None:
        _, ax = plt.subplots()
    amplitude = (np.linalg.norm(np.abs(mode_shape), axis=1)
                 if mode_shape.ndim == 2 else np.abs(mode_shape))
    cmap = cmap or 'viridis'
    triang = mtri.Triangulation(grid_coords[:, 1], grid_coords[:, 0])
    levels = np.linspace(0, amplitude.max() or 1.0, 51)
    cf = ax.tricontourf(triang, amplitude, levels=levels, cmap=cmap)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    if title:
        ax.set_title(title)
    return ax, cf


def unstack_shape(phi, mode=0):
    """
    Split one mode of a direction-stacked mode-shape matrix into per-point components.

    Parameters
    ----------
    phi : ndarray of shape (2*n_pts, n_modes)
        Mode shapes with the two measured image directions stacked vertically:
        first ``n_pts`` rows the v-direction, last ``n_pts`` rows the u-direction.
    mode : int, optional
        Column (mode) index.

    Returns
    -------
    ndarray of shape (n_pts, 2)
        Ready for ``plot_mode_shape``.
    """
    n_pts = phi.shape[0] // 2
    return np.stack([phi[:n_pts, mode], phi[n_pts:, mode]], axis=1)


def plot_mode_shapes(shapes, grid_coords, titles=None, cmap=None,
                     colorbar_label=None, panel_size=(3.5, 3)):
    """
    Plot several mode shapes side by side on the measurement grid.

    Parameters
    ----------
    shapes : sequence of ndarray, each of shape (n_pts, 2)
        Mode shapes to plot, e.g. from ``unstack_shape``.
    grid_coords : ndarray of shape (n_pts, 3)
        World-frame grid coordinates, as for ``plot_mode_shape``.
    titles : sequence of str, optional
        Per-panel titles.
    cmap : str, optional
    colorbar_label : str, optional
        If given, a colourbar with this label is added to each panel.
    panel_size : tuple, optional
        Size of a single panel in inches.

    Returns
    -------
    fig : Figure
    axes : ndarray of Axes
    """
    n = len(shapes)
    fig, axes = plt.subplots(1, n, figsize=(panel_size[0] * n, panel_size[1]))
    axes = np.atleast_1d(axes)
    for ax, shape, title in zip(axes, shapes, titles if titles is not None else [None] * n):
        _, cf = plot_mode_shape(shape, grid_coords, ax=ax, title=title, cmap=cmap)
        if colorbar_label is not None:
            fig.colorbar(cf, ax=ax, label=colorbar_label)
    return fig, axes


def print_comparison_table(rows, headers, floatfmt, colalign=None, missing='--'):
    """
    Print a modal-parameter comparison table.

    Every value is formatted to a string before tabulate sees it: tabulate
    silently ignores ``floatfmt`` for a whole column as soon as that column
    holds one non-numeric value (a missing-value placeholder, for instance),
    so numeric re-parsing is disabled here as well.

    Parameters
    ----------
    rows : sequence of sequences
        Table values; ``None`` marks a value that is not available.
    headers : sequence of str
    floatfmt : sequence of str
        Per-column format spec (e.g. ``'.2f'``); an empty spec leaves the
        value unformatted.
    colalign : sequence of str, optional
        Per-column alignment, passed to tabulate.
    missing : str, optional
        Placeholder printed for ``None`` values.
    """
    from tabulate import tabulate

    formatted = [
        [missing if value is None
         else format(value, fmt) if fmt and not isinstance(value, str)
         else str(value)
         for value, fmt in zip(row, floatfmt)]
        for row in rows
    ]
    print(tabulate(formatted, headers=headers, colalign=colalign, disable_numparse=True))


def compute_mac(phi_a, phi_b):
    """
    Cross-MAC matrix between two independently obtained mode-shape sets,
    defined over the same degrees of freedom (e.g. the same measurement points).

    Parameters
    ----------
    phi_a : ndarray of shape (n_dof, n_modes_a)
    phi_b : ndarray of shape (n_dof, n_modes_b)
        Complex or real mode shape vectors, both over the same ``n_dof``.

    Returns
    -------
    mac : ndarray of shape (n_modes_a, n_modes_b)
    """
    phi_a = np.asarray(phi_a)
    phi_b = np.asarray(phi_b)
    num = np.abs(phi_a.conj().T @ phi_b) ** 2
    norm_a = np.sum(np.abs(phi_a) ** 2, axis=0)
    norm_b = np.sum(np.abs(phi_b) ** 2, axis=0)
    return num / np.outer(norm_a, norm_b)


def plot_mac(mac, ax=None, cmap=None):
    """
    Colour-matrix plot of a MAC matrix with annotated values.

    Parameters
    ----------
    mac : ndarray of shape (n_modes_a, n_modes_b)
        Square for an auto-MAC, rectangular for a cross-MAC (e.g. from
        ``compute_mac``).
    ax : matplotlib Axes, optional
    cmap : str, optional
        Colourmap; defaults to ``'viridis'``.

    Returns
    -------
    ax : Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    mac = np.abs(mac)
    im = ax.imshow(mac, vmin=0, vmax=1, cmap=cmap or 'viridis')
    ax.set_xlabel('Mode')
    ax.set_ylabel('Mode')
    xticks = np.arange(mac.shape[1])
    yticks = np.arange(mac.shape[0])
    ax.set_xticks(xticks); ax.set_xticklabels(xticks + 1)
    ax.set_yticks(yticks); ax.set_yticklabels(yticks + 1)
    for i in range(mac.shape[0]):
        for j in range(mac.shape[1]):
            ax.text(j, i, f'{mac[i, j]:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if mac[i, j] < 0.5 else 'black')
    return ax
