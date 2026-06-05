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


def plot_mac(mac, ax=None, cmap=None):
    """
    Colour-matrix plot of a MAC matrix with annotated values.

    Parameters
    ----------
    mac : ndarray of shape (n_modes, n_modes)
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
    ticks = np.arange(mac.shape[0])
    ax.set_xticks(ticks); ax.set_xticklabels(ticks + 1)
    ax.set_yticks(ticks); ax.set_yticklabels(ticks + 1)
    for i in range(mac.shape[0]):
        for j in range(mac.shape[1]):
            ax.text(j, i, f'{mac[i, j]:.2f}', ha='center', va='center',
                    fontsize=7, color='white' if mac[i, j] < 0.5 else 'black')
    return ax
