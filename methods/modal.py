"""
Experimental modal analysis using image-based measurements (paper Section 5).

Wraps sdypy-EMA for modal parameter identification and provides FRF estimation
and mode shape validation utilities.
"""
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt


# --- FRF estimation ---------------------------------------------------------

def compute_frf(sig_files, idi_files, n_freq, view='view 0'):
    """
    Ensemble-averaged H₁, H₂, coherence and accelerometer FRF from repeated
    sine-sweep measurements.

    For each repetition the full-record DFT is computed and the spectral
    quantities are accumulated; averaging across repetitions yields consistent
    H₁ and H₂ estimators.  The force spectrum is clipped to *n_freq* bins so
    that it matches the camera Nyquist frequency.

    Parameters
    ----------
    sig_files : list of str
        Measured-signal pickle files (``InputTask`` format: channel 0 = force,
        channel 1 = accelerometer).
    idi_files : list of str
        IDI displacement pickle files; each file must contain a dict keyed by
        view label.
    n_freq : int
        Number of positive-frequency bins to retain (clips force/acc spectra to
        the camera Nyquist).
    view : str
        Dict key used to select displacement data from each IDI pickle file.

    Returns
    -------
    H1 : ndarray of shape (n_pts, n_freq, 2)
        Displacement FRF, minimises output-noise bias.
    H2 : ndarray of shape (n_pts, n_freq, 2)
        Displacement FRF, minimises input-noise bias.
    coherence : ndarray of shape (n_pts, n_freq, 2)
    H_acc : ndarray of shape (n_freq,)
        Accelerometer FRF (accelerance).
    """
    Sff_list, Sfd_list, Sdd_list, Sdf_list, Sfa_list = [], [], [], [], []

    for sig_path, idi_path in zip(sig_files, idi_files):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with open(sig_path, 'rb') as f:
                sig_data = np.array(pickle.load(f)['InputTask']['data'])
            with open(idi_path, 'rb') as f:
                disp = pickle.load(f)[view]

        force = sig_data[:, 0]
        acc   = sig_data[:, 1]

        F = np.fft.rfft(force - force.mean())[:n_freq]                      # (n_freq,)
        A = np.fft.rfft(acc   - acc.mean())[:n_freq]                        # (n_freq,)
        D = np.fft.rfft(disp  - disp.mean(axis=1, keepdims=True), axis=1)  # (n_pts, n_freq, 2)

        Sff_list.append(np.abs(F) ** 2)
        Sfd_list.append(np.conj(F)[np.newaxis, :, np.newaxis] * D)
        Sdd_list.append(np.abs(D) ** 2)
        Sdf_list.append(np.conj(D) * F[np.newaxis, :, np.newaxis])
        Sfa_list.append(np.conj(F) * A)

    Sff = np.mean(Sff_list, axis=0)
    Sfd = np.mean(Sfd_list, axis=0)
    Sdd = np.mean(Sdd_list, axis=0)
    Sdf = np.mean(Sdf_list, axis=0)
    Sfa = np.mean(Sfa_list, axis=0)

    H1        = Sfd / Sff[np.newaxis, :, np.newaxis]
    H2        = Sdd / Sdf
    coherence = H1 / H2
    H_acc     = Sfa / Sff

    return H1, H2, coherence, H_acc


# --- Visualisation ----------------------------------------------------------

def plot_mode_shape(mode_shape, grid_coords, title=None, ax=None, **scatter_kw):
    """
    Scatter plot of a mode shape amplitude on the 2-D measurement grid.

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
    **scatter_kw
        Passed to ``ax.scatter``.

    Returns
    -------
    ax : Axes
    sc : PathCollection
        The scatter artist (use for colourbar).
    """
    if ax is None:
        _, ax = plt.subplots()
    amplitude = (np.linalg.norm(np.abs(mode_shape), axis=1)
                 if mode_shape.ndim == 2 else np.abs(mode_shape))
    scatter_kw.setdefault('cmap', 'viridis')
    scatter_kw.setdefault('s', 40)
    sc = ax.scatter(grid_coords[:, 1], grid_coords[:, 0],
                    c=amplitude, **scatter_kw)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    if title:
        ax.set_title(title)
    return ax, sc


def plot_mac(mac, ax=None):
    """
    Colour-matrix plot of a MAC matrix with annotated values.

    Parameters
    ----------
    mac : ndarray of shape (n_modes, n_modes)
    ax : matplotlib Axes, optional

    Returns
    -------
    ax : Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    im = ax.imshow(mac, vmin=0, vmax=1, cmap='viridis')
    plt.colorbar(im, ax=ax, label='MAC')
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
