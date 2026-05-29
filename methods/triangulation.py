"""
3-D displacement reconstruction from multi-view camera data (paper Section 4).

Expects projection matrices produced by `calibration.py`.
"""
import numpy as np
from scipy.linalg import svd

# --- Core triangulation method ----------------------------------------------

def triangulate_dlt(x, P, K):
    """
    Triangulate a 3-D point from N ≥ 2 camera views using homogeneous DLT
    (Hartley & Zisserman §12.2).

    For each view the cross-product constraint x × PX = 0 contributes two
    linear equations; stacking all views yields an over-determined 2N × 4
    system AX = 0 whose least-squares solution is the last right singular
    vector of A.

    Parameters
    ----------
    x : ndarray of shape (N, 2)
        Image coordinates ``[u, v]`` (column, row) of the point in each of
        the N camera views.
    P : array-like of shape (N, 3, 4)
        ``[R | t]`` projection matrices, one per view, as returned by
        ``calibration.calibrate_extrinsic``; a list of (3, 4) arrays is also
        accepted.  The intrinsic matrix K is applied internally.
    K : ndarray of shape (3, 3)
        Camera intrinsic matrix, as returned by ``calibration.calibrate_camera``.

    Returns
    -------
    X : ndarray of shape (3,)
        Triangulated 3-D point in world coordinates.

    Raises
    ------
    ValueError
        If the number of point views does not match the number of cameras.
    """
    P = np.asarray(P)
    if x.shape[0] != P.shape[0]:
        raise ValueError('Number of point views does not match the number of cameras.')

    P_full = K @ P                                          # (N, 3, 4)

    rows_u = x[:, 0:1] * P_full[:, 2] - P_full[:, 0]        # (N, 4)
    rows_v = x[:, 1:2] * P_full[:, 2] - P_full[:, 1]        # (N, 4)
    A = np.stack([rows_u, rows_v], axis=1).reshape(-1, 4)   # (2N, 4)

    _, _, Vh = svd(A, full_matrices=False)
    Xh = Vh[-1]
    return (Xh / Xh[-1])[:3]


# --- Utilities --------------------------------------------------------------

def triangulate_points(x_per_view, P, K):
    """
    Triangulate M 3-D points observed in N ≥ 2 camera views.

    A thin vectorised wrapper around ``triangulate_multiview`` that processes
    each of the M points in turn.

    Input coordinates follow image convention (``[v, u]`` = ``[row, col]``),
    consistent with ``analysis_points`` and ``lk_displacements`` throughout the
    tutorial.  The flip to ``[u, v]`` required by the core DLT is handled
    internally.

    Parameters
    ----------
    x_per_view : ndarray of shape (N, M, 2)
        Image coordinates ``[v, u]`` (row, column) for each of the M points
        in each of the N camera views.
    P : array-like of shape (N, 3, 4)
        ``[R | t]`` projection matrices, one per view.
    K : ndarray of shape (3, 3)
        Camera intrinsic matrix.

    Returns
    -------
    points_3d : ndarray of shape (M, 3)
        Triangulated 3-D coordinates in world units.
    """
    return np.array([triangulate_dlt(x_per_view[:, i, ::-1], P, K)
                     for i in range(x_per_view.shape[1])])


def triangulate_ods(D_views, analysis_points, P, K, ref_3d, freq_indices):
    """
    Frequency-domain triangulation of Operating Deflection Shapes.

    For each requested frequency bin, complex displacement spectra are added to
    the reference pixel coordinates and the resulting complex image points are
    triangulated via DLT.  Subtracting the static reference gives complex 3-D
    ODS vectors; ``Re(ods * exp(j*phi))`` as phi varies over [0, 2π] traces one
    vibration period.

    Parameters
    ----------
    D_views : ndarray of shape (n_views, n_pts, n_freq, 2)
        Complex displacement spectra [v, u] for each view.
    analysis_points : ndarray of shape (n_views, n_pts, 2)
        Reference pixel positions [v, u] for each view.
    P : array-like of shape (n_views, 3, 4)
        Extrinsic projection matrices.
    K : ndarray of shape (3, 3)
        Camera intrinsic matrix.
    ref_3d : ndarray of shape (n_pts, 3)
        Triangulated 3-D reference positions.
    freq_indices : array-like of int
        Frequency bin indices to triangulate at.

    Returns
    -------
    ods_3d : ndarray of shape (n_pts, 3, n_freqs)
        Complex 3-D ODS at each requested frequency.
    """
    freq_indices = np.asarray(freq_indices)
    n_pts = D_views.shape[1]
    ods_3d = np.zeros((n_pts, 3, len(freq_indices)), dtype=complex)

    for fi, k in enumerate(freq_indices):
        # complex image coords = static reference + displacement at this frequency bin
        x_complex = analysis_points + D_views[:, :, k, :]  # (n_views, n_pts, 2)
        pts_3d = triangulate_points(x_complex, P, K)        # (n_pts, 3) complex
        ods_3d[:, :, fi] = pts_3d - ref_3d

    return ods_3d
