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
