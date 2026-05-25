"""
Camera calibration utilities (paper Section 2.2).

Outputs (projection matrices, distortion coefficients) feed into
`triangulation.py` and can also be used to undistort frames before
displacement identification.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from methods.utils import load_bw_image
from methods.visualization import LATEX_DOUBLE_COLUMN_WIDTH_MM

# --- Core calibration -------------------------------------------------------

def calibrate_camera(image_files, pattern_size, block_size):
    """
    Calibrate the camera using Zhang's method from checkerboard calibration images.

    Parameters
    ----------
    image_files : list of str 
        Paths to grayscale calibration images containing the checkerboard pattern.
    pattern_size : tuple of int
        Number of inner checkerboard corners in the `(columns, rows)` format.
    block_size : float or int
        Physical spacing between adjacent checkerboard corners in the chosen length unit.

    Returns
    -------
    K : ndarray of shape (3, 3)
        Estimated camera intrinsic matrix.
    calib : dict with keys:
        ``distCoeffs``   – lens distortion coefficients.
        ``perViewErrors``– per-image reprojection errors.
        ``valid_files``  – subset of ``image_files`` where corners were found.
        ``image_points`` – refined corners for each valid image.
        ``rvecs``        – rotation vectors (board-to-camera) per valid image.
        ``tvecs``        – translation vectors (board-to-camera) per valid image.
        ``block_size``   – physical corner spacing (passed through for downstream use).
    """
    # Load calibration images
    images = [load_bw_image(file) for file in image_files]
    
    # Set up corner detection criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points and image points lists
    object_points = []
    image_points = []

    # Prepare known object (checkerboard) points based on the pattern size and block size
    object_p = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
    object_p[:,:2] = np.mgrid[0:pattern_size[0]*block_size:block_size, 
                              0:pattern_size[1]*block_size:block_size].T.reshape(-1,2)

    # Find corners in each image and refine their positions
    valid_files = []
    for file, image in zip(image_files, images):
        found, corners = cv2.findChessboardCorners(image, pattern_size, None, 
                            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            refined_corners = cv2.cornerSubPix(image, corners, (5, 5), (-1, -1), criteria)
            if refined_corners[0, 0, 1] > refined_corners[-1, 0, 1]:
                refined_corners = refined_corners[::-1]
            image_points.append(refined_corners)
            object_points.append(object_p)
            valid_files.append(file)

    # Calibrate the camera intrinsic parameters based on detected corners and known object points
    ret, K, distCoeffs, rvecs, tvecs, st_i, st_e, perViewErrors = cv2.calibrateCameraExtended(
        objectPoints=object_points,
        imagePoints=image_points,
        imageSize=images[0].shape[::-1],
        cameraMatrix=np.zeros((3, 3), dtype=np.float32),
        distCoeffs=np.zeros((5, 1), dtype=np.float32)
    )

    calib = dict(distCoeffs=distCoeffs, perViewErrors=perViewErrors,
                 valid_files=valid_files, image_points=image_points,
                 rvecs=rvecs, tvecs=tvecs, block_size=block_size)
    return K, calib


# --- Extrinsic calibration --------------------------------------------------

def calibrate_extrinsic(object_points, image_points, K, calib):
    """
    Estimate the camera pose (extrinsic calibration) from known 3D–2D point
    correspondences using the Perspective-n-Point (PnP) algorithm.

    Given the 3D world coordinates of a set of reference points (e.g. ArUco
    marker centres) and their corresponding 2D pixel locations in the image,
    PnP recovers the rotation and translation that map the world frame into the
    camera frame.  The intrinsic matrix and distortion coefficients from
    ``calibrate_camera`` are required so that lens distortion is accounted for
    during the pose estimation.

    Parameters
    ----------
    object_points : ndarray of shape (N, 3)
        3D world coordinates of the reference points, in the same physical
        unit as ``block_size`` was during intrinsic calibration.
    image_points : ndarray of shape (N, 2)
        Corresponding 2D image coordinates of each reference point, in pixels.
    K : ndarray of shape (3, 3)
        Camera intrinsic matrix as returned by ``calibrate_camera``.
    calib : dict
        Calibration dict as returned by ``calibrate_camera``; distortion
        coefficients are read from ``calib['distCoeffs']``.

    Returns
    -------
    P : ndarray of shape (3, 4)
        Camera projection matrix ``[R | t]``.
    reprojection_error : float
        RMS reprojection error in pixels over all input point correspondences.
    """
    # Solve for the rotation vector and translation vector
    _, rvec, tvec = cv2.solvePnP(
        object_points.astype(np.float32),
        image_points.astype(np.float32),
        K, calib['distCoeffs'],
        flags=cv2.SOLVEPNP_ITERATIVE)

    # Convert the compact rotation vector to a 3×3 rotation matrix
    R = cv2.Rodrigues(rvec)[0]
    P = np.column_stack((R, tvec))

    # Project object points back into the image and compute RMS reprojection error
    projected, _ = cv2.projectPoints(
        object_points.astype(np.float32), rvec, tvec, K, calib['distCoeffs'])
    residuals = projected[:, 0, :] - image_points.astype(np.float32)
    reprojection_error = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))
    print(f"Extrinsic calibration reprojection error: {reprojection_error:.4f} px")

    return P, reprojection_error


def camera_position(P):
    """
    Convert a ``[R | t]`` camera projection matrix to the camera centre in
    world coordinates.

    The camera centre is the world point that projects to the origin of the
    image plane, i.e. the optical centre of the lens expressed in world
    coordinates.  Given the decomposition ``P = [R | t]``, the centre is
    ``C = -R^T t``.

    Parameters
    ----------
    P : ndarray of shape (3, 4)
        Camera projection matrix as returned by ``calibrate_extrinsic``.

    Returns
    -------
    position : ndarray of shape (3,)
        Camera centre in world coordinates.
    """
    R, t = P[:, :3], P[:, 3]
    return R.T @ -t

# --- Results summary and visualization --------------------------------------

# Distortion coefficient names follow the OpenCV convention (Brown-Conrady model).
_DIST_LABELS = ['k1', 'k2', 'p1', 'p2', 'k3', 'k4', 'k5', 'k6',
                's1', 's2', 's3', 's4', 'τx', 'τy']


def print_calibration_summary(K, calib, sensor_width_mm=None, sensor_width_px=None):
    """
    Print a concise summary of camera calibration results and plot per-view
    reprojection errors.

    Focal lengths are reported in pixels.  Calibration yields fx/fy in pixels
    because the projection model maps world coordinates (physical units) to image
    coordinates (pixels); the physical units cancel.  If ``sensor_width_mm`` and
    ``sensor_width_px`` are provided, focal lengths are also shown in mm using
    ``f_mm = fx * sensor_width_mm / sensor_width_px``.

    Parameters
    ----------
    K : ndarray of shape (3, 3)
        Camera intrinsic matrix as returned by ``calibrate_camera``.
    calib : dict
        Calibration dict as returned by ``calibrate_camera``.
    sensor_width_mm : float, optional
        Physical sensor width in mm.
    sensor_width_px : int, optional
        Image width in pixels at the resolution the sensor was operated at
        (may differ from calibration image width if the sensor was windowed).

    Returns
    -------
    fig : matplotlib.figure.Figure
        Per-view reprojection error chart.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    errors = calib['perViewErrors'].flatten()

    print("Intrinsic parameters")
    if sensor_width_mm is not None and sensor_width_px is not None:
        pixel_pitch = sensor_width_mm / sensor_width_px
        print(f"  Focal length : fx = {fx:.2f} px  ({fx * pixel_pitch:.3f} mm),  "
              f"fy = {fy:.2f} px  ({fy * pixel_pitch:.3f} mm)")
    else:
        print(f"  Focal length : fx = {fx:.2f} px,  fy = {fy:.2f} px")
    print(f"  Principal pt : cx = {cx:.2f} px,  cy = {cy:.2f} px")
    print(f"  Skew         : s  = {K[0, 1]:.5f}")

    print(f"Reprojection error  mean = {errors.mean():.4f} px,  max = {errors.max():.4f} px")

    # Per-view error chart
    views = np.arange(1, len(errors) + 1)

    fig, ax = plt.subplots()
    ax.vlines(views, 0, errors, lw=1.0, color='C0')
    ax.scatter(views, errors, s=18, color='C0', zorder=3)
    ax.axhline(errors.mean(), ls='--', lw=0.8, color='C1', label=f'mean = {errors.mean():.4f} px')
    ax.set_xlabel('View')
    ax.set_ylabel('Reprojection error (px)')
    ax.set_xlim(0.5, len(errors) + 0.5)
    ax.set_ylim(0)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend()
    fig.tight_layout()

    return fig


def plot_calibration_images(K, calib, n_images=5, show_axes=True):
    """
    Plot calibration images with detected checkerboard corners overlaid.

    Designed to consume the output of ``calibrate_camera`` directly.  When
    ``show_axes`` is ``True`` (the default), the board coordinate axes
    (x red, y green, z blue) are projected onto each image using the estimated
    poses, giving a visual check of the board orientation.

    Parameters
    ----------
    K : ndarray of shape (3, 3)
        Camera intrinsic matrix as returned by ``calibrate_camera``.
    calib : dict
        Calibration dict as returned by ``calibrate_camera``.
    n_images : int, optional
        Number of images to show (default 5).
    show_axes : bool, optional
        Draw the board coordinate axes overlay (default ``True``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list of matplotlib.axes.Axes
    """
    files = calib['valid_files'][:n_images]
    corners_list = calib['image_points'][:n_images]
    n = len(files)

    if show_axes:
        axis_len = calib['block_size'] * 2.5
        axis_pts = np.float32([[0, 0, 0],
                               [axis_len, 0, 0],
                               [0, axis_len, 0],
                               [0, 0, -axis_len]])

    fig_w = LATEX_DOUBLE_COLUMN_WIDTH_MM / 25.4
    sample = load_bw_image(files[0])
    panel_h = fig_w / n * (sample.shape[0] / sample.shape[1])
    fig, axes = plt.subplots(1, n, figsize=(fig_w, panel_h))
    if n == 1:
        axes = [axes]

    for i, (ax, f, corners) in enumerate(zip(axes, files, corners_list)):
        image = load_bw_image(f)
        ax.imshow(image, cmap='gray', vmin=0, vmax=255, interpolation='antialiased')
        ax.set_axis_off()

        pts = corners[:, 0, :]
        ax.scatter(pts[:, 0], pts[:, 1], s=3, c='cyan', linewidths=0, zorder=3)

        if show_axes:
            proj, _ = cv2.projectPoints(axis_pts, calib['rvecs'][i], calib['tvecs'][i],
                                        K, calib['distCoeffs'])
            proj = proj[:, 0, :]
            origin = proj[0]
            for tip, color in zip(proj[1:], ['#e03030', '#30a030', '#3060e0']):
                ax.annotate(
                    '', xy=tip, xytext=origin,
                    arrowprops=dict(arrowstyle='->', color=color,
                                    lw=1.5, mutation_scale=10),
                    zorder=4)

    fig.tight_layout(pad=0.1, w_pad=0.2)
    return fig, axes


def plot_scene(marker_positions, camera_matrices, K=None, frustum_size=50,
               marker_labels=None):
    """
    Plot the 3D measurement scene: reference marker positions and camera
    locations.

    Each camera is shown as a labelled point.  When ``K`` is provided, a
    frustum pyramid is drawn in front of each camera, giving a visual check
    of the viewing direction and approximate field of view.

    Parameters
    ----------
    marker_positions : ndarray of shape (N, 3)
        World coordinates of the reference markers (e.g. ArUco centres).
    camera_matrices : dict mapping str → ndarray of shape (3, 4)
        Camera projection matrices keyed by a descriptive label, as produced
        by ``calibrate_extrinsic``.
    K : ndarray of shape (3, 3), optional
        Camera intrinsic matrix; required to draw the frustum pyramid.
    frustum_size : float, optional
        Approximate depth of the frustum in world units (default 50).
    marker_labels : list of str, optional
        Labels for each marker; indices are used when omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax  : matplotlib.axes.Axes (3-D)
    """
    fig_w = LATEX_DOUBLE_COLUMN_WIDTH_MM / 25.4
    fig = plt.figure(figsize=(fig_w, fig_w * 0.8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot reference markers
    ax.scatter(*marker_positions.T, s=20, color='C0', zorder=5, label='Markers')
    labels = marker_labels if marker_labels is not None else range(len(marker_positions))
    for pos, lbl in zip(marker_positions, labels):
        ax.text(*pos, f' {lbl}', fontsize=7, color='C0')

    # Collect all plotted points to set equal axes later
    all_points = list(marker_positions)

    for cam_label, P in camera_matrices.items():
        pos = camera_position(P)
        all_points.append(pos)

        ax.scatter(*pos, s=50, color='C1', marker='^', zorder=5)
        ax.text(*pos, f'  {cam_label}', fontsize=8, color='C1')

        if K is not None:
            # Camera axes in world frame (columns of R^T)
            R = P[:, :3]
            cam_x, cam_y, cam_z = R.T[:, 0], R.T[:, 1], R.T[:, 2]

            # Frustum corner directions using the principal point ratio for aspect
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            half_w = cx / fx * frustum_size
            half_h = cy / fy * frustum_size
            apex = pos + frustum_size * cam_z

            corners = [
                apex + s_x * half_w * cam_x + s_y * half_h * cam_y
                for s_x, s_y in [(1, 1), (-1, 1), (-1, -1), (1, -1)]
            ]
            # Lines from camera centre to each corner
            for c in corners:
                ax.plot(*zip(pos, c), color='C1', lw=0.8, alpha=0.5)
            # Close the base rectangle
            for i in range(4):
                ax.plot(*zip(corners[i], corners[(i + 1) % 4]), color='C1', lw=0.8, alpha=0.5)

    # Equal aspect ratio: expand each axis symmetrically around the scene midpoint
    all_points = np.array(all_points)
    mid = (all_points.max(axis=0) + all_points.min(axis=0)) / 2
    half_range = (all_points.max(axis=0) - all_points.min(axis=0)).max() / 2 * 1.2
    ax.set_xlim(mid[0] - half_range, mid[0] + half_range)
    ax.set_ylim(mid[1] - half_range, mid[1] + half_range)
    ax.set_zlim(mid[2] - half_range, mid[2] + half_range)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    fig.tight_layout()
    return fig, ax
