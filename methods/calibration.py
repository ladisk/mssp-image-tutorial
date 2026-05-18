"""
Camera calibration utilities (paper Section 2.2).

Outputs (projection matrices, distortion coefficients) feed into
`triangulation.py` and can also be used to undistort frames before
displacement identification.
"""
