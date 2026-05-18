"""
Supporting methods for the MSSP tutorial:
  "An Open-source Supported Guide to High-speed Camera Based
   Structural Dynamics Identification"

Submodules
----------
utils          Image/video I/O and basic array helpers.
calibration    Camera calibration and projection matrix construction.
triangulation  Geometric (DLT) and frequency-domain 3-D reconstruction.
modal          FRF estimation, modal identification (LSCF/LSFD), and MAC.
visualization  Plotting helpers used throughout the tutorial notebook.
"""

from methods import utils, calibration, triangulation, modal, visualization

__all__ = ["utils", "calibration", "triangulation", "modal", "visualization"]
