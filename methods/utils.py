"""
General-purpose utilities: image/video I/O, format conversions, and basic
array helpers.

Not specific to any paper section.
"""

import cv2
import numpy as np

# --- Image I/O --------------------------------------------------------------

def load_bw_image(path):
    """
    Load a single-band (grayscale) image from the specified path and return it as a NumPy array.
    """
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image at path: {path}")
    
    return image

# --- Photron MRAW / CIH / CIHX utilities ------------------------------------

def metadata_overview(mraw_info):
    """
    Print a human-readable overview of the metadata contained in a CIH/CIHXfile.
    """
    frame_rate = mraw_info["Record Rate(fps)"]
    frame_count = mraw_info["Total Frame"]
    image_height = mraw_info["Image Height"]
    image_width = mraw_info["Image Width"]
    bit_depth = mraw_info["Color Bit"]
    video_size_GB = (image_height * image_width * (bit_depth / 8) * frame_count) / (1024**3)

    print(f'Frame rate: {frame_rate} fps')
    print(f'Number of frames: {frame_count}')
    print(f'Image size (h x w): {image_height} x {image_width} pixels')
    print(f'Image bit depth: {bit_depth} bits per pixel')
    print(f'Approximate video size: {video_size_GB:.2f} GB')

