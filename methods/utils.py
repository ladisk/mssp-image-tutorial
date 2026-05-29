"""
General-purpose utilities: image/video I/O, format conversions, and basic
array helpers.

Not specific to any paper section.
"""

import cv2
import numpy as np
import h5py


# --- HDF5 I/O ---------------------------------------------------------------

def save_hdf5(path, data):
    """Save a nested dict of arrays/scalars to an HDF5 file."""
    with h5py.File(path, 'w') as f:
        _write_group(f, data)


def load_hdf5(path):
    """Load an HDF5 file saved by save_hdf5 back into a nested dict."""
    with h5py.File(path, 'r') as f:
        return _read_group(f)


def _write_group(group, d):
    for key, val in d.items():
        if isinstance(val, dict):
            _write_group(group.require_group(str(key)), val)
        elif isinstance(val, list):
            group.create_dataset(str(key), data=np.array(val, dtype=object),
                                 dtype=h5py.string_dtype())
        elif isinstance(val, np.ndarray):
            group.create_dataset(str(key), data=val)
        else:
            group.create_dataset(str(key), data=val)


def _read_group(group):
    out = {}
    for key in group:
        item = group[key]
        if isinstance(item, h5py.Group):
            out[key] = _read_group(item)
        else:
            val = item[()]
            if isinstance(val, np.ndarray) and val.dtype.kind in ('S', 'O'):
                out[key] = [v.decode() if isinstance(v, bytes) else str(v) for v in val]
            elif isinstance(val, bytes):
                out[key] = val.decode()
            else:
                out[key] = val
    return out

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

