"""One-time conversion of legacy pickle data files to HDF5."""
import glob
import os
import pickle
import warnings
import sys

sys.path.insert(0, os.path.dirname(__file__))
from methods.utils import save_hdf5

PKL_GLOBS = [
    'data/measured_signals/*.pkl',
    'data/results/image_displacements/*.pkl',
]

for pattern in PKL_GLOBS:
    for pkl_path in sorted(glob.glob(pattern)):
        hdf5_path = pkl_path.replace('.pkl', '.hdf5')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
        save_hdf5(hdf5_path, data)
        print(f'{pkl_path}  →  {hdf5_path}')
