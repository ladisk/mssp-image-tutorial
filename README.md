# An Open-source Supported Guide to High-speed Camera Based Structural Dynamics Identification

**Code repository for the tutorial paper published in *Mechanical Systems and Signal Processing*.**

> D. Gorjup, K. Zaletelj, J. Slavič,
> *An Open-source Supported Tutorial to High-speed Camera Based Structural Dynamics Identification*,
> Mechanical Systems and Signal Processing, TODO (TODO).
> DOI: [TODO](https://doi.org/TODO)

---

## Setup

Python 3.11+ is required.

```bash
# Clone (videos are stored with Git LFS — install LFS first)
git lfs install
git clone git@github.com:domengorjup/mssp-image-tutorial.git
cd mssp-image-tutorial

# Create virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Register the kernel and launch the notebook
python -m ipykernel install --user --name=mssp-tutorial
jupyter notebook image_based_dynamics_tutorial.ipynb
```

> **No git?** Use the *Download ZIP* button on GitHub — it includes the real video files.  
> **No LFS?** If you cloned without LFS, fetch video files with `git lfs pull`.

---

## Repository structure

```
image_based_dynamics_tutorial.ipynb   # main tutorial notebook
methods/                              # reusable Python modules
    calibration.py                    # intrinsic and extrinsic camera calibration
    triangulation.py                  # DLT point triangulation and ODS reconstruction
    modal.py                          # FRF estimation and modal analysis utilities
    visualization.py                  # plotting and animation helpers
    utils.py                          # general utilities (I/O, image loading)
data/
    calibration/                      # checkerboard calibration images
    mraw_downsized/                   # downsampled Photron videos (two views, Git LFS)
    mraw/                             # full-resolution videos (Git LFS, available on request)
    measured_signals/                 # force and acceleration signals (HDF5)
    results/image_displacements/      # precomputed LK optical-flow displacements (HDF5)
    plate_speckle.png                 # speckle pattern image for visualization
```

---

## Measurement

The example dataset consists of a sine-sweep excitation experiment on a thin plate with a speckle pattern, recorded sequentially from **two camera views**.

- **Excitation:** swept-sine, 60–500 Hz
- **Views:** 2 (multi-view for 3-D displacement reconstruction using frequency-domain triangulation)
- **Repetitions:** 4 (view 0) / 1 (view 1) — used for ensemble-averaged FRF estimation
- **Camera:** Photron high-speed, downsampled to 384×384 px at 1000 fps for the provided example data
- **Force and acceleration** sampled at 25600 Hz (DAQ), clipped to camera Nyquist (500 Hz) during analysis

---

## Provided data (HDF5)

All precomputed data is stored in [HDF5](https://www.hdfgroup.org/solutions/hdf5/) format, readable without any proprietary software. Load with `methods.utils.load_hdf5` or directly with `h5py`.

### `data/measured_signals/view_0[_02|_03|_04].hdf5`

Force and acceleration time series for the 4 view-0 repetitions.

```
InputTask/
    data          float64 (51200, 3)   columns: force [N], acceleration [m/s²], trigger
    time          float64 (51200,)     time axis [s]
    channel_names str list             ['force', 'acc', 'trigger']
    sample_rate   int                  25600 [Hz]
```

### `data/measured_signals/view_1.hdf5`

Same structure, single repetition for view 1.

### `data/results/image_displacements/idi_lk_displacements[_02|_03|_04].hdf5`

Lucas–Kanade optical-flow displacements for each measurement repetition.

```
view 0    float64 (100, 2000, 2)   displacements [px] — axes: (point, frame, [v, u])
view 1    float64 (100, 2000, 2)
```

---

## Citation

```bibtex
@article{gorjup2026tutorial,
  author  = {Gorjup, Domen and Zaletelj, Klemen and Slavi\v{c}, Janko},
  title   = {An Open-source Supported Guide to High-speed Camera Based Structural Dynamics Identification},
  journal = {Mechanical Systems and Signal Processing},
  year    = {TODO},
  volume  = {TODO},
  pages   = {TODO},
  doi     = {TODO},
}
```

## Authors

- Domen Gorjup — University of Ljubljana, Faculty of Mechanical Engineering
- Klemen Zaletelj — University of Ljubljana, Faculty of Mechanical Engineering
- Janko Slavič — University of Ljubljana, Faculty of Mechanical Engineering
