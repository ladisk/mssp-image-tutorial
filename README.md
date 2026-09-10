# An Open-source Supported Guide to High-speed Camera Based Structural Dynamics Identification

**Code repository for the tutorial paper published in *Mechanical Systems and Signal Processing*.**

> D. Gorjup, K. Zaletelj, J. Slavič,
> *An Open-source Supported Tutorial to High-speed Camera Based Structural Dynamics Identification*,
> Mechanical Systems and Signal Processing, TODO (TODO).
> DOI: [TODO](https://doi.org/TODO)

The measured data is published separately as an open dataset on Zenodo:
[10.5281/zenodo.21476609](https://doi.org/10.5281/zenodo.21476609) (CC BY 4.0).

---

## Setup

Python 3.11+ is required.

`requirements.txt` pins the most recent versions the notebook was tested against, purely to ensure repeatability. Newer package versions will likely also work.

```bash
# Clone (no Git LFS required)
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

The measured data is **not** part of the repository — it is downloaded from Zenodo
by the notebook itself, in section 2.1. See [Measured data](#measured-data) below.

---

## Repository structure

```
image_based_dynamics_tutorial.ipynb   # main tutorial notebook
methods/                              # reusable Python modules
    data.py                           # on-demand download of the measured data from Zenodo
    calibration.py                    # intrinsic and extrinsic camera calibration
    triangulation.py                  # DLT point triangulation and ODS reconstruction
    modal.py                          # FRF estimation and modal analysis utilities
    visualization.py                  # plotting and animation helpers
    utils.py                          # general utilities (I/O, image loading)
data/
    plate_speckle.png                 # speckle pattern image for visualization (in repository)
    plate_sweep/                      # sine-sweep record — EMA (FRF, hybrid method, MAC)
        results/image_displacements/  # precomputed LK optical-flow displacements (in repository)
        mraw/                         # Photron videos, two views      ─┐
        measured_signals/             # force and acceleration signals  │ downloaded
        calibration/                  # checkerboard calibration images │ from Zenodo
    plate_random/                     # broadband-random record — OMA
        results/image_displacements/  # precomputed LK optical-flow displacements (in repository)
        mraw/                         # Photron videos, two views      ─┐
        measured_signals/             # force and acceleration signals  │ downloaded
        calibration/                  # checkerboard calibration images │ from Zenodo
    .zenodo/                          # download markers (do not edit)
```

---

## Measured data

All measurements are published as a citable Zenodo **printer-bed dataset** and downloaded on
demand. The repository pins the **version DOI**
[10.5281/zenodo.21476609](https://doi.org/10.5281/zenodo.21476609), not the concept
DOI, so that every reader obtains byte-identical files and reproduces the exact
results reported in the paper.

Each archive contains both the sweep and broadband-random records, nested under
`plate_sweep/` and `plate_random/` respectively (e.g. `mraw.zip` extracts to
`data/plate_sweep/mraw/` **and** `data/plate_random/mraw/`).

| Archive | Size | MD5 | Extracts to |
|---------|------|-----|-------------|
| `mraw.zip` | TODO | TODO | `data/{plate_sweep,plate_random}/mraw/` |
| `measured_signals.zip` | TODO | TODO | `data/{plate_sweep,plate_random}/measured_signals/` |
| `calibration.zip` | TODO | TODO | `data/{plate_sweep,plate_random}/calibration/` |

The notebook calls `ensure_dataset()` in section 2.1, before the data is first
used. It can also be called directly:

```python
from methods.data import ensure_dataset

ensure_dataset()                     # all archives (~450 MB on first run)
ensure_dataset('calibration')        # a single archive
```

The function is safe to re-run. Data already in place is detected via marker files
in `data/.zenodo/` and skipped without any network access, so re-running a notebook
cell costs nothing. Each archive is verified against the MD5 checksum published in
the Zenodo record before extraction; a mismatch raises an error rather than
proceeding with corrupt data. An interrupted download or extraction leaves no
marker behind, so the next call simply fetches the dataset again.

> **Slow or unreliable connection?** Download any of the archives manually from the
> [record page](https://zenodo.org/records/21476609) and place them in `data/`.
> `ensure_dataset()` verifies their checksums, extracts them, and skips the download.

---

## Measurement

The printer-bed dataset consists of two experiments on the same suspended thin plate
(3D-printer bed) with a speckle pattern, each recorded sequentially from **two camera
views**.

**`plate_sweep`** — swept-sine excitation, used throughout Sec.~5.1–5.3 (FRFs, the
hybrid EMA method, MAC validation):

- **Excitation:** swept-sine, 60–500 Hz
- **Views:** 2 (multi-view for 3-D displacement reconstruction using frequency-domain triangulation)
- **Repetitions:** 4 (view 0) / 1 (view 1) — used for ensemble-averaged FRF estimation
- **Camera:** Photron high-speed, 768×768 px at 5000 fps, 10000 frames (2 s)
- **Force and acceleration** sampled at 25600 Hz (DAQ)

**`plate_random`** — broadband random excitation, used for the OMA section (Sec.~5.4);
a sine sweep does not satisfy the (quasi-)stationarity assumption underlying
output-only identification, so this record was captured separately:

- **Excitation:** broadband random
- **Views:** 2 (independent camera setups; used as an OMA repeatability cross-check, not for triangulation)
- **Camera:** Photron high-speed, 768×768 px at 5000 fps, 10000 frames (2 s)
- **Force and acceleration** sampled at 25600 Hz (DAQ)

Pole identification in the notebook is restricted to the 60–500 Hz band of interest
(both datasets, both excitation types).

---

## Provided data (HDF5)

All precomputed data is stored in [HDF5](https://www.hdfgroup.org/solutions/hdf5/) format, readable without any proprietary software. Load with `methods.utils.load_hdf5` or directly with `h5py`.

### `data/plate_sweep/measured_signals/view_0[_02|_03|_04].hdf5` *(downloaded from Zenodo)*

Force and acceleration time series for the 4 view-0 repetitions.

```
InputTask/
    data          float64 (51200, 3)   columns: force [N], acceleration [m/s²], trigger
    time          float64 (51200,)     time axis [s]
    channel_names str list             ['force', 'acc', 'trigger']
    sample_rate   int                  25600 [Hz]
```

### `data/plate_sweep/measured_signals/view_1.hdf5` *(downloaded from Zenodo)*

Same structure, single repetition for view 1.

### `data/plate_sweep/results/image_displacements/idi_lk_displacements[_02|_03|_04].hdf5` *(in repository)*

Lucas–Kanade optical-flow displacements for each measurement repetition. These are
derived results, computed from the videos by the notebook, and are included in the
repository so that the modal analysis can be run without repeating the tracking.

```
view 0    float64 (100, 10000, 2)   displacements [px] — axes: (point, frame, [v, u])
view 1    float64 (100, 10000, 2)
```

### `data/plate_random/measured_signals/view_0_01.hdf5`, `view_1_01.hdf5` *(downloaded from Zenodo)*

Force and acceleration time series for the broadband random test, one file per view
(same `InputTask` structure as `plate_sweep`, `data` shape `(51200, 3)`).

### `data/plate_random/results/image_displacements/idi_lk_displacements.hdf5` *(in repository)*

Lucas–Kanade optical-flow displacements for the broadband random test, both views,
computed on the coarser 5×5 point grid used for the OMA analysis.

```
view 0    float64 (25, 10000, 2)   displacements [px] — axes: (point, frame, [v, u])
view 1    float64 (25, 10000, 2)
```

---

## Citation

Please cite both the paper and the dataset.

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

@dataset{gorjup2026dataset,
  author    = {Gorjup, Domen and Zaletelj, Klemen and Slavi\v{c}, Janko},
  title     = {Printer-bed dataset: high-speed-camera and accelerometer modal
               measurements of a suspended 3D-printer bed},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21476609},
  url       = {https://doi.org/10.5281/zenodo.21476609},
}
```

## Authors

- Domen Gorjup — University of Ljubljana, Faculty of Mechanical Engineering
- Klemen Zaletelj — University of Ljubljana, Faculty of Mechanical Engineering
- Janko Slavič — University of Ljubljana, Faculty of Mechanical Engineering
