# High-Speed-Camera-Based Structural Dynamics Identification: An Open-Source-Supported Tutorial

**Code repository for the tutorial paper published in *Mechanical Systems and Signal Processing*.**

> D. Gorjup, K. Zaletelj, J. Slavič,
> *High-Speed-Camera-Based Structural Dynamics Identification: An Open-Source-Supported Tutorial*,
> Mechanical Systems and Signal Processing, TODO (TODO).
> DOI: [TODO](https://doi.org/TODO)

The measured data is published separately as an open dataset on Zenodo:
[10.5281/zenodo.22770610](https://doi.org/10.5281/zenodo.22770610) (CC BY 4.0).

---

## Setup

Python 3.11+ is required. `requirements.txt` pins the versions the notebook was tested
against; newer versions will likely also work.

```bash
git clone git@github.com:ladisk/mssp-image-tutorial.git
cd mssp-image-tutorial

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name=mssp-tutorial
```

Open `image_based_dynamics_tutorial.ipynb` in JupyterLab or VS Code with the `mssp-tutorial` kernel. 

The notebook follows the paper section by section; the paper carries the methods and their parameters.

The measured data (~20 GB) is **not** part of the repository. The notebook downloads it
from Zenodo at the start of section 2, see [Measured data](#measured-data).

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

All measurements are published as the citable Zenodo **printer-bed dataset**. The
repository pins the **version DOI** 
[10.5281/zenodo.22770610](https://doi.org/10.5281/zenodo.22770610), so that every 
reader obtains byte-identical files and reproduces the results reported in the paper.

Each archive extracts into `data/`, with the record directories inside. The raw video is
split per record, so the sweep-only sections can be run without the broadband videos:

| Archive | Size | MD5 | Extracts to |
|---------|------|-----|-------------|
| `mraw_plate_sweep.zip` | 8.5 GB | `4bfdf017ef1ceff24e5d18d6709a6c22` | `data/plate_sweep/mraw/` |
| `mraw_plate_random.zip` | 9.3 GB | `092f9fd3462f7eb296482201f8a41e78` | `data/plate_random/mraw/` |
| `measured_signals.zip` | 6.6 MB | `ff14891f76dfb49150e7775a71f462fe` | `data/{plate_sweep,plate_random}/measured_signals/` |
| `calibration.zip` | 2.4 MB | `a8277ac4dcb82d5945b9565435872754` | `data/{plate_sweep,plate_random}/calibration/` |

`ensure_dataset()` in `methods/data.py` downloads, verifies and extracts the archives. The
notebook calls it at the start of section 2; it can also be called directly:

```python
from methods.data import ensure_dataset

ensure_dataset()                     # all archives (~20 GB on first run)
ensure_dataset('calibration')        # a single archive
```

It is safe to re-run: data already in place is detected via marker files in
`data/.zenodo/` and skipped. Every archive is checked against the MD5 checksum published
in the Zenodo record; a mismatch raises an error.

> **Slow or unreliable connection?** Download the archives manually from the
> [record page](https://zenodo.org/records/22770610) and place them in `data/`.
> `ensure_dataset()` verifies and extracts them, and skips the download.

---

## Measurement

Two experiments on the same suspended thin plate (3D-printer bed) with a speckle pattern,
each recorded sequentially from **two camera views** (Photron FASTCAM SA-Z, 768×768 px,
8-bit, 5000 fps, 10 000 frames = 2 s). 

Excitation force and reference acceleration are sampled at 25 600 Hz, hardware-synchronised with the camera.

| Record | Excitation | Camera views | Repetitions | Used in |
|--------|-----------|--------------|-------------|---------|
| `plate_sweep` | swept sine, 60–500 Hz | 2, triangulated into 3-D displacements | 4 (view 0), 1 (view 1) | Sec. 5.1–5.3: FRFs, hybrid EMA, MAC |
| `plate_random` | broadband random, 60–500 Hz | 2, independent (OMA repeatability check) | 1 per view | Sec. 5.4: OMA |

---

## Provided data (HDF5)

All data is stored in [HDF5](https://www.hdfgroup.org/solutions/hdf5/), readable with
`methods.utils.load_hdf5` or directly with `h5py`.

**Measured signals**: `data/<record>/measured_signals/*.hdf5`, downloaded from Zenodo.
Force and acceleration time series, one file per view and repetition:
* `plate_sweep`: `view_0.hdf5`, `view_0_02.hdf5`, `view_0_03.hdf5`, `view_0_04.hdf5`, `view_1.hdf5`;
* `plate_random`: `view_0_01.hdf5`, `view_1_01.hdf5`.

```
InputTask/
    data          float64 (51200, 3)   columns: force [N], acceleration [m/s²], trigger
    time          float64 (51200,)     time axis [s]
    channel_names str list             ['force', 'acc', 'trigger']
    sample_rate   int                  25600 [Hz]
```

**Image displacements**: `data/<record>/results/image_displacements/*.hdf5`, in the
repository. Lucas–Kanade optical-flow displacements, computed from the videos by the
notebook and included so that the modal analysis can be run without repeating the tracking. 

* `plate_sweep`: one file per repetition (`idi_lk_displacements.hdf5`, `_02`,
`_03`, `_04`) on the 100-point grid; 
* `plate_random`: `idi_lk_displacements.hdf5` on the coarser 5×5 grid used for OMA.

```
view 0    float64 (n_points, 10000, 2)   displacements [px]; axes: (point, frame, [v, u])
view 1    float64 (n_points, 10000, 2)   n_points = 100 (plate_sweep), 25 (plate_random)
```

---

## Citation

Please cite both the paper and the dataset.

```bibtex
@article{gorjup2026tutorial,
  author  = {Gorjup, Domen and Zaletelj, Klemen and Slavi\v{c}, Janko},
  title   = {High-Speed-Camera-Based Structural Dynamics Identification: An Open-Source-Supported Tutorial},
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
  doi       = {10.5281/zenodo.22770610},
  url       = {https://doi.org/10.5281/zenodo.22770610},
}
```

## Authors

- Domen Gorjup, University of Ljubljana, Faculty of Mechanical Engineering
- Klemen Zaletelj, University of Ljubljana, Faculty of Mechanical Engineering
- Janko Slavič, University of Ljubljana, Faculty of Mechanical Engineering
