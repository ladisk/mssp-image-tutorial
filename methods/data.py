"""
On-demand retrieval of the measured data from Zenodo.

The measurements used in this tutorial (high-speed camera videos, checkerboard
calibration images, force and acceleration signals) are not stored in this
repository. They are published as a citable Zenodo record, which is the single
source of truth, and are downloaded into `data/` on demand by `ensure_dataset`.
"""

import hashlib
import zipfile
from pathlib import Path

import requests
from tqdm.auto import tqdm


# Version DOI of the dataset. This is pinned deliberately: the tutorial claims
# that the reader reproduces the exact results reported in the paper, which
# requires byte-identical input files. Do not replace it with the concept DOI
# (10.5281/zenodo.21476608), which always resolves to the latest version.
RECORD_ID = '22770610'
DOI = '10.5281/zenodo.22770610'
RECORD_URL = f'https://zenodo.org/records/{RECORD_ID}'
API_URL = f'https://zenodo.org/api/records/{RECORD_ID}'

# Archives in the record. Each one extracts into `data/`, with the record
# directories (plate_sweep/, plate_random/) inside. The raw video is split per
# record, so the sweep-only sections can be run without the broadband videos.
DATASETS = ('mraw_plate_sweep', 'mraw_plate_random', 'measured_signals', 'calibration')

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
MARKER_DIR = DATA_DIR / '.zenodo'

_CHUNK = 1 << 20   # 1 MiB, used for both downloading and hashing
_record_files_cache = None


def ensure_dataset(files=None):
    """
    Download and extract the measured data from Zenodo, unless already present.

    Safe to call repeatedly: a dataset that has been extracted successfully is
    recognised by a marker file in `data/.zenodo/` and skipped without any
    network access, so re-running a notebook cell costs nothing.

    Every archive is verified against the MD5 checksum published in the Zenodo
    record before it is extracted. A mismatch raises, rather than proceeding
    with corrupt data. An interrupted download or extraction leaves no marker
    behind, so the next call simply fetches the dataset again.

    An archive already present in `data/` (downloaded manually from
    `RECORD_URL`, for instance, on an unreliable connection) is used as-is if
    its checksum matches, and the download is skipped.

    Parameters
    ----------
    files : iterable of str, optional
        Datasets to fetch, any of `DATASETS`. All of them by default.
    """
    if files is None:
        names = DATASETS
    elif isinstance(files, str):
        names = (files,)
    else:
        names = tuple(files)

    unknown = set(names) - set(DATASETS)
    if unknown:
        raise ValueError(f'Unknown dataset(s) {sorted(unknown)}; '
                         f'expected any of {list(DATASETS)}.')

    for name in names:
        marker = MARKER_DIR / f'{name}.md5'
        if marker.exists():
            continue

        checksum = _lookup_md5(name)
        archive = DATA_DIR / f'{name}.zip'

        if not (archive.exists() and _md5(archive) == checksum):
            print(f'Downloading {name}.zip from {RECORD_URL}')
            _download(name, archive)
            downloaded = _md5(archive)
            if downloaded != checksum:
                raise RuntimeError(
                    f'Checksum mismatch for {name}.zip: expected {checksum}, '
                    f'got {downloaded}. The download is corrupt — re-run '
                    f'ensure_dataset() to fetch it again.'
                )

        print(f'Extracting {name}.zip')
        with zipfile.ZipFile(archive) as archive_file:
            archive_file.extractall(DATA_DIR)

        # Written only once extraction has completed, so that an interrupted
        # extraction is not mistaken for a finished one on the next call.
        MARKER_DIR.mkdir(parents=True, exist_ok=True)
        marker.write_text(checksum)
        archive.unlink()


def _record_files():
    """Map archive name to MD5 checksum, from the Zenodo API (queried once)."""
    global _record_files_cache
    if _record_files_cache is None:
        response = requests.get(API_URL, timeout=30)
        response.raise_for_status()
        _record_files_cache = {
            entry['key']: entry['checksum'].removeprefix('md5:')
            for entry in response.json()['files']
        }
    return _record_files_cache


def _lookup_md5(name):
    """Published MD5 checksum of a single archive."""
    files = _record_files()
    key = f'{name}.zip'
    if key not in files:
        raise RuntimeError(f'{key} is not part of Zenodo record {RECORD_ID}. '
                           f'Available: {sorted(files)}.')
    return files[key]


def _download(name, dest):
    """Stream an archive to `dest`, showing a progress bar."""
    url = f'{API_URL}/files/{name}.zip/content'
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total = int(response.headers.get('Content-Length', 0))
        with open(dest, 'wb') as f, tqdm(total=total, unit='B', unit_scale=True,
                                         unit_divisor=1024, desc=f'{name}.zip') as bar:
            for chunk in response.iter_content(_CHUNK):
                f.write(chunk)
                bar.update(len(chunk))


def _md5(path):
    """MD5 checksum of a file, read in chunks."""
    digest = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(_CHUNK), b''):
            digest.update(chunk)
    return digest.hexdigest()
