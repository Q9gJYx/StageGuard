"""Reproducible build of the local AccuSleep demo artifact.

Downloads one mouse recording from the public AccuSleep OSF project and applies the same
load + decimate + epoch + stage-remap steps the demo notebook uses, for the held-out
night, writing raw (un-normalized) epochs to ``data/accusleep_demo.npz`` in the format
expected by ``stageguard.data.AccuSleepDataset`` (keys ``eeg`` of shape
``(n_epochs, samples_per_epoch)`` and ``labels`` of shape ``(n_epochs,)``). The notebook's
four-night training split and train-statistic z-scoring are not reproduced here.

Nothing this script produces is committed: the AccuSleep recordings carry no
redistribution license, so both the download cache (``accusleep_data/``) and the output
artifact (``data/accusleep_demo.npz``) are git-ignored. See ``data/README.md``.

Usage:
    python scripts/build_accusleep_demo.py
"""

from __future__ import annotations

import glob
import os
import urllib.request
import zipfile

import numpy as np
from scipy.io import loadmat
from scipy.signal import decimate

OSF_FILE_ID = "5d9fdbfff6b03e000d1cb300"      # Mouse03.zip on OSF project py5eb
CACHE_DIR = "accusleep_data"
OUT = os.path.join("data", "accusleep_demo.npz")
FS_RAW, DS, EPOCH_SEC = 512, 4, 2.5            # decimate 512 -> 128 Hz, 2.5 s epochs
SPE = int(FS_RAW // DS * EPOCH_SEC)            # 320 samples per epoch
REMAP = {2: 0, 3: 1, 1: 2}                     # AccuSleep 1=REM,2=Wake,3=NREM -> Wake,NREM,REM
STAGE_NAMES = ["Wake", "NREM", "REM"]
EVAL_DAY = 1                                    # the demo's held-out night


def download_and_extract() -> list[str]:
    """Download Mouse03.zip into CACHE_DIR (skip if cached) and return session dirs."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    zip_path = os.path.join(CACHE_DIR, "Mouse03.zip")
    if not (os.path.exists(zip_path) and os.path.getsize(zip_path) > 2e8):
        url = f"https://osf.io/download/{OSF_FILE_ID}/"
        print(f"Downloading Mouse03.zip (~233 MB) from {url} ...")
        req = urllib.request.Request(url, headers={"User-Agent": "stageguard-demo/1.0"})
        with urllib.request.urlopen(req, timeout=180) as r, open(zip_path, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
    assert os.path.getsize(zip_path) > 2e8, "download looks truncated; re-run"
    if not glob.glob(os.path.join(CACHE_DIR, "**", "Day*", ""), recursive=True):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(CACHE_DIR)
    return sorted(glob.glob(os.path.join(CACHE_DIR, "**", "Day*", ""), recursive=True))


def load_mat_var(path: str, name: str) -> np.ndarray:
    """Load one variable from a .mat file (v7 via scipy, v7.3/HDF5 via h5py)."""
    try:
        return np.asarray(loadmat(path)[name]).squeeze()
    except NotImplementedError:
        import h5py

        with h5py.File(path, "r") as f:
            return np.asarray(f[name]).squeeze()


def load_session(day_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (epochs (n, SPE) float32, labels (n,) in {0,1,2}) for one night."""
    eeg = load_mat_var(os.path.join(day_dir, "EEG.mat"), "EEG").astype(np.float64)
    lab = load_mat_var(os.path.join(day_dir, "labels.mat"), "labels").astype(int)
    eeg = decimate(eeg, DS, ftype="fir")
    n = len(lab)
    epochs = eeg[: n * SPE].reshape(n, SPE).astype(np.float32)
    y = np.array([REMAP[v] for v in lab], dtype=np.int64)
    return epochs, y


def main() -> None:
    sessions = download_and_extract()
    print(f"{len(sessions)} sessions found; building artifact from held-out Day{EVAL_DAY}")
    eeg, labels = load_session(sessions[EVAL_DAY - 1])
    u, c = np.unique(labels, return_counts=True)
    print("label distribution:", {STAGE_NAMES[int(k)]: int(v) for k, v in zip(u, c)})
    os.makedirs("data", exist_ok=True)
    np.savez(OUT, eeg=eeg, labels=labels)
    print(f"wrote {OUT}: eeg{eeg.shape} labels{labels.shape} "
          "(git-ignored; not for redistribution)")


if __name__ == "__main__":
    main()
