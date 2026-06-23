"""Tests for the dataset loaders in stageguard.data.

These exercise the offline surface only (construction, error paths, windowing,
and the per-loader reshape contract) with tiny synthetic fixtures; no network
access or real raw data is required.
"""

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import pytest

from stageguard.data import (
    AccuSleepDataset,
    BaseSleepDataset,
    SHHSDataset,
    SleepAccelDataset,
    SleepBRLDataset,
)

N = 12  # epochs per synthetic file


def _accusleep_dir(root):
    d = root / "accusleep"
    d.mkdir()
    np.savez(
        d / "mouse.npz",
        eeg=np.random.default_rng(0).standard_normal((N, 8)).astype(np.float32),
        labels=(np.arange(N) % 3).astype(np.int64),
    )
    return d


def _actigraphy_dir(root):
    d = root / "actigraphy"
    d.mkdir()
    rng = np.random.default_rng(1)
    pd.DataFrame(
        {
            "timestamp": np.arange(N),
            "x": rng.standard_normal(N),
            "y": rng.standard_normal(N),
            "z": rng.standard_normal(N),
            "label": np.arange(N) % 2,
        }
    ).to_csv(d / "subject.csv", index=False)
    return d


def _bioradar_dir(root):
    d = root / "bioradar"
    d.mkdir()
    np.savez(
        d / "rec.npz",
        signals=np.random.default_rng(2).standard_normal((N, 5)).astype(np.float32),
        labels=(np.arange(N) % 3).astype(np.int64),
    )
    return d


def _shhs_dir(root):
    d = root / "shhs"
    d.mkdir()
    with h5py.File(d / "rec.h5", "w") as f:
        f["features"] = np.random.default_rng(3).standard_normal((N, 6)).astype(np.float32)
        f["labels"] = (np.arange(N) % 3).astype(np.int64)
    return d


def test_reexports_are_dataset_subclasses():
    for cls in (AccuSleepDataset, SleepAccelDataset, SHHSDataset, SleepBRLDataset):
        assert issubclass(cls, BaseSleepDataset)


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        AccuSleepDataset(tmp_path / "does_not_exist", sequence_length=4)


@pytest.mark.parametrize(
    "cls, subdir",
    [
        (AccuSleepDataset, "accusleep"),
        (SleepAccelDataset, "actigraphy"),
        (SleepBRLDataset, "bioradar"),
        (SHHSDataset, "shhs"),
    ],
)
def test_empty_directory_raises(tmp_path, cls, subdir):
    empty = tmp_path / subdir
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        cls(empty, sequence_length=4)


def test_accusleep_reshape(tmp_path):
    ds = AccuSleepDataset(_accusleep_dir(tmp_path), sequence_length=4)
    assert len(ds) == N - 4 + 1
    sig, lab = ds[0]
    assert sig.shape == (4, 1, 8)  # (seq, channels, samples)
    assert lab.shape == (4,)


def test_actigraphy_reshape(tmp_path):
    ds = SleepAccelDataset(_actigraphy_dir(tmp_path), sequence_length=4)
    sig, lab = ds[0]
    assert sig.shape == (4, 3, 1)  # x/y/z channels, one sample per epoch
    assert lab.dtype == np.int64


def test_bioradar_reshape(tmp_path):
    ds = SleepBRLDataset(_bioradar_dir(tmp_path), sequence_length=4)
    sig, lab = ds[0]
    assert sig.shape == (4, 5, 1)
    assert lab.dtype == np.int64


def test_shhs_reshape(tmp_path):
    ds = SHHSDataset(_shhs_dir(tmp_path), sequence_length=4)
    sig, lab = ds[0]
    assert sig.shape == (4, 6, 1)
    assert lab.dtype == np.int64


def test_len_zero_when_sequence_longer_than_data(tmp_path):
    ds = AccuSleepDataset(_accusleep_dir(tmp_path), sequence_length=N + 5)
    assert len(ds) == 0


def test_download_instructions():
    msg = AccuSleepDataset.download_instructions()
    assert "AccuSleep" in msg and "osf.io/py5eb" in msg
    # SLEEPBRL has no download URL; instructions are still a non-empty string.
    brl = SleepBRLDataset.download_instructions()
    assert "SLEEPBRL" in brl
