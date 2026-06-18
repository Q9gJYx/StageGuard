# Datasets

StageGuard is evaluated on four sleep datasets spanning four modalities. The data loaders in
`stageguard/data/` expect the data to be **pre-downloaded**; each loader exposes a `DOWNLOAD_URL` and a
`download_instructions()` helper:

```python
from stageguard.data.mouse_eeg import AccuSleepDataset
print(AccuSleepDataset.download_instructions())
ds = AccuSleepDataset("path/to/accusleep_dir", sequence_length=100)
```

## Overview

| Dataset | Modality | Classes | Epoch | Access | Loader |
|---|---|---|---|---|---|
| AccuSleep | Mouse EEG/EMG | 3 (Wake, NREM, REM) | 4 s | Open ([Zenodo](https://zenodo.org/records/4079563)) | `data/mouse_eeg.py` |
| Sleep-Accel | Wrist actigraphy | 2 (Wake, Sleep) | 30 s | Open, ODC-By ([PhysioNet](https://physionet.org/content/sleep-accel/1.0.0/)) | `data/actigraphy.py` |
| SHHS | Cardiorespiratory | 3 (Wake, Light, Deep) | 30 s | Data-use agreement ([NSRR](https://sleepdata.org/datasets/shhs)) | `data/cardiorespiratory.py` |
| SLEEPBRL | Bioradar | 3 (Wake, Light, Deep) | 30 s | Contact authors | `data/bioradar.py` |

Each loader concatenates per-recording files found in `data_dir` and returns `(signals, labels)`.

## On-disk formats expected by each loader

- **AccuSleep** (`AccuSleepDataset`): one or more `*.npz` files, each with `eeg` of shape
  `(n_epochs, samples_per_epoch)` and integer `labels` of shape `(n_epochs,)` (Wake=0, NREM=1, REM=2).
  Defaults: `fs=256`, `epoch_sec=4.0`.
- **Sleep-Accel** (`SleepAccelDataset`): per-subject CSV with columns `timestamp, x, y, z, label`, one row per
  30 s epoch, binary `label` (Wake=0, Sleep=1).
- **SHHS** (`SHHSDataset`): `*.h5` files, each with HDF5 datasets `features` and `labels` (Wake=0, Light=1,
  Deep=2). Access requires an NSRR data-use agreement.
- **SLEEPBRL** (`SleepBRLDataset`): `*.npz` files, each with `signals` and `labels` (Wake=0, Light=1, Deep=2).
  Not publicly redistributable; contact the authors.

Modality-specific physiological constraints live in `configs/*.yaml` (one per dataset); load them with
`ModalityConfig.from_yaml(...)`.

## Demo data subset

The notebook [`notebooks/demo_stageguard.ipynb`](../notebooks/demo_stageguard.ipynb) does **not** require any of
the above. It uses a small, redistributable subset of **Sleep-Accel PSG hypnograms** (labels only, collapsed
to 3 states) shipped at [`data/sleepaccel_demo.npz`](../data/sleepaccel_demo.npz). See
[`data/README.md`](../data/README.md) for provenance, the ODC-By attribution, and the stage-collapse mapping.

## Citations

Cite the dataset you use:

- AccuSleep: Barger et al. (2019).
- Sleep-Accel: Walch et al. (2019), `10.1093/sleep/zsz180`; PhysioNet record `10.13026/hmhs-py35`.
- SHHS: Quan et al. (1997); see the NSRR for current citation requirements.
- SLEEPBRL: Tataraidze et al. (2015).
