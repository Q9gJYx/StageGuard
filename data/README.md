# Demo data

This directory ships a single small artifact, **`sleepaccel_demo.npz`**, used by
[`notebooks/demo_stageguard.ipynb`](../notebooks/demo_stageguard.ipynb). It contains **labels only**
(sleep-stage hypnograms) for 5 subjects — **no raw accelerometer or photoplethysmography signals are
redistributed.** It exists so the notebook can demonstrate StageGuard's constrained decoder end-to-end on
real PSG hypnograms without any download or data-use agreement.

> StageGuard's full datasets (AccuSleep, Sleep-Accel, SHHS, SLEEPBRL) are described in
> [`docs/datasets.md`](../docs/datasets.md); most must be obtained separately.

## Provenance

Derived from the PhysioNet dataset **"Motion and heart rate from a wrist-worn wearable and labeled sleep from
polysomnography"** v1.0.0 (Walch et al.), DOI [`10.13026/hmhs-py35`](https://doi.org/10.13026/hmhs-py35),
<https://physionet.org/content/sleep-accel/1.0.0/>. Only the per-subject PSG label files
(`labels/<id>_labeled_sleep.txt`) were used.

Subjects shipped (PhysioNet IDs): **4426783, 781756, 9961348, 759667, 1066528**.

## License & attribution

The source dataset is released under the **Open Data Commons Attribution License v1.0 (ODC-By)**. Per its
terms, this redistributed subset carries the following attribution:

> Contains information from *Motion and heart rate from a wrist-worn wearable and labeled sleep from
> polysomnography* (PhysioNet, DOI 10.13026/hmhs-py35), which is made available under the
> [ODC Attribution License](https://opendatacommons.org/licenses/by/1.0/).

Please cite the original work if you use this artifact:

```bibtex
@article{walch2019sleep,
  author  = {Walch, Olivia and Huang, Yitong and Forger, Daniel and Goldstein, Cathy},
  title   = {Sleep stage prediction with raw acceleration and photoplethysmography heart rate
             data derived from a consumer wearable device},
  journal = {Sleep},
  volume  = {42},
  number  = {12},
  pages   = {zsz180},
  year    = {2019},
  doi     = {10.1093/sleep/zsz180}
}

@misc{physionet_sleepaccel,
  author = {Walch, Olivia},
  title  = {Motion and heart rate from a wrist-worn wearable and labeled sleep from polysomnography
            (version 1.0.0)},
  year   = {2019},
  doi    = {10.13026/hmhs-py35},
  note   = {PhysioNet}
}

@article{goldberger2000physionet,
  author  = {Goldberger, A. L. and Amaral, L. A. N. and Glass, L. and others},
  title   = {{PhysioBank, PhysioToolkit, and PhysioNet}},
  journal = {Circulation},
  volume  = {101},
  number  = {23},
  pages   = {e215--e220},
  year    = {2000},
  doi     = {10.1161/01.CIR.101.23.e215}
}
```

## Stage collapse (5-stage -> 3-state)

The source labels follow AASM/R&K codes. They are collapsed to the 3-state scheme of
[`configs/sleepaccel_demo.yaml`](../configs/sleepaccel_demo.yaml):

| Source code | Meaning | StageGuard 3-state |
|---|---|---|
| 0 | Wake | `0` Wake |
| 1 | N1 | `1` NREM |
| 2 | N2 | `1` NREM |
| 3 | N3 | `1` NREM |
| 4 | (R&K stage 4, deep) | `1` NREM |
| 5 | REM | `2` REM |
| -1 | unscored / artifact | `0` Wake |

`-1` is mapped to Wake (not dropped) so that no spurious cross-gap transitions are introduced into the
ground-truth hypnogram. The shipped subjects happen to contain only codes `{0,1,2,3,5}`, but the mapping
handles `4` and `-1` for reproducibility.

## File schema (`sleepaccel_demo.npz`)

Saved with per-subject keys (no `allow_pickle` required on load):

| key | dtype | shape | meaning |
|---|---|---|---|
| `hypno_0` … `hypno_4` | `int64` | `(T_i,)` | collapsed hypnogram per subject; values in `{0,1,2}` |
| `subject_ids` | `<U…` | `(5,)` | PhysioNet subject IDs (order matches `hypno_*`) |
| `lengths` | `int64` | `(5,)` | epoch count `T_i` per subject |
| `epoch_sec` | `float64` | scalar | 30.0 |
| `stage_names` | `<U…` | `(3,)` | `["Wake", "NREM", "REM"]` |

```python
import numpy as np
d = np.load("data/sleepaccel_demo.npz")
hypno = d["hypno_0"].astype(int)          # primary subject, (T,) in {0,1,2}
epoch_sec = float(d["epoch_sec"])         # 30.0
stage_names = list(d["stage_names"])      # ["Wake", "NREM", "REM"]
```

## Disclaimer

This is a small subset for reproducing the notebook only; it is **not** a replacement for the full
Sleep-Accel dataset. For the complete data (including the raw motion/heart-rate signals), download it directly
from PhysioNet under the ODC-By terms.
