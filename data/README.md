# Demo data

**No raw data is shipped in this repository.** The demo notebook
([`notebooks/demo_stageguard.ipynb`](../notebooks/demo_stageguard.ipynb)) and the build script
([`scripts/build_accusleep_demo.py`](../scripts/build_accusleep_demo.py)) **download** a single mouse
recording from the public AccuSleep dataset at runtime, into a local `accusleep_data/` directory that is
git-ignored.

## Source: AccuSleep mouse EEG/EMG

The demo trains on the open **AccuSleep** dataset (Barger et al. 2019): mouse EEG/EMG recorded at **512 Hz**
with expert sleep-stage labels at a **2.5 s** epoch resolution.

- Project: <https://github.com/zekebarger/AccuSleep>
- Data: OSF project [`py5eb`](https://osf.io/py5eb/) (`matlab_format/4-hour_recordings/Mouse0N.zip`)
- Paper: Barger Z., Frye C. G., Liu D., Dan Y., Bouchard K. E. (2019). *Robust, automated sleep scoring by a
  compact neural network with distributional shift correction.* PLOS ONE 14(12):e0224642.
  DOI [10.1371/journal.pone.0224642](https://doi.org/10.1371/journal.pone.0224642).

**License note.** The OSF project does not declare a redistribution license, so this repository does not
bundle or re-host any of its recordings. The notebook downloads the data directly from OSF, exactly as a user
would, and uses it only for the demonstration.

## What the demo does with it

It downloads one mouse (`Mouse03`, ~233 MB), which contains five nightly 4-hour sessions
(`Day1`-`Day5`), each with `EEG.mat`, `EMG.mat`, and `labels.mat`. The demo:

1. loads the **EEG only** (the AccuSleep backbone is trained single-channel),
2. anti-aliases and downsamples 512 -> 128 Hz, then cuts the signal into 2.5 s epochs (320 samples each),
3. maps the AccuSleep stage codes to the StageGuard convention,
4. trains on four nights (`Day2`-`Day5`) and evaluates on the held-out night (`Day1`).

### Stage-code mapping

| AccuSleep code | Stage | StageGuard label |
|----------------|-------|------------------|
| 1              | REM   | 2                |
| 2              | Wake  | 0                |
| 3              | NREM  | 1                |

(`Mouse03` is fully scored; there are no `undefined` epochs to drop.)

## Reproduce locally

```bash
python scripts/build_accusleep_demo.py
```

This performs the same download + transform and writes a local `data/accusleep_demo.npz`
(git-ignored) in the format expected by `stageguard.data.AccuSleepDataset`. Nothing it produces is committed.

## Attribution

```bibtex
@article{barger2019accusleep,
  title   = {Robust, automated sleep scoring by a compact neural network with distributional shift correction},
  author  = {Barger, Zeke and Frye, Charles G. and Liu, Danqian and Dan, Yang and Bouchard, Kristofer E.},
  journal = {PLOS ONE},
  volume  = {14},
  number  = {12},
  pages   = {e0224642},
  year    = {2019},
  doi     = {10.1371/journal.pone.0224642}
}
```
