# Dependencies and Attribution

This page records the external work StageGuard builds on. StageGuard's own code (the soft transition penalty,
semi-Markov decoder, metrics, SQI, and wrappers) is original and MIT-licensed; no third-party source code is
vendored in this repository.

## Reimplemented backbone architectures

The two bundled backbones are clean reimplementations of published architectures, not ports of the original
code:

- `accusleep` - compact 1D CNN after Barger et al. (2019), *Robust, automated sleep scoring by a compact
  neural network with distributional shift correction*, PLOS ONE 14(12):e0224642,
  [10.1371/journal.pone.0224642](https://doi.org/10.1371/journal.pone.0224642).
- `usleep` - U-Net encoder-decoder after Perslev et al. (2021), *U-Sleep: resilient high-frequency sleep
  staging*, npj Digital Medicine 4:72.

The paper evaluates six backbones; the other four (DeepSleepNet, SeqSleepNet, AttnSleep, SleepTransformer)
are on the [roadmap](../ROADMAP.md) and can be wrapped today via `StageGuardWrapper` (see
[backbones.md](backbones.md)).

## Datasets and licenses

StageGuard ships no raw data. Each dataset is obtained from its source under its own terms:

| Dataset | Access | License / terms |
|---|---|---|
| AccuSleep (Barger et al. 2019) | [OSF py5eb](https://osf.io/py5eb/) | No redistribution license; the demo downloads it at runtime and re-hosts nothing. |
| Sleep-Accel (Walch et al. 2019) | [PhysioNet](https://physionet.org/content/sleep-accel/1.0.0/) | Open, ODC-By v1.0. |
| SHHS (Quan et al. 1997) | [NSRR](https://sleepdata.org/datasets/shhs) | Data-use agreement required. |
| SLEEPBRL (Tataraidze et al. 2015) | Contact authors | Not publicly redistributable. |

See [datasets.md](datasets.md) for loader formats and [../data/README.md](../data/README.md) for the demo
data provenance and stage-code mapping.

## Summary

| Component | Authored by | Status | License |
|---|---|---|---|
| StageGuard core (penalty, decoder, metrics, SQI, wrapper) | This project | Released here | MIT |
| `accusleep` / `usleep` backbones | Reimplemented from Barger 2019 / Perslev 2021 | Released here, with attribution | MIT (this code) |
| Datasets | External | Obtained from source under their own terms | see table above |

No third-party source code is vendored in this repository.
