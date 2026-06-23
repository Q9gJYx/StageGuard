# [KDD 2026 AI4Sci] StageGuard: Physiologically Constrained Sleep Staging

**Backbone-agnostic physiological constraints for neural sleep staging.**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Q9gJYx/StageGuard/blob/main/notebooks/demo_stageguard.ipynb)
[![KDD 2026](https://img.shields.io/badge/KDD-2026%20AI4Sci-1d4ed8.svg)](https://doi.org/10.1145/3770855.3818916)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3770855.3818916-blue.svg)](https://doi.org/10.1145/3770855.3818916)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org)
<!-- After depositing the release on Zenodo, add the archival DOI badge here:
[![Zenodo](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXXX) -->

Official implementation of the **KDD 2026 AI4Sciences** paper **"StageGuard: Physiologically Constrained
Sleep Staging"** by Juntang Wang, Yihan Wang, Hao Wu, Jiayu Gao, Shixin Xu, and Dongmian Zou (Zu Chongzhi
Center, Duke Kunshan University). Equal contribution: Juntang Wang, Yihan Wang, Hao Wu, Jiayu Gao.
Corresponding author: Dongmian Zou (`dongmian.zou@duke.edu`).

> Automated sleep staging is increasingly used in large-scale studies and downstream scientific analyses to
> derive sleep-architecture endpoints, including total sleep time, REM latency, sleep efficiency, and
> bout-duration statistics. Deep learning models achieve epoch-level accuracy approaching inter-rater
> agreement, yet often produce hypnograms that violate physiological invariants, such as rare state
> transitions (e.g., direct Wake to REM) or excessively fragmented sequences. Such violations can bias
> downstream sleep metrics, regardless of overall accuracy. We propose StageGuard, a plug-and-play,
> backbone-agnostic structured-inference framework that wraps any neural sleep-staging backbone with
> physiology-informed priors. StageGuard combines (1) a differentiable soft transition penalty that
> discourages physiologically rare transitions during training, and (2) a semi-Markov constrained decoder
> with a duration-augmented state space that jointly enforces transition penalties and minimum bout durations
> at inference. Across six backbones and four datasets, StageGuard reduces the transition-violation rate to
> physiologically plausible levels and lowers the fragmentation index by 56 to 62%, while maintaining or
> slightly improving classification accuracy, and improves the accuracy of derived sleep-architecture
> statistics that are not directly optimized by the method.

## News

- **2026** Accepted to KDD 2026 (AI4Sciences track), Jeju Island, Republic of Korea (Aug 9 to 13, 2026).

## Key Contributions

1. **Soft Transition Penalty** - A differentiable loss term that discourages physiologically rare stage transitions (e.g., Wake to/from REM) during training.
2. **Semi-Markov Decoder** - An augmented Viterbi decoder that enforces minimum bout durations, penalizes rare transitions, and suppresses flip-flop artifacts at inference time.
3. **Signal Quality Integration** - Per-epoch signal quality scores modulate decoder confidence, gracefully degrading predictions for noisy segments.

## Installation

```bash
# With conda
conda env create -f environment.yml
conda activate stageguard

# Or with pip
pip install -e .

# Optional: notebook/plotting extras
pip install -e ".[demo]"
```

## Demo

Run the interactive demo on Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Q9gJYx/StageGuard/blob/main/notebooks/demo_stageguard.ipynb)

It **trains the real AccuSleep backbone on real mouse EEG** (downloaded at runtime from the open AccuSleep
dataset), then runs the semi-Markov decoder on a held-out night and shows the transition-violation rate (TVR)
and fragmentation index (FI) drop while accuracy is held or improved, with a before/after hypnogram plot. The
backbone is trained EEG-only, so it confuses Wake and REM and fragments bouts - exactly the errors the decoder
repairs. Expect a ~233 MB download and roughly 3-5 minutes on a Colab CPU. See
[`notebooks/demo_stageguard.ipynb`](notebooks/demo_stageguard.ipynb).

## Quick Start

```python
import torch
from stageguard import StageGuardWrapper, ModalityConfig
from stageguard.backbones import get_backbone

# Load modality config
config = ModalityConfig.from_yaml("configs/mouse_eeg.yaml")

# Create backbone + StageGuard wrapper
backbone = get_backbone("accusleep", num_classes=config.num_classes)
model = StageGuardWrapper(backbone, config)

# Training step
x = torch.randn(4, 50, 1, 128)  # (batch, epochs, channels, samples)
targets = torch.randint(0, 3, (4, 50))
loss, details = model.training_step(x, targets)
loss.backward()

# Inference with constrained decoding
predictions = model.predict(x)  # (4, 50) numpy array
```

## Wrap Your Own Backbone

Any `nn.Module` that outputs `(B, T, num_classes)` logits works:

```python
import torch.nn as nn

class MyBackbone(nn.Module):
    def forward(self, x):
        # Your architecture here
        return logits  # (B, T, C)

model = StageGuardWrapper(MyBackbone(), config)
```

See [`docs/backbones.md`](docs/backbones.md) for the full interface contract and the `ModalityConfig` schema.

## Repository Structure

```
StageGuard/
├── configs/                      # Modality-specific YAML configurations
│   ├── mouse_eeg.yaml            # AccuSleep: 3-state, 4s epochs
│   ├── actigraphy.yaml           # Sleep-Accel: 2-state, 30s epochs
│   ├── cardiorespiratory.yaml    # SHHS: 3-state, 30s epochs
│   ├── bioradar.yaml             # SLEEPBRL: 3-state, 30s epochs
│   └── accusleep_demo.yaml       # 3-state mouse-EEG config used by the demo (2.5s epochs)
├── stageguard/
│   ├── config.py                 # ModalityConfig dataclass + YAML loader
│   ├── losses.py                 # SoftTransitionPenalty + stageguard_loss
│   ├── decoder.py                # SemiMarkovDecoder (augmented Viterbi)
│   ├── wrapper.py                # StageGuardWrapper (backbone + loss + decoder)
│   ├── metrics.py                # TVR, FI, accuracy, kappa, F1, sleep architecture
│   ├── sqi.py                    # Signal quality index per modality
│   ├── backbones/                # Backbone implementations + registry
│   └── data/                     # Dataset loaders (expect pre-downloaded data)
├── notebooks/
│   └── demo_stageguard.ipynb     # Colab-runnable train + decode demo (real mouse EEG)
├── scripts/
│   ├── build_accusleep_demo.py   # reproduce the demo's data step locally (downloads, no data shipped)
│   └── smoke_test.py             # standalone forward / train / predict sanity check
├── data/
│   └── README.md                 # demo-data provenance, license, stage mapping (no raw data shipped)
├── docs/
│   ├── datasets.md               # dataset access + loader formats
│   └── backbones.md              # interface contract + config schema
├── examples/demo.py              # end-to-end synthetic data demo
├── tests/                        # test suite
├── ROADMAP.md
├── CITATION.cff
└── .zenodo.json
```

## Datasets

| Dataset | Modality | Classes | Epoch | Access |
|---------|----------|---------|-------|--------|
| AccuSleep | Mouse EEG/EMG | 3 (Wake, NREM, REM) | 2.5s | Open ([OSF](https://osf.io/py5eb/)) |
| Sleep-Accel | Wrist actigraphy | 2 (Wake, Sleep) | 30s | Open, ODC-By ([PhysioNet](https://physionet.org/content/sleep-accel/1.0.0/)) |
| SHHS | Cardiorespiratory | 3 (Wake, Light, Deep) | 30s | Data-use agreement ([NSRR](https://sleepdata.org/datasets/shhs)) |
| SLEEPBRL | Bioradar | 3 (Wake, Light, Deep) | 30s | Contact authors |

Datasets must be downloaded separately; some require data use agreements. Each loader provides download
instructions via `Dataset.download_instructions()`. See [`docs/datasets.md`](docs/datasets.md) for the
expected on-disk formats. The demo notebook downloads one mouse recording from the open AccuSleep dataset at
runtime and trains on it; no raw data is shipped in this repository (see [`data/README.md`](data/README.md)).

## Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| `lambda_trans` | lambda | 1.0 | Transition penalty weight |
| `epsilon` | epsilon | 5.0 | Rare transition penalty in decoder |
| `gamma` | gamma | 2.0 | Anti-flip-flop penalty |
| `k` | k | 5 | Flip-flop lookback window (epochs) |
| `d_min` | d_min | per-stage | Minimum bout duration (epochs) |
| `d_max` | d_max | 30 | Maximum tracked duration (epochs) |

## Running Tests

```bash
pytest               # Run all tests
pytest -x            # Stop on first failure
pytest tests/test_decoder.py  # Single module
```

## Running the Demo

```bash
python examples/demo.py            # end-to-end on synthetic data
```

## Scripts

```bash
python scripts/build_accusleep_demo.py   # download + transform the demo's mouse-EEG data locally
python scripts/smoke_test.py             # quick forward / train / predict sanity check
```

`build_accusleep_demo.py` writes a local `data/accusleep_demo.npz` (git-ignored); no raw data is committed,
since the AccuSleep recordings carry no redistribution license (see [`data/README.md`](data/README.md)).

## Acknowledgments

We thank the authors of the datasets used in this work: AccuSleep (Barger et al., 2019), Sleep-Accel (Walch
et al., 2019, distributed via PhysioNet under ODC-By), SHHS (Quan et al., 1997, via the NSRR), and SLEEPBRL
(Tataraidze et al., 2015). The reference backbones reimplement architectures from AccuSleep (Barger et al.,
2019) and U-Sleep (Perslev et al., 2021). The demo notebook downloads AccuSleep mouse EEG from its public
[OSF project](https://osf.io/py5eb/) at runtime; that data is used only for the demonstration and is not
redistributed here.

## Citation

If you use StageGuard in your research, please cite:

```bibtex
@inproceedings{wang2026stageguard,
  title     = {StageGuard: Physiologically Constrained Sleep Staging},
  author    = {Wang, Juntang and Wang, Yihan and Wu, Hao and Gao, Jiayu and Xu, Shixin and Zou, Dongmian},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD '26)},
  year      = {2026},
  doi       = {10.1145/3770855.3818916}
}
```

A machine-readable [CITATION.cff](CITATION.cff) is also provided, which GitHub renders into a "Cite this
repository" button on the project page.

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
