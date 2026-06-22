# Backbones and the StageGuard interface

StageGuard is **backbone-agnostic**: it wraps any per-epoch sleep-staging network with a soft transition
penalty (training) and a semi-Markov constrained decoder (inference). This document describes the contract a
backbone must satisfy, the configuration schema, and how to use the decoder, metrics, and SQI standalone.

## The backbone contract

A backbone is any `torch.nn.Module` whose `forward(x)` returns **`(B, T, C)` logits**, where `B` is batch,
`T` is the number of epochs in the sequence, and `C = num_classes`. Input shape is up to the backbone; the
two bundled CNNs accept `(B, T, in_channels, epoch_samples)`.

```python
import torch.nn as nn
from stageguard import StageGuardWrapper, ModalityConfig

class MyBackbone(nn.Module):
    def forward(self, x):
        ...
        return logits          # (B, T, C)

config = ModalityConfig.from_yaml("configs/mouse_eeg.yaml")
model = StageGuardWrapper(MyBackbone(), config)
loss, details = model.training_step(x, targets)   # details: {"ce_loss", "trans_loss"}
predictions = model.predict(x)                     # constrained decode -> (B, T)
```

### Bundled backbones

| Name (registry key) | Architecture | Reference |
|---|---|---|
| `accusleep` | 2-layer 1D CNN | Barger et al. (2019) |
| `usleep` | U-Net encoder-decoder | Perslev et al. (2021) |

```python
from stageguard.backbones import get_backbone, register_backbone
backbone = get_backbone("accusleep", num_classes=3, in_channels=1)
register_backbone("my_net", MyBackbone)     # add your own to the registry
```

The paper evaluates six backbones; the other four (DeepSleepNet, SeqSleepNet, AttnSleep, SleepTransformer)
are on the [roadmap](../ROADMAP.md). Any of them can already be used today by wrapping the author's own
implementation in `StageGuardWrapper`, since the only requirement is `(B, T, C)` output.

## ModalityConfig schema

`ModalityConfig` (see `stageguard/config.py`) holds the stage definitions and constraint parameters. Load
from YAML with `ModalityConfig.from_yaml(path)`. Fields:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `stage_names` | list[str] | (required) | stage labels; `len` must equal `num_classes` |
| `num_classes` | int | (required) | number of stages `C` |
| `rare_transitions` | list[[int,int]] | (required) | `(source, target)` pairs penalized as physiologically rare |
| `lambda_trans` | float | 1.0 | weight of the soft transition penalty in training |
| `d_max` | int | 30 | maximum tracked bout duration (augmented-state size is `C*C*d_max`) |
| `epsilon` | float | 5.0 | decoder penalty for a rare transition |
| `gamma` | float | 2.0 | anti-flip-flop penalty for returning to the previous stage within `k` epochs |
| `k` | int | 5 | flip-flop look-back window (epochs) |
| `d_min` | list[int] | [] | minimum bout duration per stage (epochs); `len` must equal `num_classes` if set |
| `sqi_method` | str | "spectral_entropy" | SQI function name (see `stageguard/sqi.py`) |
| `sqi_threshold` | float | 0.5 | documentary threshold for the modality |
| `epoch_sec` | float | 30.0 | epoch length in seconds |
| `dataset_name`, `dataset_url` | str / None | None | optional metadata |

## Using the components standalone

The decoder, metrics, and SQI functions work without the wrapper (they are imported from submodules, not the
top-level package):

```python
import numpy as np
from stageguard.decoder import SemiMarkovDecoder
from stageguard.metrics import (transition_violation_rate, fragmentation_index,
                                classification_metrics, sleep_architecture)
from stageguard.sqi import compute_sqi

decoder = SemiMarkovDecoder(config)
stages = decoder.decode(log_probs)              # log_probs: (T, C) LOG-probabilities
stages = decoder.decode(log_probs, sqi_scores)  # optional (T,) SQI in [0, 1]
```

Notes:

- `decode` expects **log-probabilities** (e.g. `torch.log_softmax(logits, -1)`), not raw logits.
- `fragmentation_index` returns transitions **per epoch**; multiply by `3600 / epoch_sec` for per-hour.
- `sleep_architecture` reports TST, sleep efficiency, WASO, awakenings, and per-stage mean bout durations.
- `compute_sqi(signal, method=config.sqi_method)` returns a scalar quality score in `[0, 1]`.

See [`notebooks/demo_stageguard.ipynb`](../notebooks/demo_stageguard.ipynb) for an end-to-end example.
