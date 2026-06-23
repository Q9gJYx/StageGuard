"""Standalone smoke test for the StageGuard demo pipeline.

Exercises the real training step (cross-entropy + soft transition penalty) and the
constrained ``predict`` path of ``StageGuardWrapper`` with the AccuSleep backbone and
the demo config. Uses ``data/accusleep_demo.npz`` if present (build it with
``scripts/build_accusleep_demo.py``); otherwise it synthesizes tiny fake epochs so the
test runs with no network access.

Usage:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import os

import numpy as np
import torch

from stageguard.backbones import get_backbone
from stageguard.config import ModalityConfig
from stageguard.wrapper import StageGuardWrapper

NPZ = os.path.join("data", "accusleep_demo.npz")
SPE = 320
T = 20


def load_or_synth() -> tuple[np.ndarray, np.ndarray]:
    if os.path.exists(NPZ):
        d = np.load(NPZ)
        eeg, labels = d["eeg"].astype(np.float32), d["labels"].astype(np.int64)
        print(f"loaded {NPZ}: eeg{eeg.shape} labels{labels.shape}")
    else:
        rng = np.random.default_rng(0)
        n = 400
        labels = rng.integers(0, 3, size=n).astype(np.int64)
        eeg = rng.standard_normal((n, SPE)).astype(np.float32)
        print(f"{NPZ} not found; using synthetic epochs eeg{eeg.shape}")
    return eeg, labels


def main() -> None:
    cfg = ModalityConfig.from_yaml(os.path.join("configs", "accusleep_demo.yaml"))
    eeg, labels = load_or_synth()
    spe = eeg.shape[1]
    nw = len(labels) // T
    assert nw >= 4, "need at least 4 windows for the smoke test"
    x = torch.tensor(eeg[: nw * T].reshape(nw, T, 1, spe))
    y = torch.tensor(labels[: nw * T].reshape(nw, T))

    model = StageGuardWrapper(
        get_backbone("accusleep", num_classes=cfg.num_classes, in_channels=1), cfg
    )

    loss, details = model.training_step(x[:4], y[:4])
    assert loss.requires_grad and torch.isfinite(loss), "training_step loss invalid"
    loss.backward()
    print(f"training_step OK: total={loss.item():.4f} "
          f"ce={details['ce_loss'].item():.4f} trans={details['trans_loss'].item():.4f}")

    pred = model.predict(x[:4])
    assert pred.shape == (4, T), f"predict shape {pred.shape}"
    assert pred.min() >= 0 and pred.max() < cfg.num_classes, "predict labels out of range"
    print(f"predict OK: shape={pred.shape} labels in [0, {cfg.num_classes - 1}]")
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
