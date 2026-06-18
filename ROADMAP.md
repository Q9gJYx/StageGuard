# Roadmap

This file tracks planned work beyond the KDD 2026 camera-ready snapshot (`v1.0.0-kdd2026`). The released
library is the backbone-agnostic StageGuard core (soft transition penalty + semi-Markov decoder + metrics +
SQI) with two reference backbones and a runnable demo.

## Additional backbone adapters

The paper evaluates six backbones; two ship here (`accusleep`, `usleep`). Reference adapters for the
remaining four are planned:

- DeepSleepNet (Supratak et al., 2017)
- SeqSleepNet (Phan et al., 2019)
- AttnSleep (Eldele et al., 2021)
- SleepTransformer (Phan et al., 2022)

Any of these already works today by wrapping the author's own implementation in `StageGuardWrapper`; the only
requirement is a `(B, T, C)` logits output (see [docs/backbones.md](docs/backbones.md)).

## Reproduction harness

An `experiments/` package with per-dataset entry points (`--mode base / +stageguard`) to reproduce the
paper's tables, plus exact-version environment locks and per-run seeds.

## Packaging and distribution

- Publish to PyPI (`pip install stageguard`).
- Archive each tagged release on Zenodo for a citable DOI.

## More modalities and configs

- Additional `configs/*.yaml` for further datasets and stage schemes.
- Optional helper utilities for converting common PSG label formats into the loaders' expected layouts.
