# Mismaa session log — Phase 2 & Phase 4

Date: 2026-07-16. Autonomous build session. Starting point: Phase 1 (DSP module,
29 tests, tag `v0.1-dsp`). Ending point: Phase 2 and Phase 4 complete, tagged
`v0.2-baseline` and `v0.4-vibration`, 62 tests green, ruff clean, all pushed.

## What was built

### Phase 2 — acoustic anomaly detection (`models/`)
DCASE-2020-Task-2-style convolutional autoencoder over MIMII audio.

- `models/dataset.py` — MIMII indexing, log-Mel via the existing `dsp` module,
  5-frame context windows, DCASE train/test split (train **normal only**; test =
  held-out normal + all abnormal, per machine id), per-mel z-score, gitignored
  disk cache (`data/cache/`).
- `models/autoencoder.py` — convolutional AE; striding on the mel axis only,
  linear (dB) output, config-driven bottleneck.
- `models/evaluate.py` — mean-recon-error clip scoring, AUC + pAUC (sklearn,
  `max_fpr=0.1`, matching the DCASE baseline).
- `models/train.py` — config-driven trainer (MPS/CPU, seeds), MLflow logging,
  `results/phase2.md` writer.
- `configs/train.yaml`, `tests/test_models_phase2.py` (indexing, framing shape,
  scoring monotonicity, AUC vs sklearn + hand-computed, standardization).

### Phase 4 — vibration fault diagnosis (`models/vibration/`)
Two models on CWRU bearing data, 4-class (normal / inner / ball / outer race).

- `faults.py` — SKF 6205 BPFO/BPFI/BSF/FTF formulas (documented derivation);
  geometry read from `configs/vibration.yaml`.
- `dataset.py` — CWRU indexing, 2048-sample / 50%-overlap segmentation via
  `dsp.frame_signal`, **across-load split** (train loads 0/1/2, test load 3).
- `features.py` — per-segment RMS / kurtosis / crest + envelope-spectrum band
  energies at the fault frequencies, normalised by total envelope power.
- `cnn.py` — 1D-CNN over raw segments (wide first kernel, global-average-pool head).
- `train.py` — sklearn GradientBoosting + the CNN, accuracy + confusion matrices,
  MLflow, `results/phase4.md` writer.
- `configs/vibration.yaml`, `tests/test_vibration_phase4.py` (fault frequencies vs
  hand-computed values and documented CWRU multipliers, segmentation, feature
  shapes, an injected-fault envelope-energy check, CNN wiring).

## Key numbers

**Phase 2 (AUC / pAUC per machine id, 20 epochs, MPS):**

| | id_00 | id_02 | id_04 | id_06 | mean |
|---|---|---|---|---|---|
| pump AUC | 0.646 | 0.549 | 0.990 | 0.752 | **0.734** |
| pump pAUC | 0.509 | 0.557 | 0.950 | 0.612 | **0.657** |
| fan AUC | 0.463 | 0.741 | 0.585 | 0.859 | **0.662** |
| fan pAUC | 0.496 | 0.580 | 0.507 | 0.667 | **0.563** |

Comparable to the published DCASE 2020 Task 2 baseline (pump ~0.72, fan ~0.65).
Total training ~8.5 min for all 8 machine ids (within the ≤30 min/type budget).

**Phase 4 (accuracy on held-out load 3):** gradient boosting **97.5%**,
1D-CNN **100%**. GB feature importances are physically sensible (RMS, then BPFI
and BPFO envelope energies).

## Decisions & rationale

- **Per-mel standardization was essential.** Raw log-Mel dB fed to the AE gave an
  *inverted* AUC (0.25 on pump/id_00) because reconstruction MSE was dominated by
  the large per-band level offsets, not spectral shape. Per-mel z-score (fit on
  train only) fixed it (→0.66). Added as `features.standardize`.
- **Context-window hop = 4, not 1.** The DCASE baseline slides one window per
  frame; that is ~4× the windows and compute. Hop 4 keeps a 5-frame context with
  light overlap and lands the full 8-machine run inside the laptop budget. A
  documented, principled reduction (config knob `features.context_hop`).
- **Across-load split for Phase 4.** Train loads 0/1/2, test load 3. A random
  segment split leaks near-duplicate neighbouring windows from the same file into
  test and inflates accuracy; holding out a whole operating condition tests real
  generalisation.
- **Envelope band energies normalised by total envelope power** so they are
  amplitude/load invariant — important for the across-load split.
- **MLflow file store.** MLflow 3.x gates `./mlruns` behind
  `MLFLOW_ALLOW_FILE_STORE=true` (set in `models/common.mlflow_setup`); kept the
  repo-local store the brief asked for, no server.
- **No dependencies added** — torch, sklearn, mlflow were already pinned.

## Open questions / caveats

- **Phase 4 numbers are near-perfect because CWRU 4-class is a famously separable
  benchmark**, not because the models are extraordinary. The honest metric is the
  held-out-load test (load 3); the CNN's per-epoch `val_acc` hits 1.0 by epoch 2
  but that val slice is random *train* segments, so it suffers file-level leakage
  and is optimistic — do not read it as the generalisation score. A genuinely
  harder task would be per-diameter (10-class) or cross-dataset transfer.
- **Phase 2 weak spots:** fan/id_00 (0.463, below chance) and pump/id_02 (0.549)
  are the known-hard machines and drag the means down. Worth trying before calling
  it done: context_hop=1 (more data), more epochs, a smaller bottleneck, or
  per-machine threshold/architecture tuning. The current run is a faithful
  *baseline*, deliberately un-tuned per machine.
- **MPS determinism:** seeds are set, but MPS kernels are not bit-reproducible, so
  reruns can vary by ~±0.01 AUC.

## Reproduce

```sh
make test                                             # 62 tests
python -m models.train --machine-type all --machine-id all --results-out results/phase2.md
python -m models.vibration.train --models both        # writes results/phase4.md
# feature caches land in data/cache/ (gitignored); MLflow runs in ./mlruns
```

## Suggested next steps

1. Phase 3 (streaming/serving) — wire the trained AE + classifier behind the
   `streaming/` and `serving/` stubs.
2. Phase 2 tuning pass (hop=1, bottleneck sweep) if higher AUC is wanted.
3. Harder vibration task (per-diameter or cross-dataset) to stress the models.
4. Persist trained model weights (currently re-trained each run; only features
   are cached).
