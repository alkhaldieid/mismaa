# Phase 4 — Vibration fault diagnosis (CWRU)

4-class bearing-fault classification: normal, inner race, ball, outer race.

**Split:** generalisation across load — train on motor loads 0, 1, 2 (HP), test on the unseen load 3. Holding out a whole operating condition is harder and more honest than a random segment split, which would leak near-duplicate neighbouring windows from the same file into the test set and inflate accuracy.

| Model | Test accuracy | Train segments | Test segments |
|---|---|---|---|
| gradient_boosting | 0.9752 | 4352 | 1534 |
| cnn_1d | 1.0000 | 4352 | 1534 |

## gradient_boosting — confusion matrix

| true \ pred | normal | inner_race | ball | outer_race |
|---|---|---|---|---|
| **normal** | 473 | 0 | 0 | 0 |
| **inner_race** | 0 | 335 | 19 | 0 |
| **ball** | 2 | 1 | 350 | 0 |
| **outer_race** | 0 | 2 | 14 | 338 |

Top features by gradient-boosting importance:

- `rms`: 0.436
- `bpfi_h1_energy_ratio`: 0.243
- `bpfo_h1_energy_ratio`: 0.157
- `kurtosis`: 0.068
- `bsf_h2_energy_ratio`: 0.061
- `bsf_h1_energy_ratio`: 0.017

## cnn_1d — confusion matrix

| true \ pred | normal | inner_race | ball | outer_race |
|---|---|---|---|---|
| **normal** | 473 | 0 | 0 | 0 |
| **inner_race** | 0 | 354 | 0 | 0 |
| **ball** | 0 | 0 | 353 | 0 |
| **outer_race** | 0 | 0 | 0 | 354 |
