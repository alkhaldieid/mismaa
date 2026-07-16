# Phase 2 — Acoustic anomaly detection

**Method:** DCASE-2020-Task-2-style baseline. Each 10 s MIMII clip (single mic, channel 0) becomes a 64-band log-Mel spectrogram (`configs/dsp.yaml`), sliced into 5-frame context windows. A convolutional autoencoder is trained on **normal clips only** (per-mel-band z-score fit on train); a clip's anomaly score is the mean reconstruction error over its windows. AUC and partial-AUC (max FPR = 0.1, McClish-standardised) are computed with scikit-learn. Higher is better; 0.5 is chance. Test sets are balanced (held-out normal count matched to abnormal), per machine id.

| Machine | id | AUC | pAUC (p=0.1) | Train windows | Test (norm/abn) | Epochs | Time (s) |
|---|---|---|---|---|---|---|---|
| pump | id_00 | 0.6460 | 0.5087 | 134628 | 143/143 | 20 | 60 |
| pump | id_02 | 0.5487 | 0.5568 | 139464 | 111/111 | 20 | 74 |
| pump | id_04 | 0.9905 | 0.9500 | 93912 | 100/100 | 20 | 51 |
| pump | id_06 | 0.7522 | 0.6124 | 145704 | 102/102 | 20 | 77 |
| fan | id_00 | 0.4630 | 0.4963 | 94224 | 407/407 | 20 | 59 |
| fan | id_02 | 0.7411 | 0.5802 | 102492 | 359/359 | 20 | 63 |
| fan | id_04 | 0.5847 | 0.5066 | 106860 | 348/348 | 20 | 64 |
| fan | id_06 | 0.8585 | 0.6672 | 102024 | 361/361 | 20 | 62 |

## Mean per machine type

| Machine | mean AUC | mean pAUC |
|---|---|---|
| pump | 0.7344 | 0.6570 |
| fan | 0.6618 | 0.5626 |
