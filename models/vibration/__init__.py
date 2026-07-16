"""Phase 4 — CWRU bearing vibration fault diagnosis.

Two models over the same segmentation of the CWRU 12 kHz drive-end records:

* a **feature-based gradient-boosting** classifier over physically-motivated
  per-segment features (time-domain health indicators + envelope-spectrum energy
  at the theoretical bearing fault frequencies), and
* a **1D-CNN** over the raw acceleration segments.

Both are evaluated on a *generalisation-across-load* split (train on motor loads
0/1/2, test on the unseen load 3). See :mod:`models.vibration.faults` for the
fault-frequency derivation.
"""

from __future__ import annotations
