"""Mismaa modelling package.

Two model families live here:

* Phase 2 — acoustic anomaly detection (:mod:`models.dataset`,
  :mod:`models.autoencoder`, :mod:`models.train`, :mod:`models.evaluate`):
  a DCASE-2020-Task-2-style convolutional autoencoder trained on *normal*
  MIMII clips only and scored by reconstruction error.
* Phase 4 — vibration fault diagnosis (:mod:`models.vibration`): a
  feature-based gradient-boosting classifier and a raw-signal 1D-CNN over
  CWRU bearing records.

Both phases consume features from the existing :mod:`dsp` package; no DSP is
reimplemented here.
"""

from __future__ import annotations
