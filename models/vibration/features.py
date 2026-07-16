"""Per-segment vibration features for the gradient-boosting classifier.

Two physically-motivated feature groups, both computed with the existing
:mod:`dsp` module (no DSP is reimplemented here):

1. **Time-domain health indicators** — RMS (overall energy), excess kurtosis and
   crest factor (both impulsiveness measures). A localised bearing defect injects
   sharp periodic impacts that raise kurtosis/crest well before the RMS level
   moves, so together they separate healthy from faulty bearings regardless of
   fault location.
2. **Envelope-spectrum band energies at the theoretical fault frequencies.** The
   raw impacts are weak but they amplitude-modulate a high-frequency structural
   resonance; band-passing that resonance and taking the Hilbert envelope
   demodulates them so BPFO/BPFI/BSF (and harmonics) appear as spectral peaks
   (see :mod:`models.vibration.faults`). Energy at BPFO/BPFI/BSF is what tells the
   *location* of the fault apart (outer vs inner race vs ball).

Envelope band energies are normalised by total envelope power so they are
amplitude/​load invariant — important for the across-load split, where absolute
vibration levels change with the motor load.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dsp import envelope_spectrum, frame_crest_factor, frame_kurtosis, frame_rms

from .faults import BearingGeometry, fault_frequencies

# The three fault frequencies whose energy discriminates fault *location*.
_FAULT_KEYS = ("BPFO", "BPFI", "BSF")


def _scalar_stat(fn, segment: np.ndarray) -> float:
    """Evaluate a per-frame dsp statistic over the whole segment (single frame)."""
    n = segment.shape[0]
    return float(fn(segment, frame_length=n, hop_length=n)[0])


def _band_energy(freqs: np.ndarray, power: np.ndarray, center: float, halfwidth: float) -> float:
    """Total envelope power within ``center +/- halfwidth`` Hz."""
    mask = (freqs >= center - halfwidth) & (freqs <= center + halfwidth)
    return float(power[mask].sum())


def feature_names(feat_cfg: dict[str, Any]) -> list[str]:
    """Ordered feature names matching :func:`segment_features` output."""
    n_harm = int(feat_cfg["n_harmonics"])
    names = ["rms", "kurtosis", "crest_factor"]
    for key in _FAULT_KEYS:
        for h in range(1, n_harm + 1):
            names.append(f"{key.lower()}_h{h}_energy_ratio")
    return names


def segment_features(
    segment: np.ndarray,
    rpm: float,
    geometry: BearingGeometry,
    sample_rate: int,
    feat_cfg: dict[str, Any],
) -> np.ndarray:
    """Extract the feature vector for one raw acceleration segment.

    Args:
        segment: 1-D segment.
        rpm: Measured shaft speed for this segment's source file.
        geometry: Bearing geometry (for the fault frequencies).
        sample_rate: Hz.
        feat_cfg: The ``features`` block of ``configs/vibration.yaml``.

    Returns:
        1-D ``float64`` feature vector ordered as :func:`feature_names`.
    """
    n_harm = int(feat_cfg["n_harmonics"])
    halfwidth = float(feat_cfg["band_halfwidth_hz"])
    band = (
        float(feat_cfg["resonance_band_low_hz"]),
        float(feat_cfg["resonance_band_high_hz"]),
    )
    order = int(feat_cfg["bandpass_order"])

    feats = [
        _scalar_stat(frame_rms, segment),
        _scalar_stat(frame_kurtosis, segment),
        _scalar_stat(frame_crest_factor, segment),
    ]

    freqs, mag = envelope_spectrum(
        segment, sample_rate=sample_rate, band=band, order=order, zero_phase=True
    )
    power = mag**2
    total = float(power.sum()) + 1e-12  # normaliser -> amplitude/load invariance

    ff = fault_frequencies(geometry, rpm)
    for key in _FAULT_KEYS:
        base = ff[key]
        for h in range(1, n_harm + 1):
            energy = _band_energy(freqs, power, base * h, halfwidth)
            feats.append(energy / total)
    return np.asarray(feats, dtype=np.float64)


def build_feature_matrix(
    segments: np.ndarray,
    rpms: np.ndarray,
    geometry: BearingGeometry,
    sample_rate: int,
    feat_cfg: dict[str, Any],
) -> np.ndarray:
    """Stack :func:`segment_features` over every segment into a design matrix.

    Args:
        segments: ``(n_segments, segment_length)``.
        rpms: ``(n_segments,)`` shaft speeds.
        geometry, sample_rate, feat_cfg: See :func:`segment_features`.

    Returns:
        ``(n_segments, n_features)`` float64 matrix.
    """
    rows = [
        segment_features(segments[i], float(rpms[i]), geometry, sample_rate, feat_cfg)
        for i in range(segments.shape[0])
    ]
    if not rows:
        return np.empty((0, len(feature_names(feat_cfg))), dtype=np.float64)
    return np.vstack(rows)
