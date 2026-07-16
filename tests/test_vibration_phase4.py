"""Unit tests for the Phase 4 vibration fault-diagnosis pipeline.

The fault-frequency tests are the load-bearing ones: they pin the SKF 6205
formulas to independently hand-computed values and to the documented CWRU
per-revolution multipliers. The rest are fast shape/wiring checks on synthetic
signals (no real .mat files, no training).
"""

from __future__ import annotations

import numpy as np
import pytest

from models.vibration.cnn import Cnn1D
from models.vibration.dataset import CLASS_NAMES, CLASS_TO_IDX, segment_signal
from models.vibration.faults import (
    BearingGeometry,
    bpfi,
    bpfo,
    bsf,
    fault_frequencies,
    ftf,
    geometry_from_config,
    shaft_frequency,
)
from models.vibration.features import build_feature_matrix, feature_names, segment_features

# SKF 6205-2RS JEM drive-end bearing geometry (documented manufacturer values).
SKF_6205 = BearingGeometry(
    n_balls=9, ball_diameter=0.3126, pitch_diameter=1.537, contact_angle_deg=0.0
)

FEAT_CFG = {
    "n_harmonics": 3,
    "band_halfwidth_hz": 15.0,
    "resonance_band_low_hz": 500.0,
    "resonance_band_high_hz": 5000.0,
    "bandpass_order": 4,
}


# ---------------------------------------------------------------------------
# Fault frequencies vs hand-computed values and documented CWRU multipliers
# ---------------------------------------------------------------------------


def test_shaft_frequency():
    assert shaft_frequency(1797) == pytest.approx(29.95)


def test_fault_frequencies_hand_computed_at_1797_rpm():
    rpm = 1797  # motor load 0 nominal speed
    # Hand-computed (Hz): see models/vibration/faults.py derivation.
    assert bpfo(SKF_6205, rpm) == pytest.approx(107.36, abs=0.02)
    assert bpfi(SKF_6205, rpm) == pytest.approx(162.19, abs=0.02)
    assert bsf(SKF_6205, rpm) == pytest.approx(70.59, abs=0.02)
    assert ftf(SKF_6205, rpm) == pytest.approx(11.93, abs=0.02)


def test_fault_frequency_multipliers_match_documented_cwru_values():
    # Dividing out shaft frequency must recover the published SKF 6205 multipliers,
    # independent of RPM.
    rpm = 1730
    fr = shaft_frequency(rpm)
    assert bpfo(SKF_6205, rpm) / fr == pytest.approx(3.5848, abs=1e-3)
    assert bpfi(SKF_6205, rpm) / fr == pytest.approx(5.4152, abs=1e-3)
    assert bsf(SKF_6205, rpm) / fr == pytest.approx(2.3568, abs=1e-3)
    assert ftf(SKF_6205, rpm) / fr == pytest.approx(0.3983, abs=1e-3)


def test_fault_frequencies_physical_ordering():
    ff = fault_frequencies(SKF_6205, 1750)
    # Inner-race pass rate exceeds outer-race, which exceeds ball-spin, exceeds cage.
    assert ff["BPFI"] > ff["BPFO"] > ff["BSF"] > ff["FTF"]


def test_fault_frequencies_scale_linearly_with_rpm():
    a = fault_frequencies(SKF_6205, 1000)
    b = fault_frequencies(SKF_6205, 2000)
    for key in a:
        assert b[key] == pytest.approx(2.0 * a[key])


def test_geometry_from_config():
    cfg = {
        "bearing": {
            "n_balls": 9,
            "ball_diameter": 0.3126,
            "pitch_diameter": 1.537,
            "contact_angle_deg": 0.0,
        }
    }
    g = geometry_from_config(cfg)
    assert g.n_balls == 9
    assert g.ball_diameter == pytest.approx(0.3126)
    assert g.pitch_diameter == pytest.approx(1.537)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def test_segment_signal_shape_and_overlap():
    sig = np.arange(10000, dtype=np.float64)
    segs = segment_signal(sig, length=2048, overlap=0.5)  # hop = 1024
    expected = 1 + (10000 - 2048) // 1024
    assert segs.shape == (expected, 2048)
    # 50% overlap -> segment 1 starts 1024 samples after segment 0.
    assert segs[1, 0] == pytest.approx(1024.0)


def test_segment_signal_too_short_returns_empty():
    assert segment_signal(np.zeros(1000), length=2048, overlap=0.5).shape == (0, 2048)


def test_segment_signal_rejects_bad_overlap():
    with pytest.raises(ValueError):
        segment_signal(np.zeros(4096), length=2048, overlap=1.0)


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def test_feature_names_length():
    names = feature_names(FEAT_CFG)
    # 3 time-domain + 3 fault types x n_harmonics envelope-energy features.
    assert len(names) == 3 + 3 * FEAT_CFG["n_harmonics"]
    assert names[:3] == ["rms", "kurtosis", "crest_factor"]


def test_segment_features_vector_matches_names():
    rng = np.random.default_rng(0)
    seg = rng.standard_normal(2048)
    vec = segment_features(seg, rpm=1797, geometry=SKF_6205, sample_rate=12000, feat_cfg=FEAT_CFG)
    assert vec.shape == (len(feature_names(FEAT_CFG)),)
    assert np.all(np.isfinite(vec))


def test_build_feature_matrix_shape():
    rng = np.random.default_rng(1)
    segs = rng.standard_normal((5, 2048))
    rpms = np.full(5, 1797.0)
    x = build_feature_matrix(segs, rpms, SKF_6205, 12000, FEAT_CFG)
    assert x.shape == (5, len(feature_names(FEAT_CFG)))


def test_envelope_energy_peaks_at_injected_fault_frequency():
    # Synthesise an outer-race-like signal: a carrier resonance amplitude-modulated
    # at BPFO. Its BPFO envelope-energy feature should dominate the BPFI/BSF ones.
    sr = 12000
    n = 4096
    t = np.arange(n) / sr
    f_bpfo = bpfo(SKF_6205, 1797)
    carrier = np.sin(2 * np.pi * 3000 * t)  # inside the 500-5000 Hz resonance band
    modulation = 1.0 + np.sin(2 * np.pi * f_bpfo * t)
    seg = carrier * modulation
    vec = segment_features(seg, 1797, SKF_6205, sr, FEAT_CFG)
    names = feature_names(FEAT_CFG)
    idx = {name: i for i, name in enumerate(names)}
    assert vec[idx["bpfo_h1_energy_ratio"]] > vec[idx["bpfi_h1_energy_ratio"]]
    assert vec[idx["bpfo_h1_energy_ratio"]] > vec[idx["bsf_h1_energy_ratio"]]


# ---------------------------------------------------------------------------
# Class map + CNN wiring
# ---------------------------------------------------------------------------


def test_class_map_is_stable():
    assert CLASS_NAMES == ("normal", "inner_race", "ball", "outer_race")
    assert CLASS_TO_IDX == {"normal": 0, "inner_race": 1, "ball": 2, "outer_race": 3}


def test_cnn_forward_shape():
    import torch

    model = Cnn1D(n_classes=4, base_channels=8, dropout=0.3)
    out = model(torch.randn(3, 1, 2048))
    assert out.shape == (3, 4)
