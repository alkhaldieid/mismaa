"""Unit tests for the Phase 2 acoustic anomaly-detection pipeline.

Fast by construction: indexing hits the on-disk MIMII tree (a cheap glob), and
everything else runs on tiny synthetic arrays / a toy network. No full feature
extraction or real training happens here.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from models.dataset import (
    LABEL_ABNORMAL,
    LABEL_NORMAL,
    assign_split,
    context_windows,
    index_machine,
    standardize_apply,
    standardize_fit,
)
from models.evaluate import clip_scores, compute_auc, compute_pauc

DATA_ROOT = "data/mimii"


# ---------------------------------------------------------------------------
# Dataset indexing
# ---------------------------------------------------------------------------


def test_index_machine_labels_and_counts():
    clips = index_machine(DATA_ROOT, "pump", "id_00")
    assert len(clips) > 0
    # Every clip is labelled by its parent folder and carries the right metadata.
    assert {c.label for c in clips} == {LABEL_NORMAL, LABEL_ABNORMAL}
    assert all(c.machine_type == "pump" and c.machine_id == "id_00" for c in clips)
    assert all(c.path.endswith(".wav") for c in clips)
    # Normals are listed before abnormals (deterministic ordering).
    labels = [c.label for c in clips]
    assert labels == sorted(labels)


def test_index_machine_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        index_machine(DATA_ROOT, "pump", "id_99")


def test_assign_split_dcase_convention():
    clips = index_machine(DATA_ROOT, "pump", "id_00")
    split_cfg = {
        "seed": 1234,
        "test_normal_mode": "match_abnormal",
        "test_normal_fraction": 0.25,
    }
    out = assign_split(clips, split_cfg)

    train = [c for c in out if c.split == "train"]
    test = [c for c in out if c.split == "test"]
    # Train is normal-only (the DCASE 2020 Task 2 rule).
    assert all(c.label == LABEL_NORMAL for c in train)
    # Every abnormal clip is in the test set.
    n_abn = sum(1 for c in clips if c.label == LABEL_ABNORMAL)
    assert sum(1 for c in test if c.label == LABEL_ABNORMAL) == n_abn
    # match_abnormal holds out exactly n_abn normal clips for a balanced test.
    assert sum(1 for c in test if c.label == LABEL_NORMAL) == n_abn
    # No clip is dropped or duplicated.
    assert len(train) + len(test) == len(clips)


def test_assign_split_is_reproducible():
    clips = index_machine(DATA_ROOT, "pump", "id_00")
    cfg = {"seed": 7, "test_normal_mode": "fraction", "test_normal_fraction": 0.2}
    a = {c.path: c.split for c in assign_split(clips, cfg)}
    b = {c.path: c.split for c in assign_split(clips, cfg)}
    assert a == b


# ---------------------------------------------------------------------------
# Context-window framing
# ---------------------------------------------------------------------------


def test_context_windows_shape_and_values():
    n_mels, n_time, n_frames, hop = 64, 20, 5, 4
    log_mel = np.arange(n_mels * n_time, dtype=np.float64).reshape(n_mels, n_time)
    w = context_windows(log_mel, n_frames, hop)

    expected_windows = 1 + (n_time - n_frames) // hop
    assert w.shape == (expected_windows, n_mels, n_frames)
    # Window 0 is the first n_frames columns; window 1 starts hop columns later.
    np.testing.assert_array_equal(w[0], log_mel[:, 0:n_frames])
    np.testing.assert_array_equal(w[1], log_mel[:, hop : hop + n_frames])


def test_context_windows_too_short_returns_empty():
    log_mel = np.zeros((64, 3))
    w = context_windows(log_mel, n_frames=5, hop=1)
    assert w.shape == (0, 64, 5)


def test_context_windows_hop_one_is_dense():
    log_mel = np.zeros((8, 10))
    w = context_windows(log_mel, n_frames=5, hop=1)
    assert w.shape[0] == 10 - 5 + 1


# ---------------------------------------------------------------------------
# Scoring: monotonicity of the clip aggregate
# ---------------------------------------------------------------------------


def test_clip_scores_mean_aggregate():
    # Two clips: windows 0,1 -> clip 0; windows 2,3,4 -> clip 1.
    errors = np.array([1.0, 3.0, 2.0, 4.0, 6.0])
    window_clip = np.array([0, 0, 1, 1, 1])
    scores = clip_scores(errors, window_clip, n_clips=2)
    np.testing.assert_allclose(scores, [2.0, 4.0])


def test_clip_scores_monotonic_in_reconstruction_error():
    window_clip = np.array([0, 0, 1, 1])
    base = np.array([1.0, 1.0, 1.0, 1.0])
    higher = base.copy()
    higher[2:] += 5.0  # raise clip 1's window errors only
    s_base = clip_scores(base, window_clip, 2)
    s_higher = clip_scores(higher, window_clip, 2)
    # Clip 1's score strictly increases; clip 0 unchanged -> ranking is monotone.
    assert s_higher[1] > s_base[1]
    assert s_higher[0] == s_base[0]


def test_clip_scores_empty_clip_scores_zero():
    scores = clip_scores(np.array([2.0, 4.0]), np.array([0, 0]), n_clips=3)
    assert scores[2] == 0.0


# ---------------------------------------------------------------------------
# AUC / pAUC against sklearn and a hand-computed value
# ---------------------------------------------------------------------------


def test_compute_auc_matches_hand_computed_and_sklearn():
    y = np.array([0, 0, 1, 1, 1])
    scores = np.array([0.2, 0.5, 0.1, 0.6, 0.9])
    # Mann-Whitney: fraction of (pos, neg) pairs with pos ranked above neg = 4/6.
    assert compute_auc(y, scores) == pytest.approx(4.0 / 6.0)
    assert compute_auc(y, scores) == pytest.approx(roc_auc_score(y, scores))


def test_compute_auc_perfect_ranking():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.9])
    assert compute_auc(y, scores) == pytest.approx(1.0)


def test_compute_pauc_matches_sklearn_maxfpr():
    rng = np.random.default_rng(0)
    y = np.array([0] * 20 + [1] * 20)
    scores = np.concatenate([rng.normal(0, 1, 20), rng.normal(1.5, 1, 20)])
    assert compute_pauc(y, scores, p=0.1) == pytest.approx(
        roc_auc_score(y, scores, max_fpr=0.1)
    )
    # Perfectly separable -> pAUC saturates at 1.0.
    perfect = np.array([0, 0, 1, 1])
    assert compute_pauc(perfect, np.array([0.0, 0.1, 0.8, 0.9]), p=0.1) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Feature standardization (per mel band, train-fit)
# ---------------------------------------------------------------------------


def test_standardize_fit_apply_zero_mean_unit_std_per_band():
    rng = np.random.default_rng(0)
    # Distinct per-band levels and scales, like real log-Mel dB bands.
    train = rng.normal(loc=0.0, scale=1.0, size=(200, 8, 5)).astype(np.float32)
    train += np.arange(8, dtype=np.float32)[None, :, None] * 10.0  # per-band offset
    mean, std = standardize_fit(train)
    assert mean.shape == (1, 8, 1)
    assert std.shape == (1, 8, 1)
    z = standardize_apply(train, mean, std)
    # Each band standardised to ~0 mean, ~1 std across windows and frames.
    np.testing.assert_allclose(z.mean(axis=(0, 2)), 0.0, atol=1e-4)
    np.testing.assert_allclose(z.std(axis=(0, 2)), 1.0, atol=1e-2)


def test_standardize_apply_empty_is_noop():
    empty = np.empty((0, 8, 5), dtype=np.float32)
    mean, std = np.zeros((1, 8, 1), np.float32), np.ones((1, 8, 1), np.float32)
    assert standardize_apply(empty, mean, std).shape == (0, 8, 5)


# ---------------------------------------------------------------------------
# Model wiring (tiny, CPU): shape round-trip + scoring integration
# ---------------------------------------------------------------------------


def test_autoencoder_round_trip_shape():
    from models.autoencoder import ConvAutoencoder

    model = ConvAutoencoder(n_mels=64, n_frames=5, bottleneck=8, base_channels=4)
    import torch

    x = torch.randn(3, 1, 64, 5)
    out = model(x)
    assert out.shape == x.shape


def test_autoencoder_rejects_bad_n_mels():
    from models.autoencoder import ConvAutoencoder

    with pytest.raises(ValueError):
        ConvAutoencoder(n_mels=60, n_frames=5, bottleneck=8, base_channels=4)


def test_reconstruction_error_is_nonnegative_and_perwindow():
    from models.autoencoder import ConvAutoencoder
    from models.evaluate import reconstruction_error_per_window

    model = ConvAutoencoder(n_mels=64, n_frames=5, bottleneck=8, base_channels=4)
    x = np.random.default_rng(0).standard_normal((7, 64, 5)).astype(np.float32)
    err = reconstruction_error_per_window(model, x, device="cpu", batch_size=4)
    assert err.shape == (7,)
    assert np.all(err >= 0.0)
