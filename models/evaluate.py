"""Scoring and evaluation for the Phase 2 acoustic anomaly detector.

The anomaly score follows the DCASE 2020 Task 2 baseline exactly: a clip's score
is the **mean reconstruction error** of its context windows. Higher error =>
more anomalous, because the autoencoder was trained to reconstruct only normal
audio and generalises poorly to unseen anomalies.

AUC and partial-AUC are computed with scikit-learn so the numbers are directly
comparable to the published baseline (which uses the same functions).
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def clip_scores(
    window_errors: np.ndarray, window_clip: np.ndarray, n_clips: int
) -> np.ndarray:
    """Aggregate per-window reconstruction errors into one score per clip.

    The score is the arithmetic mean of a clip's window errors — a monotonic
    aggregate, so raising any window's error can only raise (never lower) that
    clip's score. This is the property the anomaly ranking relies on.

    Args:
        window_errors: ``(n_windows,)`` reconstruction error per window.
        window_clip: ``(n_windows,)`` clip index each window belongs to.
        n_clips: Total number of clips (defines the output length).

    Returns:
        ``(n_clips,)`` mean error per clip. Clips with no windows score 0.
    """
    window_errors = np.asarray(window_errors, dtype=np.float64)
    window_clip = np.asarray(window_clip, dtype=np.int64)
    totals = np.bincount(window_clip, weights=window_errors, minlength=n_clips)
    counts = np.bincount(window_clip, minlength=n_clips)
    scores = np.zeros(n_clips, dtype=np.float64)
    nonzero = counts > 0
    scores[nonzero] = totals[nonzero] / counts[nonzero]
    return scores


def compute_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Area under the ROC curve (anomaly = positive class)."""
    return float(roc_auc_score(y_true, scores))


def compute_pauc(y_true: np.ndarray, scores: np.ndarray, p: float = 0.1) -> float:
    """Partial AUC over the low false-positive-rate range ``[0, p]``.

    Uses scikit-learn's ``max_fpr`` (McClish-standardised), matching the DCASE
    2020 Task 2 baseline. pAUC rewards detectors that rank true anomalies above
    normal clips *while keeping the false-alarm rate low* — the operating regime
    that matters for condition monitoring, where false alarms are expensive.
    """
    return float(roc_auc_score(y_true, scores, max_fpr=p))


def reconstruction_error_per_window(
    model,
    x: np.ndarray,
    *,
    device: str = "cpu",
    batch_size: int = 512,
) -> np.ndarray:
    """Mean-squared reconstruction error for every window in ``x``.

    Args:
        model: A trained autoencoder ``nn.Module`` mapping ``(B, 1, n_mels,
            n_frames)`` to the same shape.
        x: ``(N, n_mels, n_frames)`` float array of context windows.
        device: Torch device to run on.
        batch_size: Inference batch size.

    Returns:
        ``(N,)`` per-window MSE (mean over the mel x frame elements).
    """
    import torch

    model.eval()
    errors = np.empty(x.shape[0], dtype=np.float64)
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = x[start : start + batch_size]
            t = torch.from_numpy(np.ascontiguousarray(batch)).float().unsqueeze(1)
            t = t.to(device)
            recon = model(t)
            # Mean over channel/mel/frame dims -> one error per window.
            err = torch.mean((recon - t) ** 2, dim=(1, 2, 3))
            errors[start : start + batch.shape[0]] = err.detach().cpu().numpy()
    return errors
