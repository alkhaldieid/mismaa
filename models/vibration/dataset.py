"""CWRU dataset builder for Phase 4: index, segment, split across loads.

The 40 drive-end ``.mat`` records span four fault classes (normal / inner-race /
ball / outer-race) at four motor loads (0-3 HP). We load each record with the
existing :func:`dsp.load_cwru` (which also recovers the measured RPM), cut it
into fixed-length overlapping segments using the existing :func:`dsp.frame_signal`,
and label every segment with its file's fault class.

The train/test split is *by motor load*: loads 0/1/2 train, load 3 tests. This
holds out a whole operating condition rather than random segments (see
``configs/vibration.yaml`` for why that is the honest choice).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dsp import frame_signal, load_cwru

# Fixed label order so class indices are stable across models and confusion
# matrices. Matches dsp.loaders fault_type strings.
CLASS_NAMES = ("normal", "inner_race", "ball", "outer_race")
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}


@dataclass(frozen=True)
class SegmentedCwru:
    """Segmented CWRU dataset, already split by load.

    Segments are raw acceleration windows (for the CNN); ``rpm`` is carried per
    segment so the feature extractor can compute file-specific fault frequencies.

    Attributes (train/ mirrored by test):
        train_segments: ``(n_train, segment_length)`` float32 raw windows.
        train_labels: ``(n_train,)`` class index (see :data:`CLASS_TO_IDX`).
        train_rpm: ``(n_train,)`` measured shaft speed per segment.
        train_load: ``(n_train,)`` motor load (HP) per segment.
        train_file: ``(n_train,)`` source-file stem per segment.
        segment_length: Samples per segment.
        sample_rate: Sampling rate (Hz).
    """

    train_segments: np.ndarray
    train_labels: np.ndarray
    train_rpm: np.ndarray
    train_load: np.ndarray
    train_file: np.ndarray
    test_segments: np.ndarray
    test_labels: np.ndarray
    test_rpm: np.ndarray
    test_load: np.ndarray
    test_file: np.ndarray
    segment_length: int
    sample_rate: int


def segment_signal(signal: np.ndarray, length: int, overlap: float) -> np.ndarray:
    """Cut a 1-D signal into overlapping fixed-length segments.

    Uses the existing :func:`dsp.frame_signal` (non-centered) so no framing logic
    is reimplemented here.

    Args:
        signal: 1-D acceleration series.
        length: Samples per segment.
        overlap: Fractional overlap in ``[0, 1)``; ``hop = round(length*(1-overlap))``.

    Returns:
        ``(n_segments, length)`` contiguous float32 array (empty if the signal is
        shorter than one segment).
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")
    hop = max(1, int(round(length * (1.0 - overlap))))
    if signal.shape[0] < length:
        return np.empty((0, length), dtype=np.float32)
    frames = frame_signal(signal, frame_length=length, hop_length=hop, center=False)
    return np.ascontiguousarray(frames, dtype=np.float32)


def build_segmented_dataset(cfg: dict[str, Any]) -> SegmentedCwru:
    """Load, segment and load-split every CWRU record under ``cfg['data']['root']``.

    Args:
        cfg: Parsed ``configs/vibration.yaml`` dict.

    Returns:
        A :class:`SegmentedCwru`.
    """
    root = Path(cfg["data"]["root"])
    sr = int(cfg["data"]["sample_rate"])
    channel = str(cfg["data"]["channel"])
    length = int(cfg["segment"]["length"])
    overlap = float(cfg["segment"]["overlap"])
    train_loads = set(int(x) for x in cfg["split"]["train_loads"])
    test_load = int(cfg["split"]["test_load"])

    buckets: dict[str, dict[str, list]] = {
        "train": {"seg": [], "lab": [], "rpm": [], "load": [], "file": []},
        "test": {"seg": [], "lab": [], "rpm": [], "load": [], "file": []},
    }

    for mat in sorted(root.glob("*.mat")):
        rec = load_cwru(mat, channel=channel, sample_rate=sr)
        load_hp = rec.label.load_hp
        if load_hp in train_loads:
            split = "train"
        elif load_hp == test_load:
            split = "test"
        else:
            continue  # load not part of this split

        segs = segment_signal(rec.signal, length, overlap)
        if segs.shape[0] == 0:
            continue
        n = segs.shape[0]
        label_idx = CLASS_TO_IDX[rec.label.fault_type]
        rpm = float(rec.label.rpm) if rec.label.rpm is not None else np.nan

        b = buckets[split]
        b["seg"].append(segs)
        b["lab"].append(np.full(n, label_idx, dtype=np.int64))
        b["rpm"].append(np.full(n, rpm, dtype=np.float64))
        b["load"].append(np.full(n, load_hp, dtype=np.int64))
        b["file"].append(np.array([mat.stem] * n, dtype=object))

    def stack(split: str, key: str, empty_shape: tuple) -> np.ndarray:
        parts = buckets[split][key]
        if not parts:
            return np.empty(empty_shape, dtype=np.float32 if key == "seg" else np.int64)
        return np.concatenate(parts, axis=0)

    return SegmentedCwru(
        train_segments=stack("train", "seg", (0, length)),
        train_labels=stack("train", "lab", (0,)),
        train_rpm=stack("train", "rpm", (0,)),
        train_load=stack("train", "load", (0,)),
        train_file=stack("train", "file", (0,)),
        test_segments=stack("test", "seg", (0, length)),
        test_labels=stack("test", "lab", (0,)),
        test_rpm=stack("test", "rpm", (0,)),
        test_load=stack("test", "load", (0,)),
        test_file=stack("test", "file", (0,)),
        segment_length=length,
        sample_rate=sr,
    )
