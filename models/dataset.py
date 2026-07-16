"""MIMII dataset builder for the Phase 2 acoustic anomaly detector.

Responsibilities:

* index ``data/mimii/{pump,fan}/id_XX/{normal,abnormal}/*.wav`` into typed
  clip records with binary labels;
* turn each clip into a log-Mel spectrogram using the **existing** :mod:`dsp`
  module (no DSP is reimplemented here);
* slice the spectrogram into DCASE-style fixed context windows;
* apply the DCASE 2020 Task 2 train/test split (train on normal only; test on
  held-out normal + all abnormal, per machine id);
* cache the resulting feature tensors to ``data/cache/`` so repeated training
  runs skip the (slow) audio decode + STFT.

Feature parameters that belong to the signal front-end (n_fft, hop, n_mels, mel
warping) come from :func:`dsp.load_config`; only the windowing/split/caching
policy lives in ``configs/train.yaml``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dsp import DspConfig, apply_preprocess, load_config, load_wav, log_mel_from_config

MACHINE_TYPES = ("pump", "fan")
MACHINE_IDS = ("id_00", "id_02", "id_04", "id_06")

# Binary anomaly labels (DCASE convention: 0 = normal/healthy, 1 = anomaly).
LABEL_NORMAL = 0
LABEL_ABNORMAL = 1


@dataclass(frozen=True)
class MimiiClip:
    """One MIMII recording plus its parsed metadata.

    Attributes:
        path: Absolute path to the ``.wav``.
        machine_type: ``"pump"`` or ``"fan"``.
        machine_id: ``"id_00"`` .. ``"id_06"``.
        label: :data:`LABEL_NORMAL` or :data:`LABEL_ABNORMAL`.
        split: ``"train"`` | ``"test"`` | ``""`` (unassigned until split).
    """

    path: str
    machine_type: str
    machine_id: str
    label: int
    split: str = ""


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------


def index_machine(root: str | Path, machine_type: str, machine_id: str) -> list[MimiiClip]:
    """Index every ``.wav`` for one machine id, labelled by its parent folder.

    Args:
        root: MIMII root (``data/mimii``).
        machine_type: ``"pump"`` or ``"fan"``.
        machine_id: e.g. ``"id_00"``.

    Returns:
        Clips sorted by (label, path) for deterministic ordering. Normal first.

    Raises:
        FileNotFoundError: If the machine-id directory does not exist.
    """
    base = Path(root) / machine_type / machine_id
    if not base.is_dir():
        raise FileNotFoundError(f"no such machine directory: {base}")

    clips: list[MimiiClip] = []
    for sub, label in (("normal", LABEL_NORMAL), ("abnormal", LABEL_ABNORMAL)):
        for wav in sorted((base / sub).glob("*.wav")):
            clips.append(
                MimiiClip(
                    path=str(wav.resolve()),
                    machine_type=machine_type,
                    machine_id=machine_id,
                    label=label,
                )
            )
    return sorted(clips, key=lambda c: (c.label, c.path))


# ---------------------------------------------------------------------------
# Train/test split (DCASE 2020 Task 2)
# ---------------------------------------------------------------------------


def assign_split(clips: list[MimiiClip], split_cfg: dict[str, Any]) -> list[MimiiClip]:
    """Assign each clip to ``train`` or ``test`` per the DCASE convention.

    Train receives only normal clips. Test receives a held-out slice of normal
    clips plus every abnormal clip. The number of held-out normals is either
    matched to the abnormal count (``match_abnormal`` -> a balanced test) or a
    fraction of the normals; the choice of which normals is a seeded shuffle so
    the split is reproducible.

    Args:
        clips: All clips for one machine id (from :func:`index_machine`).
        split_cfg: The ``split`` block of ``configs/train.yaml``.

    Returns:
        New clip records with ``split`` populated.
    """
    normals = [c for c in clips if c.label == LABEL_NORMAL]
    abnormals = [c for c in clips if c.label == LABEL_ABNORMAL]

    rng = np.random.default_rng(int(split_cfg["seed"]))
    order = rng.permutation(len(normals))

    mode = split_cfg["test_normal_mode"]
    if mode == "match_abnormal":
        n_test_normal = len(abnormals)
    elif mode == "fraction":
        n_test_normal = int(round(float(split_cfg["test_normal_fraction"]) * len(normals)))
    else:
        raise ValueError(f"unknown test_normal_mode {mode!r}")
    # Always keep at least one normal for training.
    n_test_normal = max(0, min(n_test_normal, len(normals) - 1))

    test_idx = set(order[:n_test_normal].tolist())
    out: list[MimiiClip] = []
    for i, clip in enumerate(normals):
        out.append(MimiiClip(clip.path, clip.machine_type, clip.machine_id, clip.label,
                             "test" if i in test_idx else "train"))
    for clip in abnormals:
        out.append(MimiiClip(clip.path, clip.machine_type, clip.machine_id, clip.label, "test"))
    return out


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def clip_log_mel(
    path: str | Path,
    dsp_cfg: DspConfig,
    *,
    channel: int = 0,
    preprocess: bool = False,
) -> np.ndarray:
    """Log-Mel spectrogram (dB) of a single MIMII channel via the ``dsp`` module.

    Args:
        path: WAV path.
        dsp_cfg: Parsed :class:`dsp.DspConfig` (owns all STFT/mel parameters).
        channel: Mic index to analyse (DCASE baseline uses one mic).
        preprocess: If true, run :func:`dsp.apply_preprocess` first; otherwise the
            log-Mel is taken on the raw mic, matching the DCASE baseline.

    Returns:
        ``(n_mels, n_frames)`` float64 log-Mel spectrogram.
    """
    clip = load_wav(
        path, channel=channel, expected_sample_rate=dsp_cfg.audio.sample_rate
    )
    signal = clip.signal
    if preprocess:
        signal = apply_preprocess(signal, sample_rate=clip.sample_rate, config=dsp_cfg)
    return log_mel_from_config(signal, clip.sample_rate, dsp_cfg)


def context_windows(log_mel: np.ndarray, n_frames: int, hop: int) -> np.ndarray:
    """Slice a log-Mel spectrogram into overlapping fixed-length context windows.

    Each window stacks ``n_frames`` consecutive frames, giving the model local
    temporal context (a single frame is ambiguous; a short run of frames captures
    the machine's periodic signature). We keep the 2-D ``(n_mels, n_frames)`` patch
    rather than flattening it (as the dense DCASE baseline does) so a convolutional
    AE can exploit spectral locality.

    Args:
        log_mel: ``(n_mels, n_time)`` spectrogram.
        n_frames: Frames per window.
        hop: Stride between successive windows, in frames.

    Returns:
        ``(n_windows, n_mels, n_frames)`` contiguous float array. Empty
        ``(0, n_mels, n_frames)`` if the clip is shorter than one window.
    """
    if n_frames < 1 or hop < 1:
        raise ValueError("n_frames and hop must be >= 1")
    n_mels, n_time = log_mel.shape
    if n_time < n_frames:
        return np.empty((0, n_mels, n_frames), dtype=log_mel.dtype)

    n_windows = 1 + (n_time - n_frames) // hop
    src = np.ascontiguousarray(log_mel)
    s_mel, s_time = src.strides
    windows = np.lib.stride_tricks.as_strided(
        src,
        shape=(n_windows, n_mels, n_frames),
        strides=(hop * s_time, s_mel, s_time),
        writeable=False,
    )
    return np.ascontiguousarray(windows)


# ---------------------------------------------------------------------------
# Cached per-machine feature bundle
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MachineFeatures:
    """Feature tensors for one machine id, ready for training and scoring.

    Attributes:
        machine_type, machine_id: Identify the machine.
        train_x: ``(n_train_windows, n_mels, n_frames)`` — normal clips only.
        test_x: ``(n_test_windows, n_mels, n_frames)`` — heldout normal + abnormal.
        test_window_clip: ``(n_test_windows,)`` clip index each test window belongs to.
        test_clip_label: ``(n_test_clips,)`` binary label per test clip.
        test_clip_path: Source path per test clip (parallel to ``test_clip_label``).
    """

    machine_type: str
    machine_id: str
    train_x: np.ndarray
    test_x: np.ndarray
    test_window_clip: np.ndarray
    test_clip_label: np.ndarray
    test_clip_path: list[str]

    @property
    def n_test_clips(self) -> int:
        return len(self.test_clip_label)


def feature_signature(dsp_cfg: DspConfig, cfg: dict[str, Any]) -> str:
    """Short deterministic hash of every parameter that affects the cached tensors.

    Any change to the mic/preprocess choice, the DSP front-end, the context
    windowing, or the split invalidates the cache automatically.
    """
    payload = {
        "channel": cfg["data"]["channel"],
        "preprocess": cfg["data"]["preprocess"],
        "sample_rate": dsp_cfg.audio.sample_rate,
        "n_fft": dsp_cfg.stft.n_fft,
        "hop_length": dsp_cfg.framing.hop_length,
        "win_length": dsp_cfg.framing.frame_length,
        "window": dsp_cfg.framing.window,
        "n_mels": dsp_cfg.mel.n_mels,
        "mel_scale": dsp_cfg.mel.mel_scale,
        "context_frames": cfg["features"]["context_frames"],
        "context_hop": cfg["features"]["context_hop"],
        "max_train_windows": cfg["features"]["max_train_windows"],
        "split": cfg["split"],
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:12]


def _extract_split(
    clips: list[MimiiClip],
    dsp_cfg: DspConfig,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Run feature extraction for both splits of one machine id (no caching)."""
    n_frames = int(cfg["features"]["context_frames"])
    hop = int(cfg["features"]["context_hop"])
    channel = int(cfg["data"]["channel"])
    preprocess = bool(cfg["data"]["preprocess"])

    def windows_for(clip: MimiiClip) -> np.ndarray:
        lm = clip_log_mel(clip.path, dsp_cfg, channel=channel, preprocess=preprocess)
        return context_windows(lm, n_frames, hop).astype(np.float32)

    train_clips = [c for c in clips if c.split == "train"]
    test_clips = [c for c in clips if c.split == "test"]

    train_parts = [windows_for(c) for c in train_clips]
    train_x = (
        np.concatenate(train_parts, axis=0)
        if train_parts
        else np.empty((0, dsp_cfg.mel.n_mels, n_frames), dtype=np.float32)
    )

    # Cap + seeded subsample so RAM/compute stay bounded on large machine ids.
    cap = int(cfg["features"]["max_train_windows"])
    if train_x.shape[0] > cap:
        rng = np.random.default_rng(int(cfg["train"]["seed"]))
        keep = rng.choice(train_x.shape[0], size=cap, replace=False)
        keep.sort()
        train_x = train_x[keep]

    test_parts: list[np.ndarray] = []
    window_clip: list[np.ndarray] = []
    test_labels: list[int] = []
    test_paths: list[str] = []
    for ci, clip in enumerate(test_clips):
        w = windows_for(clip)
        if w.shape[0] == 0:
            continue
        test_parts.append(w)
        window_clip.append(np.full(w.shape[0], ci, dtype=np.int64))
        test_labels.append(clip.label)
        test_paths.append(clip.path)
    test_x = (
        np.concatenate(test_parts, axis=0)
        if test_parts
        else np.empty((0, dsp_cfg.mel.n_mels, n_frames), dtype=np.float32)
    )
    test_window_clip = (
        np.concatenate(window_clip) if window_clip else np.empty((0,), dtype=np.int64)
    )
    return (
        train_x,
        test_x,
        test_window_clip,
        np.asarray(test_labels, dtype=np.int64),
        test_paths,
    )


def prepare_machine(
    machine_type: str,
    machine_id: str,
    dsp_cfg: DspConfig,
    cfg: dict[str, Any],
    *,
    use_cache: bool = True,
    verbose: bool = False,
) -> MachineFeatures:
    """Build (or load from cache) the feature bundle for one machine id.

    Args:
        machine_type, machine_id: Machine to prepare.
        dsp_cfg: Parsed DSP config.
        cfg: Parsed ``configs/train.yaml`` dict.
        use_cache: Read/write ``data/cache/`` when true.
        verbose: Print progress.

    Returns:
        A :class:`MachineFeatures`.
    """
    cache_dir = Path(cfg["data"]["cache_dir"])
    sig = feature_signature(dsp_cfg, cfg)
    cache_path = cache_dir / f"mimii_{machine_type}_{machine_id}_{sig}.npz"

    if use_cache and cache_path.exists():
        if verbose:
            print(f"[cache] {cache_path.name}")
        data = np.load(cache_path, allow_pickle=True)
        return MachineFeatures(
            machine_type=machine_type,
            machine_id=machine_id,
            train_x=data["train_x"],
            test_x=data["test_x"],
            test_window_clip=data["test_window_clip"],
            test_clip_label=data["test_clip_label"],
            test_clip_path=list(data["test_clip_path"]),
        )

    if verbose:
        print(f"[build] {machine_type}/{machine_id} ...")
    clips = assign_split(
        index_machine(cfg["data"]["root"], machine_type, machine_id), cfg["split"]
    )
    train_x, test_x, window_clip, labels, paths = _extract_split(clips, dsp_cfg, cfg)

    if use_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            train_x=train_x,
            test_x=test_x,
            test_window_clip=window_clip,
            test_clip_label=labels,
            test_clip_path=np.asarray(paths, dtype=object),
        )
        if verbose:
            print(f"[cache] wrote {cache_path.name}")

    return MachineFeatures(
        machine_type=machine_type,
        machine_id=machine_id,
        train_x=train_x,
        test_x=test_x,
        test_window_clip=window_clip,
        test_clip_label=labels,
        test_clip_path=paths,
    )


def standardize_fit(train_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit per-mel-band mean/std over the training windows.

    Statistics are computed across windows and frames but kept separate per mel
    band, because log-Mel dB bands live at very different absolute levels. Fitting
    on normal-only training data (and applying the same transform to test) keeps
    the anomaly detector honest: the model never sees test statistics.

    Args:
        train_x: ``(n_windows, n_mels, n_frames)`` training windows.

    Returns:
        ``(mean, std)`` each shaped ``(1, n_mels, 1)`` for broadcasting; ``std``
        is floored to avoid divide-by-zero on a constant band.
    """
    mean = train_x.mean(axis=(0, 2), keepdims=True)
    std = train_x.std(axis=(0, 2), keepdims=True)
    std = np.maximum(std, 1e-6)
    return mean.astype(np.float32), std.astype(np.float32)


def standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Apply a fitted per-mel-band z-score transform."""
    if x.shape[0] == 0:
        return x
    return ((x - mean) / std).astype(np.float32)


def default_dsp_config() -> DspConfig:
    """Convenience: load the repo's DSP config."""
    return load_config()
