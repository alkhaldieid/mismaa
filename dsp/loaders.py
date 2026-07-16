"""Dataset loaders for Mismaa.

Two sources are supported:

* MIMII acoustic ``.wav`` files — 10 s, 16 kHz, 8-channel (eight-mic array).
* CWRU bearing vibration ``.mat`` files — 12 kHz drive-end accelerometer, with
  the fault label encoded in the filename.

Both loaders return small frozen dataclasses carrying the signal plus enough
metadata (channel count, sample rate, parsed label) to make downstream code and
tests self-documenting.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import scipy.io as sio
import soundfile as sf

# ---------------------------------------------------------------------------
# MIMII acoustic WAV
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioClip:
    """A loaded acoustic clip.

    Attributes:
        signal: 1-D array (single/mixed channel) or 2-D ``(n_frames, n_channels)``
            when the whole array was requested. Always ``float64`` in [-1, 1].
        sample_rate: Samples per second.
        n_frames: Number of time samples in the file (per channel).
        n_channels: Number of channels present in the source file.
        path: Source path.
    """

    signal: np.ndarray
    sample_rate: int
    n_frames: int
    n_channels: int
    path: str


def load_wav(
    path: str | Path,
    *,
    channel: int | None = 0,
    mono_mixdown: bool = False,
    expected_sample_rate: int | None = None,
) -> AudioClip:
    """Load a (possibly multi-channel) WAV file.

    The eight-mic MIMII array is stored as 8 interleaved channels. Choosing a
    single channel keeps the raw per-mic signal; mixing down to mono averages the
    mics, which raises SNR for a spatially-diffuse source but discards the phase
    differences a beamformer would use.

    Args:
        path: WAV path.
        channel: Zero-based channel index to extract. ``None`` returns the full
            ``(n_frames, n_channels)`` array. Ignored when ``mono_mixdown`` is set.
        mono_mixdown: If true, average across all channels to a single 1-D signal.
        expected_sample_rate: If given, assert the file matches it (guards against
            accidentally feeding, e.g., 48 kHz data into a 16 kHz pipeline).

    Returns:
        An :class:`AudioClip`. ``n_channels``/``n_frames`` always describe the
        source file, regardless of channel selection.
    """
    # always_2d gives a consistent (frames, channels) shape even for mono files.
    data, sr = sf.read(str(path), always_2d=True, dtype="float64")
    n_frames, n_channels = data.shape

    if expected_sample_rate is not None and sr != expected_sample_rate:
        raise ValueError(
            f"{path}: sample rate {sr} != expected {expected_sample_rate}"
        )

    if mono_mixdown:
        signal: np.ndarray = data.mean(axis=1)
    elif channel is None:
        signal = data
    else:
        if not 0 <= channel < n_channels:
            raise IndexError(
                f"{path}: channel {channel} out of range for {n_channels} channels"
            )
        signal = data[:, channel]

    return AudioClip(
        signal=np.ascontiguousarray(signal),
        sample_rate=int(sr),
        n_frames=int(n_frames),
        n_channels=int(n_channels),
        path=str(path),
    )


# ---------------------------------------------------------------------------
# CWRU vibration .mat
# ---------------------------------------------------------------------------

# Filename code -> canonical fault type.
_FAULT_TYPES = {
    "Normal": "normal",
    "IR": "inner_race",
    "B": "ball",
    "OR": "outer_race",
}

# Documented CWRU bench speed (RPM) by motor load (HP). Used only when a record
# omits its measured RPM (some Normal_*.mat files do). Kept here as a module
# constant so the pure filename parser needs no config object; the loader still
# prefers the measured value stored in the file.
CWRU_NOMINAL_RPM = {0: 1797, 1: 1772, 2: 1750, 3: 1730}

# Examples: Normal_0, IR007_1, B014_2, OR021at6_3.
#   code       Normal | IR | B | OR
#   diameter   three digits (mils), absent for Normal
#   position   OR only: clock position of the fault relative to the load zone
#   load       final digit: motor load in HP (0–3)
_CWRU_NAME_RE = re.compile(
    r"^(?P<code>Normal|IR|B|OR)"
    r"(?P<diameter>\d{3})?"
    r"(?:at(?P<position>\d+))?"
    r"_(?P<load>\d)$"
)


@dataclass(frozen=True)
class CwruLabel:
    """Fault label parsed from a CWRU filename (plus measured RPM when known).

    Attributes:
        fault_type: ``normal`` | ``inner_race`` | ``ball`` | ``outer_race``.
        fault_diameter_mils: Fault diameter in thousandths of an inch (0 for normal).
        load_hp: Motor load in horsepower (0–3).
        rpm: Shaft speed. Measured value from the ``.mat`` file when the loader
            fills it; otherwise the nominal bench speed for the load.
        orientation: Outer-race clock position (e.g. ``"6"``) or ``None``.
        raw_name: The filename stem the label was parsed from.
    """

    fault_type: str
    fault_diameter_mils: int
    load_hp: int
    rpm: int | None
    orientation: str | None
    raw_name: str


@dataclass(frozen=True)
class CwruRecord:
    """A loaded CWRU vibration record.

    Attributes:
        signal: 1-D ``float64`` acceleration time series (selected channel).
        sample_rate: Samples per second (12000 for this dataset).
        label: Parsed :class:`CwruLabel` with measured RPM applied when available.
        source_key: The ``.mat`` variable the signal came from (e.g. ``X097_DE_time``).
        path: Source path.
    """

    signal: np.ndarray
    sample_rate: int
    label: CwruLabel
    source_key: str
    path: str


def parse_cwru_filename(name: str) -> CwruLabel:
    """Parse a CWRU filename (or stem) into a :class:`CwruLabel`.

    The RPM is filled from :data:`CWRU_NOMINAL_RPM`; :func:`load_cwru` overwrites
    it with the measured value stored in the file when present.

    Args:
        name: Filename, path, or bare stem (``"OR021at6_3"``, ``"IR007_0.mat"``, ...).

    Raises:
        ValueError: If the name does not match the CWRU naming scheme.
    """
    stem = Path(name).stem
    m = _CWRU_NAME_RE.match(stem)
    if m is None:
        raise ValueError(f"unrecognized CWRU filename: {name!r}")

    code = m.group("code")
    diameter = m.group("diameter")
    load_hp = int(m.group("load"))

    return CwruLabel(
        fault_type=_FAULT_TYPES[code],
        fault_diameter_mils=int(diameter) if diameter is not None else 0,
        load_hp=load_hp,
        rpm=CWRU_NOMINAL_RPM.get(load_hp),
        orientation=m.group("position"),
        raw_name=stem,
    )


def _select_signal_key(keys: list[str], channel: str) -> str:
    """Pick the ``*_<channel>_time`` variable, deterministically.

    A few CWRU baseline files store more than one record (e.g. Normal_2.mat holds
    both X098 and X099). We sort the matching keys and take the first so the
    choice is reproducible; callers can inspect ``CwruRecord.source_key`` to see
    which one was used.
    """
    suffix = f"_{channel}_time"
    matches = sorted(k for k in keys if k.endswith(suffix))
    if not matches:
        raise KeyError(f"no '*{suffix}' variable found among {keys}")
    return matches[0]


def load_cwru(
    path: str | Path,
    *,
    channel: str = "DE",
    sample_rate: int = 12000,
) -> CwruRecord:
    """Load a CWRU ``.mat`` bearing record.

    Args:
        path: ``.mat`` path.
        channel: Accelerometer position — ``DE`` (drive end), ``FE`` (fan end),
            or ``BA`` (base). Selects the ``*_DE_time`` / ``*_FE_time`` / ``*_BA_time``
            variable. Not every record has every channel.
        sample_rate: Sampling rate to attach to the record (12000 for this dataset).

    Returns:
        A :class:`CwruRecord`. The label's RPM is the file's measured value when
        present, else the nominal speed for the load.
    """
    channel = channel.upper()
    if channel not in {"DE", "FE", "BA"}:
        raise ValueError(f"channel must be one of DE/FE/BA, got {channel!r}")

    mat = sio.loadmat(str(path))
    user_keys = [k for k in mat if not k.startswith("__")]

    signal_key = _select_signal_key(user_keys, channel)
    signal = np.asarray(mat[signal_key], dtype="float64").ravel()

    label = parse_cwru_filename(path)

    # Prefer the measured RPM stored alongside the signal (key like "X097RPM").
    rpm_keys = [k for k in user_keys if k.endswith("RPM")]
    if rpm_keys:
        measured_rpm = int(np.asarray(mat[rpm_keys[0]]).ravel()[0])
        label = replace(label, rpm=measured_rpm)

    return CwruRecord(
        signal=signal,
        sample_rate=int(sample_rate),
        label=label,
        source_key=signal_key,
        path=str(path),
    )
