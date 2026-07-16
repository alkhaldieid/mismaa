"""Typed access to ``configs/dsp.yaml``.

The YAML file is the single source of truth for every numeric parameter in the
DSP pipeline. This module maps it onto frozen dataclasses so callers get
attribute access, type hints, and a hard failure (``KeyError``) if a required
key is missing — rather than silent, magic in-code defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# configs/dsp.yaml sits next to the repo's dsp/ package: dsp/ -> repo root -> configs/.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "dsp.yaml"


@dataclass(frozen=True)
class AudioConfig:
    """MIMII acoustic acquisition settings."""

    sample_rate: int
    default_channel: int
    mono_mixdown: bool


@dataclass(frozen=True)
class CwruConfig:
    """CWRU vibration acquisition settings.

    ``nominal_rpm`` maps motor load (HP, 0–3) to the documented bench speed and is
    used only when a ``.mat`` record does not store its measured RPM.
    """

    sample_rate: int
    channel: str
    nominal_rpm: dict[int, int]


@dataclass(frozen=True)
class DcRemovalConfig:
    enabled: bool
    method: str
    highpass_hz: float


@dataclass(frozen=True)
class BandpassConfig:
    enabled: bool
    order: int
    low_hz: float
    high_hz: float
    zero_phase: bool


@dataclass(frozen=True)
class PreEmphasisConfig:
    enabled: bool
    coefficient: float


@dataclass(frozen=True)
class SpectralSubtractionConfig:
    enabled: bool
    n_fft: int
    hop_length: int
    window: str
    noise_frames: int
    over_subtraction: float
    spectral_floor: float


@dataclass(frozen=True)
class PreprocessConfig:
    dc_removal: DcRemovalConfig
    bandpass: BandpassConfig
    pre_emphasis: PreEmphasisConfig
    spectral_subtraction: SpectralSubtractionConfig


@dataclass(frozen=True)
class FramingConfig:
    frame_length: int
    hop_length: int
    window: str
    center: bool
    pad_mode: str


@dataclass(frozen=True)
class StftConfig:
    n_fft: int


@dataclass(frozen=True)
class MelConfig:
    n_mels: int
    fmin: float
    fmax: float | None
    norm: str | None
    mel_scale: str


@dataclass(frozen=True)
class MfccConfig:
    n_mfcc: int
    lifter: int
    deltas: bool
    delta_width: int


@dataclass(frozen=True)
class EnvelopeConfig:
    enabled: bool
    bandpass: BandpassConfig


@dataclass(frozen=True)
class StatsConfig:
    frame_length: int
    hop_length: int


@dataclass(frozen=True)
class VibrationConfig:
    envelope: EnvelopeConfig
    stats: StatsConfig


@dataclass(frozen=True)
class DspConfig:
    """Root of the parsed ``dsp.yaml`` tree."""

    audio: AudioConfig
    cwru: CwruConfig
    preprocess: PreprocessConfig
    framing: FramingConfig
    stft: StftConfig
    mel: MelConfig
    mfcc: MfccConfig
    vibration: VibrationConfig


def _bandpass(raw: dict) -> BandpassConfig:
    return BandpassConfig(
        enabled=bool(raw["enabled"]),
        order=int(raw["order"]),
        low_hz=float(raw["low_hz"]),
        high_hz=float(raw["high_hz"]),
        zero_phase=bool(raw["zero_phase"]),
    )


def _build(raw: dict) -> DspConfig:
    """Construct a :class:`DspConfig` from the parsed YAML mapping."""
    audio = raw["audio"]
    cwru = raw["cwru"]
    pre = raw["preprocess"]
    framing = raw["framing"]
    stft = raw["stft"]
    mel = raw["mel"]
    mfcc = raw["mfcc"]
    vib = raw["vibration"]

    return DspConfig(
        audio=AudioConfig(
            sample_rate=int(audio["sample_rate"]),
            default_channel=int(audio["default_channel"]),
            mono_mixdown=bool(audio["mono_mixdown"]),
        ),
        cwru=CwruConfig(
            sample_rate=int(cwru["sample_rate"]),
            channel=str(cwru["channel"]),
            nominal_rpm={int(k): int(v) for k, v in cwru["nominal_rpm"].items()},
        ),
        preprocess=PreprocessConfig(
            dc_removal=DcRemovalConfig(
                enabled=bool(pre["dc_removal"]["enabled"]),
                method=str(pre["dc_removal"]["method"]),
                highpass_hz=float(pre["dc_removal"]["highpass_hz"]),
            ),
            bandpass=_bandpass(pre["bandpass"]),
            pre_emphasis=PreEmphasisConfig(
                enabled=bool(pre["pre_emphasis"]["enabled"]),
                coefficient=float(pre["pre_emphasis"]["coefficient"]),
            ),
            spectral_subtraction=SpectralSubtractionConfig(
                enabled=bool(pre["spectral_subtraction"]["enabled"]),
                n_fft=int(pre["spectral_subtraction"]["n_fft"]),
                hop_length=int(pre["spectral_subtraction"]["hop_length"]),
                window=str(pre["spectral_subtraction"]["window"]),
                noise_frames=int(pre["spectral_subtraction"]["noise_frames"]),
                over_subtraction=float(pre["spectral_subtraction"]["over_subtraction"]),
                spectral_floor=float(pre["spectral_subtraction"]["spectral_floor"]),
            ),
        ),
        framing=FramingConfig(
            frame_length=int(framing["frame_length"]),
            hop_length=int(framing["hop_length"]),
            window=str(framing["window"]),
            center=bool(framing["center"]),
            pad_mode=str(framing["pad_mode"]),
        ),
        stft=StftConfig(n_fft=int(stft["n_fft"])),
        mel=MelConfig(
            n_mels=int(mel["n_mels"]),
            fmin=float(mel["fmin"]),
            fmax=None if mel["fmax"] is None else float(mel["fmax"]),
            norm=None if mel["norm"] is None else str(mel["norm"]),
            mel_scale=str(mel["mel_scale"]),
        ),
        mfcc=MfccConfig(
            n_mfcc=int(mfcc["n_mfcc"]),
            lifter=int(mfcc["lifter"]),
            deltas=bool(mfcc["deltas"]),
            delta_width=int(mfcc["delta_width"]),
        ),
        vibration=VibrationConfig(
            envelope=EnvelopeConfig(
                enabled=bool(vib["envelope"]["enabled"]),
                bandpass=_bandpass(vib["envelope"]["bandpass"]),
            ),
            stats=StatsConfig(
                frame_length=int(vib["stats"]["frame_length"]),
                hop_length=int(vib["stats"]["hop_length"]),
            ),
        ),
    )


def load_config(path: str | Path | None = None) -> DspConfig:
    """Load and parse the DSP config.

    Args:
        path: Path to a YAML file. Defaults to :data:`DEFAULT_CONFIG_PATH`.

    Returns:
        A fully-populated :class:`DspConfig`.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    with open(cfg_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return _build(raw)
