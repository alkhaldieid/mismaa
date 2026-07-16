"""Mismaa DSP package: loaders, preprocessing, and feature extraction.

Phase 1 preprocessing/feature module for MIMII acoustic and CWRU vibration data.
All numeric parameters are config-driven via :mod:`dsp.config` (``configs/dsp.yaml``).
"""

from __future__ import annotations

from dsp.config import DspConfig, load_config
from dsp.features import (
    delta,
    envelope_spectrum,
    frame_crest_factor,
    frame_kurtosis,
    frame_rms,
    frame_signal,
    hilbert_envelope,
    hz_to_mel,
    log_mel_from_config,
    mel_filterbank,
    mel_spectrogram,
    mel_to_hz,
    mfcc,
    mfcc_from_config,
    power_spectrogram,
    power_to_db,
    stft,
)
from dsp.loaders import (
    AudioClip,
    CwruLabel,
    CwruRecord,
    load_cwru,
    load_wav,
    parse_cwru_filename,
)
from dsp.preprocess import (
    apply_preprocess,
    butter_bandpass,
    pre_emphasis,
    remove_dc,
    spectral_subtraction,
)

__all__ = [
    "AudioClip",
    "CwruLabel",
    "CwruRecord",
    "DspConfig",
    "apply_preprocess",
    "butter_bandpass",
    "delta",
    "envelope_spectrum",
    "frame_crest_factor",
    "frame_kurtosis",
    "frame_rms",
    "frame_signal",
    "hilbert_envelope",
    "hz_to_mel",
    "load_config",
    "load_cwru",
    "load_wav",
    "log_mel_from_config",
    "mel_filterbank",
    "mel_spectrogram",
    "mel_to_hz",
    "mfcc",
    "mfcc_from_config",
    "parse_cwru_filename",
    "power_spectrogram",
    "power_to_db",
    "pre_emphasis",
    "remove_dc",
    "spectral_subtraction",
    "stft",
]
