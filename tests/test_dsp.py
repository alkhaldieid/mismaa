"""Tests for the Mismaa DSP module.

Strategy:

* Synthetic signals with *known* spectral content — a sine on an exact FFT bin, a
  linear chirp, an amplitude-modulated impact train at a known repetition rate —
  so every transform can be checked against an analytic expectation.
* The custom mel filterbank is validated against ``librosa`` (reference impl).
* Two integration tests load a real MIMII WAV and a real CWRU ``.mat`` and assert
  the documented shape / label. They ``skip`` (not fail) when the data is absent,
  so the suite still runs in a data-less CI.
"""

from __future__ import annotations

import dataclasses as dc
import glob
import math

import numpy as np
import pytest

from dsp import features, loaders, preprocess
from dsp.config import DspConfig, load_config

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cfg() -> DspConfig:
    return load_config()


def _sine(freq: float, sample_rate: int, n: int, amp: float = 1.0) -> np.ndarray:
    t = np.arange(n) / sample_rate
    return amp * np.sin(2.0 * np.pi * freq * t)


def _first_existing(pattern: str) -> str | None:
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def test_config_loads_and_is_typed(cfg: DspConfig) -> None:
    assert cfg.audio.sample_rate == 16000
    assert cfg.cwru.sample_rate == 12000
    assert cfg.cwru.nominal_rpm[0] == 1797
    assert cfg.mel.n_mels in (64, 128)
    # fmax null in YAML -> None in the dataclass.
    assert cfg.mel.fmax is None
    assert cfg.framing.frame_length > cfg.framing.hop_length  # real overlap


# ---------------------------------------------------------------------------
# CWRU filename parser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,ftype,dia,load,orient",
    [
        ("Normal_0", "normal", 0, 0, None),
        ("IR007_1", "inner_race", 7, 1, None),
        ("B014_2", "ball", 14, 2, None),
        ("OR021at6_3", "outer_race", 21, 3, "6"),
        ("data/cwru/IR021_0.mat", "inner_race", 21, 0, None),
    ],
)
def test_parse_cwru_filename(name, ftype, dia, load, orient) -> None:
    label = loaders.parse_cwru_filename(name)
    assert label.fault_type == ftype
    assert label.fault_diameter_mils == dia
    assert label.load_hp == load
    assert label.orientation == orient
    # Nominal RPM is filled from the documented bench-speed table.
    assert label.rpm == loaders.CWRU_NOMINAL_RPM[load]


def test_parse_cwru_filename_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        loaders.parse_cwru_filename("totally_not_cwru.mat")


# ---------------------------------------------------------------------------
# Framing / windowing
# ---------------------------------------------------------------------------


def test_frame_signal_shapes_and_hop() -> None:
    x = np.arange(1000, dtype=np.float64)
    frames = features.frame_signal(x, frame_length=100, hop_length=25, center=False)
    assert frames.shape == (1 + (1000 - 100) // 25, 100)
    # Row t begins hop*t samples into the signal.
    assert frames[0, 0] == 0.0
    assert frames[1, 0] == 25.0
    assert frames[2, 0] == 50.0


def test_window_length_and_periodic() -> None:
    w = features.get_window("hann", 512)
    assert w.shape == (512,)
    # Periodic Hann starts at 0 and does NOT return to 0 at the end (that is the
    # symmetric form); this is the fftbins=True convention.
    assert w[0] == pytest.approx(0.0, abs=1e-12)
    assert w[-1] > 0.0


# ---------------------------------------------------------------------------
# STFT — sine on a known bin, chirp with rising instantaneous frequency
# ---------------------------------------------------------------------------


def test_stft_sine_peaks_on_expected_bin() -> None:
    sr, n_fft = 16000, 1024
    k = 100  # target bin
    f0 = k * sr / n_fft  # exactly on bin k -> no leakage across bins
    x = _sine(f0, sr, sr)  # 1 s
    spec = features.stft(x, n_fft=n_fft, hop_length=256, window="hann")
    power = features.power_spectrogram(spec).mean(axis=1)  # average over time
    assert int(np.argmax(power)) == k
    # Peak should dominate its neighbours by orders of magnitude.
    assert power[k] > 100 * power[k + 3]


def test_stft_chirp_instantaneous_frequency_rises() -> None:
    import scipy.signal as sps

    sr, n_fft, hop = 16000, 1024, 256
    n = 2 * sr
    t = np.arange(n) / sr
    x = sps.chirp(t, f0=500, t1=t[-1], f1=6000, method="linear")
    spec = features.stft(x, n_fft=n_fft, hop_length=hop, window="hann")
    peak_bins = np.argmax(np.abs(spec), axis=0)  # per-frame dominant bin
    n_frames = peak_bins.shape[0]
    first_quarter = peak_bins[: n_frames // 4].mean()
    last_quarter = peak_bins[-n_frames // 4 :].mean()
    assert last_quarter > first_quarter  # frequency sweeps upward
    # Monotone-ish: strong positive rank correlation with frame index.
    corr = np.corrcoef(np.arange(n_frames), peak_bins)[0, 1]
    assert corr > 0.95


def test_istft_reconstructs_signal() -> None:
    sr, n_fft, hop = 16000, 1024, 256
    x = _sine(1000.0, sr, sr) + 0.3 * _sine(2500.0, sr, sr)
    spec = features.stft(x, n_fft=n_fft, hop_length=hop, window="hann")
    y = features.istft(spec, hop_length=hop, window="hann", length=len(x))
    # Ignore a few edge frames where COLA is incomplete.
    edge = n_fft
    err = np.abs(x[edge:-edge] - y[edge:-edge]).max()
    assert err < 1e-6


# ---------------------------------------------------------------------------
# Mel filterbank — validated against librosa
# ---------------------------------------------------------------------------


def test_mel_filterbank_matches_librosa_slaney() -> None:
    librosa = pytest.importorskip("librosa")
    kw = dict(sr=16000, n_fft=1024, n_mels=64, fmin=0.0, fmax=8000.0)
    ref = librosa.filters.mel(**kw, htk=False, norm="slaney")
    mine = features.mel_filterbank(
        sample_rate=16000, n_fft=1024, n_mels=64, fmin=0.0, fmax=8000.0,
        norm="slaney", scale="slaney",
    )
    assert mine.shape == ref.shape
    assert np.allclose(mine, ref, atol=1e-6)


def test_mel_filterbank_matches_librosa_htk() -> None:
    librosa = pytest.importorskip("librosa")
    ref = librosa.filters.mel(
        sr=16000, n_fft=1024, n_mels=40, fmin=0.0, fmax=8000.0, htk=True, norm=None
    )
    mine = features.mel_filterbank(
        sample_rate=16000, n_fft=1024, n_mels=40, fmin=0.0, fmax=8000.0,
        norm=None, scale="htk",
    )
    assert np.allclose(mine, ref, atol=1e-6)


def test_hz_mel_roundtrip() -> None:
    freqs = np.array([0.0, 100.0, 1000.0, 4000.0, 8000.0])
    for scale in ("slaney", "htk"):
        back = features.mel_to_hz(features.hz_to_mel(freqs, scale=scale), scale=scale)
        assert np.allclose(back, freqs, atol=1e-6)


# ---------------------------------------------------------------------------
# Log-Mel / MFCC / deltas
# ---------------------------------------------------------------------------


def test_log_mel_and_mfcc_shapes(cfg: DspConfig) -> None:
    sr = cfg.audio.sample_rate
    x = _sine(1000.0, sr, sr)
    log_mel = features.log_mel_from_config(x, sr, cfg)
    assert log_mel.shape[0] == cfg.mel.n_mels

    mfcc_stack = features.mfcc_from_config(x, sr, cfg)
    # base + delta + delta-delta when deltas enabled.
    expected_rows = cfg.mfcc.n_mfcc * (3 if cfg.mfcc.deltas else 1)
    assert mfcc_stack.shape[0] == expected_rows
    assert mfcc_stack.shape[1] == log_mel.shape[1]
    assert np.all(np.isfinite(mfcc_stack))


def test_delta_of_linear_ramp_is_constant_slope() -> None:
    # feat[f, t] = t  ->  time-derivative is 1 everywhere in the interior.
    t = np.arange(50, dtype=np.float64)
    feat = np.vstack([t, 2 * t + 5])  # two features, both linear in t
    d = features.delta(feat, width=9)
    interior = d[:, 5:-5]
    # Feature 0 has slope 1, feature 1 has slope 2.
    assert np.allclose(interior[0], 1.0, atol=1e-9)
    assert np.allclose(interior[1], 2.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def test_remove_dc_mean() -> None:
    x = _sine(50.0, 12000, 12000) + 3.7  # 3.7 DC offset
    y = preprocess.remove_dc(x, method="mean")
    assert abs(np.mean(y)) < 1e-9


def test_butter_bandpass_passes_in_band_rejects_out_of_band() -> None:
    sr = 12000
    n = 12000
    in_band = _sine(1000.0, sr, n)   # inside 500–5000
    out_band = _sine(100.0, sr, n)   # below the passband
    kw = dict(sample_rate=sr, low_hz=500.0, high_hz=5000.0, order=4)

    passed = preprocess.butter_bandpass(in_band, **kw)
    rejected = preprocess.butter_bandpass(out_band, **kw)

    rms = lambda s: np.sqrt(np.mean(s**2))  # noqa: E731
    # In-band tone survives; out-of-band tone is strongly attenuated.
    assert rms(passed) > 0.9 * rms(in_band)
    assert rms(rejected) < 0.05 * rms(out_band)


def test_butter_bandpass_validates_cutoffs() -> None:
    with pytest.raises(ValueError):
        preprocess.butter_bandpass(
            _sine(100.0, 12000, 1200), sample_rate=12000, low_hz=5000.0, high_hz=7000.0
        )  # high > Nyquist (6000)


def test_pre_emphasis_boosts_high_attenuates_low() -> None:
    sr, n = 16000, 16000
    low = _sine(20.0, sr, n)
    high = _sine(7500.0, sr, n)  # near Nyquist
    rms = lambda s: np.sqrt(np.mean(s**2))  # noqa: E731

    low_out = preprocess.pre_emphasis(low, coefficient=0.97)
    high_out = preprocess.pre_emphasis(high, coefficient=0.97)
    assert rms(low_out) < 0.1 * rms(low)   # |1 - a| ≈ 0.03 at DC
    assert rms(high_out) > 1.5 * rms(high)  # |1 + a| ≈ 1.97 at Nyquist


def test_spectral_subtraction_lowers_noise_floor() -> None:
    rng = np.random.default_rng(0)
    sr, n = 16000, 16000
    n_fft, hop = 1024, 256
    tone = _sine(1000.0, sr, n, amp=1.0)
    noise = rng.normal(0.0, 0.5, n)
    noisy = tone + noise

    # Independent noise realisation -> honest noise-magnitude estimate.
    noise_ref = rng.normal(0.0, 0.5, n)
    noise_mag = np.abs(
        features.stft(noise_ref, n_fft=n_fft, hop_length=hop, window="hann")
    ).mean(axis=1)

    denoised = preprocess.spectral_subtraction(
        noisy, sample_rate=sr, n_fft=n_fft, hop_length=hop,
        noise_magnitude=noise_mag, over_subtraction=2.0, spectral_floor=0.02,
    )

    # Compare the noise floor in bins away from the 1 kHz tone (3–5 kHz).
    def floor(sig: np.ndarray) -> float:
        mag = np.abs(features.stft(sig, n_fft=n_fft, hop_length=hop, window="hann"))
        lo = int(3000 * n_fft / sr)
        hi = int(5000 * n_fft / sr)
        return float(np.median(mag[lo:hi]))

    assert denoised.shape == noisy.shape
    assert floor(denoised) < 0.7 * floor(noisy)


def test_apply_preprocess_full_chain_runs(cfg: DspConfig) -> None:
    # Enable every stage and confirm the chain runs and preserves length.
    pre = cfg.preprocess
    all_on = dc.replace(
        pre,
        dc_removal=dc.replace(pre.dc_removal, enabled=True),
        bandpass=dc.replace(pre.bandpass, enabled=True, low_hz=20.0, high_hz=6000.0),
        pre_emphasis=dc.replace(pre.pre_emphasis, enabled=True),
        spectral_subtraction=dc.replace(pre.spectral_subtraction, enabled=True),
    )
    full = dc.replace(cfg, preprocess=all_on)
    sr = 16000
    x = _sine(1000.0, sr, sr) + 0.4
    y = preprocess.apply_preprocess(x, sample_rate=sr, config=full)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


# ---------------------------------------------------------------------------
# Vibration diagnostics
# ---------------------------------------------------------------------------


def test_envelope_spectrum_recovers_impact_repetition_rate() -> None:
    # Amplitude-modulated resonance: impacts at fr excite a 3 kHz decaying
    # ringdown. The raw spectrum shows the 3 kHz carrier; the ENVELOPE spectrum
    # should reveal the fr repetition rate (classic bearing-fault signature).
    sr = 12000
    fr = 100.0                 # impact repetition rate (Hz)
    period = int(sr / fr)      # 120 samples
    fc = 3000.0                # excited resonance
    n = sr                     # 1 s -> 1 Hz frequency resolution

    k = np.arange(400)
    ringdown = np.exp(-k / 40.0) * np.sin(2.0 * np.pi * fc * k / sr)
    x = np.zeros(n)
    for start in range(0, n - len(ringdown), period):
        x[start : start + len(ringdown)] += ringdown
    x += np.random.default_rng(1).normal(0.0, 1e-3, n)

    freqs, mag = features.envelope_spectrum(
        x, sample_rate=sr, band=(2000.0, 4000.0), order=4
    )
    # Ignore the near-DC region; the dominant envelope peak is the repetition rate.
    mask = freqs > 10.0
    peak_freq = freqs[mask][np.argmax(mag[mask])]
    assert abs(peak_freq - fr) < 3.0


def test_frame_rms_of_sine() -> None:
    sr, n = 16000, 16384
    amp = 2.0
    x = _sine(1000.0, sr, n, amp=amp)
    rms = features.frame_rms(x, frame_length=4096, hop_length=4096)
    assert np.allclose(rms, amp / math.sqrt(2), rtol=0.02)


def test_frame_kurtosis_gaussian_near_zero() -> None:
    x = np.random.default_rng(2).normal(0.0, 1.0, 16384)
    kurt = features.frame_kurtosis(x, frame_length=4096, hop_length=4096)
    assert abs(float(np.mean(kurt))) < 0.3  # Fisher: Gaussian -> 0


def test_frame_crest_factor_of_sine() -> None:
    sr, n = 16000, 16384
    x = _sine(1000.0, sr, n, amp=1.0)
    crest = features.frame_crest_factor(x, frame_length=4096, hop_length=4096)
    assert np.allclose(crest, math.sqrt(2), rtol=0.03)  # sine crest factor = √2


# ---------------------------------------------------------------------------
# Integration — real data (skip if not present)
# ---------------------------------------------------------------------------


def test_integration_real_mimii_wav() -> None:
    path = _first_existing("data/mimii/*/*/*/*.wav")
    if path is None:
        pytest.skip("no MIMII WAV data on disk")
    # channel=None returns the full multichannel array to check the raw shape.
    clip = loaders.load_wav(path, channel=None, expected_sample_rate=16000)
    assert clip.sample_rate == 16000
    assert clip.n_frames == 160000  # 10 s @ 16 kHz
    assert clip.n_channels == 8     # eight-mic array
    assert clip.signal.shape == (160000, 8)

    # Default single-channel selection collapses to 1-D of the same length.
    mono = loaders.load_wav(path, channel=0)
    assert mono.signal.shape == (160000,)
    assert mono.n_channels == 8


def test_integration_real_cwru_mat() -> None:
    path = _first_existing("data/cwru/IR007_0.mat")
    if path is None:
        path = _first_existing("data/cwru/*.mat")
    if path is None:
        pytest.skip("no CWRU .mat data on disk")

    rec = loaders.load_cwru(path, channel="DE")
    assert rec.source_key.endswith("_DE_time")  # DE_time channel resolved
    assert rec.sample_rate == 12000
    assert rec.signal.ndim == 1 and rec.signal.size > 1000
    assert np.issubdtype(rec.signal.dtype, np.floating)

    if path.endswith("IR007_0.mat"):
        assert rec.label.fault_type == "inner_race"
        assert rec.label.fault_diameter_mils == 7
        assert rec.label.load_hp == 0
        # Measured RPM read from the file overrides the nominal table.
        assert rec.label.rpm == 1797
