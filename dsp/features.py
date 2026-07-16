"""Feature extraction for Mismaa.

Everything here is built transparently on top of NumPy/SciPy so each step can be
reasoned about directly:

* framing / windowing, STFT and its inverse,
* power spectrogram, log-Mel spectrogram (own mel filterbank), MFCC + deltas,
* vibration diagnostics: Hilbert envelope spectrum, per-frame RMS, kurtosis,
  crest factor.

The mel filterbank is implemented from the mel warping formulae rather than
pulled from a library, and the test-suite validates it against ``librosa`` to
prove the implementation is correct.

Spectrogram matrices follow the ``(n_freq, n_frames)`` convention (frequency down
the rows, time across the columns), matching common spectral-analysis tooling.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sps
from scipy.fft import dct
from scipy.stats import kurtosis as _scipy_kurtosis

from dsp.config import DspConfig

# ---------------------------------------------------------------------------
# Windowing and framing
# ---------------------------------------------------------------------------


def get_window(window: str, length: int) -> np.ndarray:
    """Return a length-``length`` analysis window.

    Windows are generated in *periodic* (``fftbins=True``) form, which is the
    correct convention for spectral analysis: it makes the window seamlessly
    tile under the DFT's implicit periodic extension and avoids the 1-sample
    bias of the symmetric (filter-design) form.

    Window choice trades main-lobe width against side-lobe level: a rectangular
    (``boxcar``) window gives the narrowest main lobe (best frequency resolution)
    but −13 dB side lobes (worst leakage); Hann/Hamming widen the main lobe for
    much lower leakage; Blackman lowers it further still. Hann is the balanced
    default.
    """
    return sps.get_window(window, length, fftbins=True).astype(np.float64)


def _pad_center(x: np.ndarray, size: int) -> np.ndarray:
    """Center ``x`` inside a length-``size`` zero array (used to place a short
    window inside a longer FFT)."""
    if x.shape[-1] > size:
        raise ValueError(f"cannot pad_center length {x.shape[-1]} into {size}")
    total = size - x.shape[-1]
    left = total // 2
    return np.pad(x, (left, total - left))


def frame_signal(
    x: np.ndarray,
    frame_length: int,
    hop_length: int,
    *,
    center: bool = True,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """Slice a 1-D signal into overlapping frames.

    Args:
        x: 1-D signal.
        frame_length: Samples per frame. With the analysis window this sets the
            frequency resolution ``df = sample_rate / frame_length``: longer frames
            resolve closely-spaced tones but smear transients in time.
        hop_length: Advance between successive frames. Overlap is
            ``1 - hop_length/frame_length``; 75% overlap (hop = frame/4) keeps the
            short-time analysis smooth and satisfies COLA for Hann.
        center: If true, reflect-pad by ``frame_length // 2`` on each side so frame
            ``t`` is centered on sample ``t * hop_length``.
        pad_mode: NumPy pad mode used when ``center`` is true.

    Returns:
        Array of shape ``(n_frames, frame_length)`` (a view where possible).
    """
    if center:
        pad = frame_length // 2
        x = np.pad(x, pad, mode=pad_mode)

    if x.shape[0] < frame_length:
        raise ValueError(
            f"signal length {x.shape[0]} shorter than frame_length {frame_length}"
        )

    n_frames = 1 + (x.shape[0] - frame_length) // hop_length
    # Zero-copy strided view: each row starts hop_length samples after the last.
    stride = x.strides[0]
    return np.lib.stride_tricks.as_strided(
        x,
        shape=(n_frames, frame_length),
        strides=(hop_length * stride, stride),
        writeable=False,
    )


# ---------------------------------------------------------------------------
# STFT / ISTFT
# ---------------------------------------------------------------------------


def stft(
    x: np.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    win_length: int | None = None,
    window: str = "hann",
    center: bool = True,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """Short-time Fourier transform.

    Args:
        x: 1-D signal.
        n_fft: FFT length. If greater than ``win_length`` the windowed frame is
            zero-padded to ``n_fft``, which interpolates the spectrum onto denser
            bins (smoother plots) without adding true resolution.
        hop_length: Frame advance in samples.
        win_length: Analysis-window length (defaults to ``n_fft``).
        window: Window name (see :func:`get_window`).
        center, pad_mode: Framing padding (see :func:`frame_signal`).

    Returns:
        Complex array ``(1 + n_fft // 2, n_frames)`` (one-sided spectrum per frame).
    """
    win_length = win_length or n_fft
    if win_length > n_fft:
        raise ValueError(f"win_length {win_length} > n_fft {n_fft}")

    fft_window = get_window(window, win_length)
    if win_length < n_fft:
        fft_window = _pad_center(fft_window, n_fft)

    frames = frame_signal(
        x, frame_length=n_fft, hop_length=hop_length, center=center, pad_mode=pad_mode
    )
    windowed = frames * fft_window  # broadcast (n_frames, n_fft) * (n_fft,)
    return np.fft.rfft(windowed, n=n_fft, axis=1).T


def istft(
    spectrum: np.ndarray,
    *,
    hop_length: int,
    window: str = "hann",
    center: bool = True,
    length: int | None = None,
) -> np.ndarray:
    """Inverse STFT via weighted overlap-add.

    Reconstructs the time signal from a one-sided complex spectrogram produced by
    :func:`stft` with ``win_length == n_fft``. Each synthesized frame is multiplied
    by the same window again and the running window-energy is divided out; under
    the COLA condition this exactly inverts the windowed analysis. When the
    spectrogram has been modified (e.g. by spectral subtraction) the reconstruction
    is the least-squares signal consistent with the modified frames.

    Args:
        spectrum: Complex ``(1 + n_fft // 2, n_frames)`` array.
        hop_length: Frame advance used in the forward transform.
        window: Window name used in the forward transform.
        center: Whether the forward transform used center padding (trimmed here).
        length: If given, crop/zero-pad the output to exactly this many samples.

    Returns:
        Real 1-D signal.
    """
    n_freq, n_frames = spectrum.shape
    n_fft = 2 * (n_freq - 1)
    fft_window = get_window(window, n_fft)

    frames = np.fft.irfft(spectrum, n=n_fft, axis=0)  # (n_fft, n_frames), real
    expected_len = n_fft + hop_length * (n_frames - 1)
    y = np.zeros(expected_len, dtype=np.float64)
    win_sum = np.zeros(expected_len, dtype=np.float64)

    for t in range(n_frames):
        start = t * hop_length
        y[start : start + n_fft] += frames[:, t] * fft_window
        win_sum[start : start + n_fft] += fft_window**2

    nonzero = win_sum > 1e-8
    y[nonzero] /= win_sum[nonzero]

    if center:
        pad = n_fft // 2
        y = y[pad : len(y) - pad]

    if length is not None:
        if len(y) < length:
            y = np.pad(y, (0, length - len(y)))
        else:
            y = y[:length]
    return y


def power_spectrogram(spectrum: np.ndarray) -> np.ndarray:
    """Power spectrogram ``|S|^2`` from a complex STFT."""
    return np.abs(spectrum) ** 2


def magnitude_spectrogram(spectrum: np.ndarray) -> np.ndarray:
    """Magnitude spectrogram ``|S|`` from a complex STFT."""
    return np.abs(spectrum)


# ---------------------------------------------------------------------------
# Mel scale and filterbank
# ---------------------------------------------------------------------------


def hz_to_mel(freq: np.ndarray | float, *, scale: str = "slaney") -> np.ndarray | float:
    """Convert Hz to mel.

    ``htk`` uses the single-log O'Shaughnessy formula. ``slaney`` (the Auditory
    Toolbox / librosa default) is linear below 1 kHz and logarithmic above, which
    better matches measured critical-band spacing at low frequencies.
    """
    freq = np.asanyarray(freq, dtype=np.float64)
    if scale == "htk":
        return 2595.0 * np.log10(1.0 + freq / 700.0)
    if scale != "slaney":
        raise ValueError(f"unknown mel scale {scale!r}")

    f_min = 0.0
    f_sp = 200.0 / 3  # 66.67 Hz per mel in the linear region
    mels = (freq - f_min) / f_sp
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = freq >= min_log_hz
    mels = np.where(
        log_region,
        min_log_mel + np.log(np.where(log_region, freq, min_log_hz) / min_log_hz) / logstep,
        mels,
    )
    return mels


def mel_to_hz(mels: np.ndarray | float, *, scale: str = "slaney") -> np.ndarray | float:
    """Inverse of :func:`hz_to_mel`."""
    mels = np.asanyarray(mels, dtype=np.float64)
    if scale == "htk":
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    if scale != "slaney":
        raise ValueError(f"unknown mel scale {scale!r}")

    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - f_min) / f_sp
    logstep = np.log(6.4) / 27.0
    log_region = mels >= min_log_mel
    freqs = np.where(
        log_region,
        min_log_hz * np.exp(logstep * (np.where(log_region, mels, min_log_mel) - min_log_mel)),
        freqs,
    )
    return freqs


def mel_filterbank(
    *,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
    norm: str | None = "slaney",
    scale: str = "slaney",
) -> np.ndarray:
    """Build a triangular mel filterbank matrix.

    Args:
        sample_rate: Signal sample rate.
        n_fft: FFT length that produced the spectrogram (``1 + n_fft//2`` bins).
        n_mels: Number of mel bands. More bands = finer spectral detail but more
            correlated features and fewer FFT bins per band at high frequency.
        fmin, fmax: Frequency range spanned by the bank (``fmax`` defaults to Nyquist).
        norm: ``"slaney"`` scales each triangle to unit area (equal energy per band,
            counteracting the widening of high-frequency triangles); ``None`` leaves
            unit-peak triangles.
        scale: Mel warping formula, ``"slaney"`` or ``"htk"``.

    Returns:
        ``(n_mels, 1 + n_fft // 2)`` matrix ``M`` such that ``M @ power_spec`` gives
        the mel spectrogram.
    """
    if fmax is None:
        fmax = sample_rate / 2.0

    n_freq = 1 + n_fft // 2
    fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)  # (n_freq,)

    # n_mels + 2 equally-mel-spaced band edges, mapped back to Hz.
    mel_min = hz_to_mel(fmin, scale=scale)
    mel_max = hz_to_mel(fmax, scale=scale)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points, scale=scale)  # (n_mels + 2,)

    weights = np.zeros((n_mels, n_freq), dtype=np.float64)
    fdiff = np.diff(hz_points)  # spacing between adjacent edges
    ramps = hz_points[:, np.newaxis] - fft_freqs[np.newaxis, :]  # (n_mels+2, n_freq)

    for i in range(n_mels):
        # Rising edge from band i to i+1, falling edge from i+1 to i+2.
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))

    if norm == "slaney":
        enorm = 2.0 / (hz_points[2 : n_mels + 2] - hz_points[:n_mels])
        weights *= enorm[:, np.newaxis]
    elif norm is not None:
        raise ValueError(f"unknown mel norm {norm!r}")

    return weights


# ---------------------------------------------------------------------------
# Log-Mel and MFCC
# ---------------------------------------------------------------------------


def power_to_db(power: np.ndarray, *, ref: float = 1.0, amin: float = 1e-10) -> np.ndarray:
    """Convert a power spectrogram to decibels: ``10 * log10(max(power, amin)/ref)``.

    ``amin`` floors the input so silent bins do not produce ``-inf``; it sets the
    effective dynamic-range floor.
    """
    return 10.0 * np.log10(np.maximum(amin, power) / ref)


def mel_spectrogram(
    x: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    window: str,
    n_mels: int,
    fmin: float,
    fmax: float | None,
    norm: str | None,
    scale: str,
    center: bool = True,
    pad_mode: str = "reflect",
) -> np.ndarray:
    """Mel power spectrogram: mel filterbank applied to the STFT power spectrum."""
    s = stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=center,
        pad_mode=pad_mode,
    )
    power = power_spectrogram(s)
    fb = mel_filterbank(
        sample_rate=sample_rate,
        n_fft=n_fft,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        norm=norm,
        scale=scale,
    )
    return fb @ power


def mfcc(
    log_mel: np.ndarray,
    *,
    n_mfcc: int,
    lifter: int = 0,
) -> np.ndarray:
    """Mel-frequency cepstral coefficients from a log-Mel spectrogram.

    Applies a type-II orthonormal DCT down the mel axis and keeps the first
    ``n_mfcc`` coefficients. The DCT decorrelates the (highly correlated) mel bands
    and compacts the spectral-envelope energy into the low coefficients; truncating
    keeps the smooth envelope and drops fine ripple (harmonics / pitch).

    Args:
        log_mel: ``(n_mels, n_frames)`` log-Mel spectrogram.
        n_mfcc: Number of cepstral coefficients to keep.
        lifter: Sinusoidal liftering factor; 0 disables. Liftering upweights the
            higher cepstral coefficients, historically improving distance metrics.

    Returns:
        ``(n_mfcc, n_frames)`` MFCC matrix.
    """
    coeffs = dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]
    if lifter > 0:
        n = np.arange(n_mfcc)
        lift = 1.0 + (lifter / 2.0) * np.sin(np.pi * n / lifter)
        coeffs = coeffs * lift[:, np.newaxis]
    return coeffs


def delta(feat: np.ndarray, *, width: int = 9) -> np.ndarray:
    """Temporal delta (regression derivative) of a feature matrix.

    Uses the standard symmetric least-squares regression over a window of
    ``width`` frames (Young et al.). A wider window yields a smoother derivative
    that is more noise-robust but responds with more lag.

    Args:
        feat: ``(n_features, n_frames)`` matrix; the derivative is along time.
        width: Odd regression window length (>= 3).

    Returns:
        ``(n_features, n_frames)`` delta matrix.
    """
    if width < 3 or width % 2 == 0:
        raise ValueError("delta width must be an odd integer >= 3")
    n = (width - 1) // 2
    denom = 2.0 * sum(k * k for k in range(1, n + 1))
    # Edge-pad in time so endpoints use a one-sided-ish (edge-replicated) window.
    padded = np.pad(feat, ((0, 0), (n, n)), mode="edge")
    t = feat.shape[1]
    out = np.zeros_like(feat, dtype=np.float64)
    for k in range(1, n + 1):
        out += k * (padded[:, n + k : n + k + t] - padded[:, n - k : n - k + t])
    return out / denom


# ---------------------------------------------------------------------------
# Vibration diagnostics
# ---------------------------------------------------------------------------


def hilbert_envelope(x: np.ndarray) -> np.ndarray:
    """Amplitude envelope ``|analytic(x)|`` via the Hilbert transform.

    The analytic signal ``x + j*H{x}`` has magnitude equal to the instantaneous
    amplitude, demodulating the carrier. For bearings, high-frequency structural
    resonances are amplitude-modulated by the periodic impacts; the envelope
    recovers that slow modulation.
    """
    return np.abs(sps.hilbert(x))


def envelope_spectrum(
    x: np.ndarray,
    *,
    sample_rate: int,
    band: tuple[float, float] | None = None,
    order: int = 4,
    zero_phase: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Envelope (squared-envelope) spectrum for bearing fault detection.

    Pipeline: optional band-pass around a structural resonance -> Hilbert envelope
    -> remove the envelope's DC (the envelope is non-negative and otherwise
    dominates bin 0) -> one-sided magnitude spectrum. Fault-characteristic
    frequencies (BPFO/BPFI/BSF and harmonics) appear as peaks even when they are
    invisible in the raw spectrum, because band-passing isolates the resonance the
    impacts excite and the Hilbert transform demodulates it.

    Args:
        x: 1-D vibration signal.
        sample_rate: Samples per second.
        band: ``(low_hz, high_hz)`` resonance band to isolate first, or ``None``.
        order: Butterworth order for the pre-filter.
        zero_phase: Use zero-phase (filtfilt) band-pass.

    Returns:
        ``(freqs, magnitude)`` — one-sided frequency axis (Hz) and amplitude
        spectrum of the (mean-removed) envelope.
    """
    sig = x
    if band is not None:
        # Local import avoids an import cycle (preprocess imports this module).
        from dsp.preprocess import butter_bandpass

        sig = butter_bandpass(
            x,
            sample_rate=sample_rate,
            low_hz=band[0],
            high_hz=band[1],
            order=order,
            zero_phase=zero_phase,
        )

    env = hilbert_envelope(sig)
    env = env - env.mean()  # drop the DC bump so real modulation dominates
    spectrum = np.abs(np.fft.rfft(env)) * (2.0 / len(env))  # one-sided amplitude
    freqs = np.fft.rfftfreq(len(env), d=1.0 / sample_rate)
    return freqs, spectrum


def _framewise(x: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    """Non-centered framing for time-domain statistics (each row is real data)."""
    return frame_signal(x, frame_length, hop_length, center=False)


def frame_rms(x: np.ndarray, *, frame_length: int, hop_length: int) -> np.ndarray:
    """Per-frame root-mean-square (energy level over time)."""
    frames = _framewise(x, frame_length, hop_length)
    return np.sqrt(np.mean(frames**2, axis=1))


def frame_kurtosis(x: np.ndarray, *, frame_length: int, hop_length: int) -> np.ndarray:
    """Per-frame excess kurtosis (Fisher: Gaussian -> 0).

    Kurtosis measures the "peakedness"/impulsiveness of the amplitude
    distribution. Incipient bearing faults add sharp impulses that raise kurtosis
    well above 0 before the RMS level changes appreciably, making it an early
    indicator.
    """
    frames = _framewise(x, frame_length, hop_length)
    return _scipy_kurtosis(frames, axis=1, fisher=True, bias=False)


def frame_crest_factor(x: np.ndarray, *, frame_length: int, hop_length: int) -> np.ndarray:
    """Per-frame crest factor = peak amplitude / RMS.

    A pure sine has crest factor sqrt(2) (~1.41); impulsive faults push it higher.
    Like kurtosis it flags impulsiveness, but it saturates once a fault is
    advanced and the signal becomes broadly energetic.
    """
    frames = _framewise(x, frame_length, hop_length)
    peak = np.max(np.abs(frames), axis=1)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    return peak / np.maximum(rms, 1e-12)


# ---------------------------------------------------------------------------
# Config-driven convenience wrappers
# ---------------------------------------------------------------------------


def log_mel_from_config(x: np.ndarray, sample_rate: int, cfg: DspConfig) -> np.ndarray:
    """Log-Mel spectrogram (in dB) using the framing/STFT/mel blocks from config."""
    mel = mel_spectrogram(
        x,
        sample_rate=sample_rate,
        n_fft=cfg.stft.n_fft,
        hop_length=cfg.framing.hop_length,
        win_length=cfg.framing.frame_length,
        window=cfg.framing.window,
        n_mels=cfg.mel.n_mels,
        fmin=cfg.mel.fmin,
        fmax=cfg.mel.fmax,
        norm=cfg.mel.norm,
        scale=cfg.mel.mel_scale,
        center=cfg.framing.center,
        pad_mode=cfg.framing.pad_mode,
    )
    return power_to_db(mel)


def mfcc_from_config(x: np.ndarray, sample_rate: int, cfg: DspConfig) -> np.ndarray:
    """MFCCs (optionally stacked with delta / delta-delta) using config settings."""
    log_mel = log_mel_from_config(x, sample_rate, cfg)
    base = mfcc(log_mel, n_mfcc=cfg.mfcc.n_mfcc, lifter=cfg.mfcc.lifter)
    if not cfg.mfcc.deltas:
        return base
    d1 = delta(base, width=cfg.mfcc.delta_width)
    d2 = delta(d1, width=cfg.mfcc.delta_width)
    return np.vstack([base, d1, d2])
