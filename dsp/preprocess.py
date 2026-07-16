"""Signal preprocessing for Mismaa.

Single-channel time-domain conditioning applied before feature extraction:

* DC / mean removal,
* Butterworth band-pass (configurable order and cutoffs),
* pre-emphasis,
* single-channel spectral-subtraction denoising.

Each function is pure (returns a new array) and takes explicit parameters;
:func:`apply_preprocess` chains the enabled steps from a :class:`DspConfig`.
"""

from __future__ import annotations

import numpy as np
import scipy.signal as sps

from dsp.config import DspConfig, PreprocessConfig
from dsp.features import istft, stft

# ---------------------------------------------------------------------------
# DC removal
# ---------------------------------------------------------------------------


def remove_dc(
    x: np.ndarray,
    *,
    method: str = "mean",
    sample_rate: int | None = None,
    highpass_hz: float = 5.0,
    order: int = 2,
) -> np.ndarray:
    """Remove the DC / low-frequency bias from a signal.

    A residual DC offset biases RMS and kurtosis and dumps energy into the lowest
    STFT bin. ``mean`` subtraction is exact and zero-phase but leaves slow drift;
    a Butterworth ``highpass`` also removes drift/wander at the cost of a
    low-frequency startup transient.

    Args:
        x: 1-D signal.
        method: ``"mean"`` or ``"highpass"``.
        sample_rate: Required for ``highpass``.
        highpass_hz: High-pass corner in Hz (``highpass`` only).
        order: Butterworth order (``highpass`` only).
    """
    if method == "mean":
        return x - np.mean(x)
    if method == "highpass":
        if sample_rate is None:
            raise ValueError("highpass DC removal requires sample_rate")
        sos = sps.butter(order, highpass_hz, btype="highpass", fs=sample_rate, output="sos")
        return sps.sosfiltfilt(sos, x)
    raise ValueError(f"unknown DC-removal method {method!r}")


# ---------------------------------------------------------------------------
# Butterworth band-pass
# ---------------------------------------------------------------------------


def butter_bandpass(
    x: np.ndarray,
    *,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """Butterworth band-pass filter.

    Butterworth gives a maximally-flat passband (no ripple) with a monotonic
    roll-off, trading a gentler transition than Chebyshev/elliptic for the absence
    of passband ripple. Higher ``order`` sharpens the transition but lengthens the
    impulse response (more ringing around transients) and, for causal filtering,
    adds group delay.

    Implemented in second-order-section (SOS) form for numerical stability at high
    order. ``zero_phase`` uses forward-backward filtering (``sosfiltfilt``): it
    cancels phase distortion and squares the magnitude response (so the effective
    order is ``2 * order``), but it is non-causal — offline use only.

    Args:
        x: 1-D signal.
        sample_rate: Samples per second.
        low_hz, high_hz: Passband edges. Must satisfy ``0 < low < high < sample_rate/2``.
        order: Per-direction Butterworth order.
        zero_phase: Forward-backward (``True``) vs causal single-pass (``False``).
    """
    nyquist = sample_rate / 2.0
    if not 0.0 < low_hz < high_hz < nyquist:
        raise ValueError(
            f"require 0 < low ({low_hz}) < high ({high_hz}) < Nyquist ({nyquist})"
        )
    sos = sps.butter(
        order, [low_hz, high_hz], btype="bandpass", fs=sample_rate, output="sos"
    )
    if zero_phase:
        return sps.sosfiltfilt(sos, x)
    return sps.sosfilt(sos, x)


# ---------------------------------------------------------------------------
# Pre-emphasis
# ---------------------------------------------------------------------------


def pre_emphasis(x: np.ndarray, *, coefficient: float = 0.97) -> np.ndarray:
    """First-order pre-emphasis high-pass: ``y[n] = x[n] - a * x[n-1]``.

    This whitens the roughly −6 dB/octave spectral tilt of many mechanical and
    speech signals, boosting weak high-frequency content so it survives the
    dynamic-range compression of the log-Mel step. ``a`` near 1 gives a stronger
    boost; ``a = 0`` is a passthrough. The first sample is kept as-is.

    Args:
        x: 1-D signal.
        coefficient: Emphasis coefficient ``a`` in ``[0, 1)``.
    """
    if not 0.0 <= coefficient < 1.0:
        raise ValueError("pre-emphasis coefficient must be in [0, 1)")
    return np.append(x[0], x[1:] - coefficient * x[:-1])


# ---------------------------------------------------------------------------
# Spectral subtraction
# ---------------------------------------------------------------------------


def spectral_subtraction(
    x: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    window: str = "hann",
    noise_frames: int = 6,
    over_subtraction: float = 2.0,
    spectral_floor: float = 0.02,
    noise_magnitude: np.ndarray | None = None,
) -> np.ndarray:
    """Single-channel spectral-subtraction denoising (Boll / Berouti form).

    Estimates a stationary noise magnitude spectrum (by default from the first
    ``noise_frames`` frames, assumed noise-only), subtracts an over-scaled version
    of it from every frame's magnitude, and floors the result to avoid negative
    magnitudes — then reconstructs using the original noisy phase.

    Parameter trade-offs:

    * ``over_subtraction`` (alpha): removes more noise but, by cutting into the
      speech/signal magnitude, introduces isolated spurious tones ("musical
      noise").
    * ``spectral_floor`` (beta): the residual noise kept as a fraction of the noise
      estimate. Raising it masks musical noise with a low steady floor; lowering it
      removes more noise but exposes the artefacts.

    The stationarity assumption is the key limitation: the noise estimate is fixed,
    so non-stationary noise is only partially removed.

    Args:
        x: 1-D noisy signal.
        sample_rate: Samples per second (used only for consistency/annotation).
        n_fft, hop_length, window: STFT analysis parameters.
        noise_frames: Number of leading frames used to estimate the noise spectrum
            when ``noise_magnitude`` is not supplied.
        over_subtraction: Alpha, ``>= 1``.
        spectral_floor: Beta, in ``(0, 1)``.
        noise_magnitude: Optional precomputed noise magnitude spectrum
            ``(1 + n_fft // 2,)`` (e.g. from a dedicated silence recording).

    Returns:
        The denoised signal, cropped to ``len(x)``.
    """
    spec = stft(x, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window=window)
    magnitude = np.abs(spec)
    phase = np.angle(spec)

    if noise_magnitude is None:
        k = min(noise_frames, magnitude.shape[1])
        noise_magnitude = magnitude[:, :k].mean(axis=1)
    noise_magnitude = noise_magnitude[:, np.newaxis]  # broadcast over frames

    # Berouti: keep max(mag - alpha*noise, beta*noise) so magnitude stays >= a floor.
    subtracted = magnitude - over_subtraction * noise_magnitude
    floored = np.maximum(subtracted, spectral_floor * noise_magnitude)

    denoised_spec = floored * np.exp(1j * phase)
    return istft(denoised_spec, hop_length=hop_length, window=window, length=len(x))


# ---------------------------------------------------------------------------
# Config-driven pipeline
# ---------------------------------------------------------------------------


def apply_preprocess(
    x: np.ndarray,
    *,
    sample_rate: int,
    config: PreprocessConfig | DspConfig,
) -> np.ndarray:
    """Apply the enabled preprocessing steps in a fixed, sensible order.

    Order: DC removal -> band-pass -> pre-emphasis -> spectral subtraction. DC is
    removed first so it does not bias the filters; band-pass restricts the band of
    interest before the (spectral) steps; pre-emphasis whitens the tilt; spectral
    subtraction runs last on the conditioned signal. Each step is applied only if
    enabled in the config.

    Args:
        x: 1-D signal.
        sample_rate: Samples per second.
        config: A :class:`PreprocessConfig`, or a full :class:`DspConfig` (its
            ``.preprocess`` is used).

    Returns:
        The conditioned signal.
    """
    pre = config.preprocess if isinstance(config, DspConfig) else config
    y = np.asarray(x, dtype=np.float64)

    if pre.dc_removal.enabled:
        y = remove_dc(
            y,
            method=pre.dc_removal.method,
            sample_rate=sample_rate,
            highpass_hz=pre.dc_removal.highpass_hz,
        )
    if pre.bandpass.enabled:
        y = butter_bandpass(
            y,
            sample_rate=sample_rate,
            low_hz=pre.bandpass.low_hz,
            high_hz=pre.bandpass.high_hz,
            order=pre.bandpass.order,
            zero_phase=pre.bandpass.zero_phase,
        )
    if pre.pre_emphasis.enabled:
        y = pre_emphasis(y, coefficient=pre.pre_emphasis.coefficient)
    if pre.spectral_subtraction.enabled:
        ss = pre.spectral_subtraction
        y = spectral_subtraction(
            y,
            sample_rate=sample_rate,
            n_fft=ss.n_fft,
            hop_length=ss.hop_length,
            window=ss.window,
            noise_frames=ss.noise_frames,
            over_subtraction=ss.over_subtraction,
            spectral_floor=ss.spectral_floor,
        )
    return y
