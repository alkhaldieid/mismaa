"""Rolling-element bearing fault frequencies.

The CWRU drive-end bearing is an **SKF 6205-2RS JEM** deep-groove ball bearing.
A localised spall on a raceway or ball produces a periodic impact each time a
ball rolls over (or the defect passes through) the load zone. The repetition
rates are fixed by the bearing geometry and the shaft speed, and each fault type
has its own characteristic frequency:

* **BPFO** — Ball Pass Frequency, Outer race
* **BPFI** — Ball Pass Frequency, Inner race
* **BSF**  — Ball Spin Frequency (a ball defect strikes a race twice per spin,
  so ball faults often show energy at ``2 * BSF``)
* **FTF**  — Fundamental Train Frequency (cage rotation)

With shaft rotation frequency ``fr = rpm / 60`` (Hz), number of balls ``n``,
ball diameter ``d``, pitch diameter ``D`` and contact angle ``theta``:

    BPFO = (n / 2) * fr * (1 - (d / D) * cos(theta))
    BPFI = (n / 2) * fr * (1 + (d / D) * cos(theta))
    BSF  = (D / (2 * d)) * fr * (1 - ((d / D) * cos(theta)) ** 2)
    FTF  = (1 / 2)     * fr * (1 - (d / D) * cos(theta))

For the SKF 6205 (n=9, d=0.3126 in, D=1.537 in, theta=0) these reduce to the
well-documented per-revolution multipliers BPFO=3.5848, BPFI=5.4152, BSF=2.3568,
FTF=0.3983 times ``fr`` — the values used to sanity-check this module in the
tests. The geometry itself lives in ``configs/vibration.yaml`` (no bearing
constants are hard-coded here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BearingGeometry:
    """Physical geometry of a rolling-element bearing.

    Attributes:
        n_balls: Number of rolling elements.
        ball_diameter: Ball diameter ``d`` (any consistent length unit).
        pitch_diameter: Pitch (cage) diameter ``D`` (same unit as ``ball_diameter``).
        contact_angle_deg: Contact angle ``theta`` in degrees (0 for deep-groove
            radial bearings such as the 6205).
    """

    n_balls: int
    ball_diameter: float
    pitch_diameter: float
    contact_angle_deg: float = 0.0

    @property
    def _ratio_cos(self) -> float:
        """The recurring term ``(d / D) * cos(theta)``."""
        return (self.ball_diameter / self.pitch_diameter) * math.cos(
            math.radians(self.contact_angle_deg)
        )


def shaft_frequency(rpm: float) -> float:
    """Shaft rotation frequency in Hz from RPM (``fr = rpm / 60``)."""
    return rpm / 60.0


def bpfo(geometry: BearingGeometry, rpm: float) -> float:
    """Ball Pass Frequency, Outer race (Hz)."""
    return (geometry.n_balls / 2.0) * shaft_frequency(rpm) * (1.0 - geometry._ratio_cos)


def bpfi(geometry: BearingGeometry, rpm: float) -> float:
    """Ball Pass Frequency, Inner race (Hz)."""
    return (geometry.n_balls / 2.0) * shaft_frequency(rpm) * (1.0 + geometry._ratio_cos)


def bsf(geometry: BearingGeometry, rpm: float) -> float:
    """Ball Spin Frequency (Hz). Ball defects also excite ``2 * BSF``."""
    ratio = geometry._ratio_cos
    return (
        (geometry.pitch_diameter / (2.0 * geometry.ball_diameter))
        * shaft_frequency(rpm)
        * (1.0 - ratio**2)
    )


def ftf(geometry: BearingGeometry, rpm: float) -> float:
    """Fundamental Train Frequency / cage frequency (Hz)."""
    return 0.5 * shaft_frequency(rpm) * (1.0 - geometry._ratio_cos)


def fault_frequencies(geometry: BearingGeometry, rpm: float) -> dict[str, float]:
    """All characteristic fault frequencies (Hz) at the given shaft speed."""
    return {
        "BPFO": bpfo(geometry, rpm),
        "BPFI": bpfi(geometry, rpm),
        "BSF": bsf(geometry, rpm),
        "FTF": ftf(geometry, rpm),
    }


def geometry_from_config(cfg: dict[str, Any]) -> BearingGeometry:
    """Build a :class:`BearingGeometry` from the ``bearing`` block of a config."""
    b = cfg["bearing"]
    return BearingGeometry(
        n_balls=int(b["n_balls"]),
        ball_diameter=float(b["ball_diameter"]),
        pitch_diameter=float(b["pitch_diameter"]),
        contact_angle_deg=float(b["contact_angle_deg"]),
    )
