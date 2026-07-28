"""Twenty-second per-user calibration.

Why this exists: hand-span normalization removes camera distance, but not
anatomy. Finger-length-to-palm-length ratio varies enough between people that a
pinch aperture of 0.45 hand-spans is "wide open" for one user and "half closed"
for another. Hardcoded ranges are the reason most gesture demos feel great for
their author and bad for everyone else.

The flow: prompt the user through a small number of extreme poses, collect
samples, and take **robust percentiles** rather than min/max. Min/max is
brittle -- a single frame of landmark garbage sets your range for the whole
session. The 5th/95th percentiles over a few hundred frames are stable.

Calibration profiles are JSON and versioned, so a profile recorded against an
older signal definition is rejected rather than silently misinterpreted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mapper import SIGNAL_NAMES, SignalRange

#: Bump when the meaning of any raw signal changes.
CALIBRATION_VERSION = 1

#: Fallback ranges measured on the author's hand. Usable, but calibrate.
_DEFAULT_RANGES: dict[str, tuple[float, float]] = {
    "pinch": (0.06, 0.95),
    "openness": (0.55, 1.85),
    "curl": (0.10, 2.60),
    "spread": (0.02, 0.40),
    "pitch": (-1.40, 1.40),
    "roll": (-1.60, 1.60),
    "yaw": (-1.60, 1.60),
}


@dataclass(frozen=True)
class CalibrationStep:
    """One prompted pose in the calibration sequence."""

    key: str
    prompt: str
    seconds: float = 3.0


CALIBRATION_SEQUENCE: tuple[CalibrationStep, ...] = (
    CalibrationStep("open", "Open your hand wide, fingers splayed. Hold.", 3.0),
    CalibrationStep("fist", "Close into a loose fist. Hold.", 3.0),
    CalibrationStep("pinch_closed", "Touch thumb and index tips together. Hold.", 3.0),
    CalibrationStep("pinch_open", "Spread thumb and index as far apart as you can. Hold.", 3.0),
    CalibrationStep(
        "rotate", "Slowly rotate your wrist through its full comfortable range.", 6.0
    ),
)


def default_ranges() -> dict[str, SignalRange]:
    """Uncalibrated fallback ranges."""
    return {k: SignalRange(lo=lo, hi=hi) for k, (lo, hi) in _DEFAULT_RANGES.items()}


def ranges_from_samples(
    samples: dict[str, list[float]],
    low_pct: float = 5.0,
    high_pct: float = 95.0,
    min_samples: int = 30,
) -> dict[str, SignalRange]:
    """Derive per-signal ranges from collected raw samples.

    Parameters
    ----------
    samples:
        Mapping of signal name -> list of raw observed values, pooled across all
        calibration steps.
    low_pct, high_pct:
        Percentiles used as range endpoints. Robust to landmark outliers.
    min_samples:
        Signals with fewer samples than this fall back to the default range,
        because a percentile over a handful of frames is noise.

    Returns
    -------
    dict
        Signal name -> :class:`~gesturebloom.control.mapper.SignalRange`.
    """
    out = default_ranges()
    for name, values in samples.items():
        arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
        if arr.size < min_samples:
            continue
        lo = float(np.percentile(arr, low_pct))
        hi = float(np.percentile(arr, high_pct))
        if hi - lo < 1e-4:
            continue
        out[name] = SignalRange(lo=lo, hi=hi)
    return out


@dataclass
class CalibrationProfile:
    """A saved calibration, with provenance."""

    ranges: dict[str, SignalRange]
    version: int = CALIBRATION_VERSION
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "note": self.note,
            "ranges": {
                k: {"lo": v.lo, "hi": v.hi, "inset": v.inset} for k, v in self.ranges.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> CalibrationProfile:
        version = int(payload.get("version", 0))
        if version != CALIBRATION_VERSION:
            raise ValueError(
                f"calibration profile version {version} does not match "
                f"current version {CALIBRATION_VERSION}; please re-run "
                f"`gesturebloom calibrate`"
            )
        ranges = {
            k: SignalRange(lo=float(v["lo"]), hi=float(v["hi"]), inset=float(v.get("inset", 0.06)))
            for k, v in payload.get("ranges", {}).items()
        }
        return cls(ranges=ranges, version=version, note=str(payload.get("note", "")))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> CalibrationProfile:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class CalibrationCollector:
    """Accumulates raw signal samples during the calibration flow.

    Deliberately decoupled from the capture loop so it can be driven by a
    webcam, by a recorded ``.npz``, or by synthetic data in tests.

    Examples
    --------
    >>> col = CalibrationCollector()
    >>> for _ in range(40):
    ...     col.add({"pinch": 0.2, "openness": 1.0})
    >>> for _ in range(40):
    ...     col.add({"pinch": 0.8, "openness": 1.6})
    >>> profile = col.finish()
    >>> round(profile.ranges["pinch"].lo, 2), round(profile.ranges["pinch"].hi, 2)
    (0.2, 0.8)
    """

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = {name: [] for name in SIGNAL_NAMES}
        self._samples["yaw"] = []

    def add(self, raw: dict[str, float]) -> None:
        for name, value in raw.items():
            self._samples.setdefault(name, []).append(float(value))

    @property
    def counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self._samples.items()}

    def finish(self, note: str = "") -> CalibrationProfile:
        return CalibrationProfile(ranges=ranges_from_samples(self._samples), note=note)
