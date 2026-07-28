"""Continuous control extraction: canonical landmarks -> named scalar signals.

This is the shared module. Project B (print emulation) imports the same
:class:`ControlMapper` and binds the same signals to different render
parameters -- which is the point of keeping the abstraction at "named
normalized scalars" rather than at "flower parameters".

Design decisions worth defending in the README:

**Regression, not classification, for continuous controls.** ``grow`` and
``bloom`` are not discrete states. Binning them into classes produces visible
stepping in the render and throws away the precision the landmarks already have.

**Raw measures are unbounded and user-specific.** A person with long fingers
produces a larger maximum pinch aperture than someone with short fingers, even
after hand-span normalization, because finger-length-to-palm-length ratio
varies. So every measure is *calibrated* per user into ``[0, 1]`` -- see
:mod:`gesturebloom.control.calibration`.

**Deadzones at both ends.** Without them the flower can never quite reach fully
closed or fully open, which reads as broken. The calibrated range is inset by a
small epsilon so the extremes are comfortably reachable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..landmarks.canonical import (
    FINGER_CHAINS,
    INDEX_MCP,
    INDEX_TIP,
    MIDDLE_MCP,
    PINKY_MCP,
    THUMB_TIP,
    WRIST,
)
from ..landmarks.filters import OneEuroFilter

#: Names of the raw measures this module produces, in a stable order.
SIGNAL_NAMES = (
    "pinch",
    "openness",
    "curl",
    "spread",
    "pitch",
    "roll",
)


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    c = float(np.dot(a, b)) / (na * nb)
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def raw_signals(canonical: np.ndarray) -> dict[str, float]:
    """Compute raw, uncalibrated control measures from canonical landmarks.

    All distances are in hand-span units (see
    :func:`~gesturebloom.landmarks.canonical.canonicalize`), so they are already
    camera-distance invariant. They are *not* yet user invariant.

    Returns
    -------
    dict
        Keys are :data:`SIGNAL_NAMES`.

        ``pinch``
            Thumb-tip to index-tip distance. The primary precision control.
        ``openness``
            Mean fingertip distance from the wrist. Coarse, high-range, robust.
        ``curl``
            Mean interphalangeal flexion across the four fingers, in radians.
            Complements ``openness``: distinguishes a fist from a flat hand
            held edge-on, which ``openness`` alone conflates.
        ``spread``
            Mean angle between adjacent finger direction vectors -- splay.
        ``pitch``
            Palm-normal elevation. Because the canonical frame removes hand
            rotation, this is recovered from the *pre-canonical* frame by the
            caller; here we approximate it from thumb-plane geometry so the
            function stays pure. See :func:`frame_signals` for the exact version.
        ``roll``
            Same caveat as ``pitch``.
    """
    c = np.asarray(canonical, dtype=np.float64)

    pinch = float(np.linalg.norm(c[THUMB_TIP] - c[INDEX_TIP]))

    tips = [chain[3] for chain in FINGER_CHAINS]
    openness = float(np.mean([np.linalg.norm(c[t] - c[WRIST]) for t in tips]))

    curls: list[float] = []
    for mcp, pip, dip, tip in FINGER_CHAINS:
        curls.append(
            _angle_between(c[pip] - c[mcp], c[dip] - c[pip])
            + _angle_between(c[dip] - c[pip], c[tip] - c[dip])
        )
    curl = float(np.mean(curls))

    dirs = []
    for mcp, _pip, _dip, tip in FINGER_CHAINS:
        v = c[tip] - c[mcp]
        n = float(np.linalg.norm(v))
        if n > 1e-9:
            dirs.append(v / n)
    if len(dirs) >= 2:
        spread = float(
            np.mean([_angle_between(dirs[i], dirs[i + 1]) for i in range(len(dirs) - 1)])
        )
    else:
        spread = 0.0

    # Placeholders; the honest versions need the pre-canonical frame.
    palm_span = c[INDEX_MCP] - c[PINKY_MCP]
    pitch = float(np.arctan2(c[MIDDLE_MCP][2], c[MIDDLE_MCP][1]))
    roll = float(np.arctan2(palm_span[2], palm_span[0]))

    return {
        "pinch": pinch,
        "openness": openness,
        "curl": curl,
        "spread": spread,
        "pitch": pitch,
        "roll": roll,
    }


def frame_signals(basis: np.ndarray) -> dict[str, float]:
    """Recover global hand orientation from the canonicalization basis.

    Canonicalization deliberately discards world rotation -- that is what makes
    the classifier robust. But rotation is *useful* as a control signal, so we
    read it back out of the :class:`~gesturebloom.landmarks.canonical.HandFrame`
    basis rather than trying to infer it from the canonical points.

    Parameters
    ----------
    basis:
        ``(3, 3)`` row-stacked ``[x, y, z]`` orthonormal basis in source coords.

    Returns
    -------
    dict
        ``pitch`` -- palm-normal elevation above the image plane, radians.
        ``roll`` -- rotation of the palm axis within the image plane, radians.
        ``yaw``  -- palm-normal azimuth, radians.
    """
    b = np.asarray(basis, dtype=np.float64)
    x_axis, y_axis, z_axis = b[0], b[1], b[2]
    pitch = float(np.arcsin(np.clip(-z_axis[1], -1.0, 1.0)))
    roll = float(np.arctan2(x_axis[1], y_axis[1]))
    yaw = float(np.arctan2(z_axis[0], z_axis[2]))
    return {"pitch": pitch, "roll": roll, "yaw": yaw}


@dataclass
class SignalRange:
    """Calibrated ``[lo, hi]`` range for one raw measure."""

    lo: float
    hi: float
    inset: float = 0.06

    def normalize(self, value: float) -> float:
        """Map a raw value into ``[0, 1]`` with symmetric deadzones."""
        span = self.hi - self.lo
        if span < 1e-9:
            return 0.0
        t = (float(value) - self.lo) / span
        # Inset the usable range so the extremes are comfortably reachable.
        t = (t - self.inset) / max(1.0 - 2.0 * self.inset, 1e-6)
        return float(np.clip(t, 0.0, 1.0))


@dataclass
class ControlBinding:
    """Binds one output parameter to one raw signal, with a response curve.

    Attributes
    ----------
    signal:
        Key from :data:`SIGNAL_NAMES` (or ``"yaw"``).
    gamma:
        Response exponent applied after normalization. ``> 1`` gives finer
        control near zero; ``< 1`` gives finer control near one. Bloom wants
        ``gamma > 1`` because the visually interesting range is the early
        opening.
    invert:
        Flip the mapping.
    smooth_min_cutoff, smooth_beta:
        Per-parameter One Euro settings. Bloom can tolerate more lag than grow,
        so it gets a lower cutoff and looks calmer.
    """

    signal: str
    gamma: float = 1.0
    invert: bool = False
    smooth_min_cutoff: float = 1.2
    smooth_beta: float = 0.03


DEFAULT_BINDINGS: dict[str, ControlBinding] = {
    "grow": ControlBinding(signal="openness", gamma=1.0, smooth_min_cutoff=2.0, smooth_beta=0.05),
    "bloom": ControlBinding(signal="pinch", gamma=1.6, invert=True, smooth_min_cutoff=1.2),
    "sway": ControlBinding(signal="roll", gamma=1.0, smooth_min_cutoff=0.8),
}


@dataclass
class ControlMapper:
    """Turn canonical landmarks into smoothed, normalized render parameters.

    Examples
    --------
    >>> import numpy as np
    >>> from gesturebloom.control.calibration import default_ranges
    >>> mapper = ControlMapper(ranges=default_ranges())
    >>> c = np.zeros((21, 3)); c[9] = [0, 1, 0]; c[8] = [0.1, 0.9, 0]; c[4] = [0.2, 0.5, 0]
    >>> params = mapper.update(c, dt=1 / 60)
    >>> sorted(params)
    ['bloom', 'grow', 'sway']
    >>> all(0.0 <= v <= 1.0 for v in params.values())
    True
    """

    ranges: dict[str, SignalRange]
    bindings: dict[str, ControlBinding] = field(default_factory=lambda: dict(DEFAULT_BINDINGS))
    freq: float = 60.0
    _filters: dict[str, OneEuroFilter] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for name, binding in self.bindings.items():
            self._filters[name] = OneEuroFilter(
                freq=self.freq,
                min_cutoff=binding.smooth_min_cutoff,
                beta=binding.smooth_beta,
            )

    def update(
        self,
        canonical: np.ndarray,
        dt: float | None = None,
        basis: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Compute one frame of render parameters, each in ``[0, 1]``."""
        raw = raw_signals(canonical)
        if basis is not None:
            raw.update(frame_signals(basis))

        out: dict[str, float] = {}
        for name, binding in self.bindings.items():
            value = raw.get(binding.signal)
            if value is None:
                out[name] = 0.0
                continue
            rng = self.ranges.get(binding.signal)
            t = rng.normalize(value) if rng is not None else float(np.clip(value, 0.0, 1.0))
            if binding.invert:
                t = 1.0 - t
            if binding.gamma != 1.0:
                t = float(np.power(t, binding.gamma))
            out[name] = float(np.clip(self._filters[name](t, dt=dt), 0.0, 1.0))
        return out

    def reset(self) -> None:
        for f in self._filters.values():
            f.reset()
