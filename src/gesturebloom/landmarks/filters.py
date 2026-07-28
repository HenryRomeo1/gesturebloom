"""Signal smoothing for landmark streams and derived control values.

MediaPipe landmarks jitter by a few pixels frame to frame even on a perfectly
still hand. Left unsmoothed, that jitter is *visible* in the render as a
constant shimmer, and it destroys onset detection by producing spurious
threshold crossings.

The naive fix -- an exponential moving average -- trades jitter for lag at a
fixed ratio, so you must choose between a shimmering render and a render that
feels unresponsive. The One Euro filter (Casiez, Roussel & Vogel, CHI 2012)
breaks that tradeoff with a single idea: make the cutoff frequency a function of
the estimated speed of the signal. Slow motion gets aggressive smoothing;
fast motion gets almost none. Two intuitive parameters:

``min_cutoff``
    Cutoff at zero velocity. Lower = smoother when still. Tune this first,
    watching a stationary hand until the shimmer disappears.
``beta``
    Velocity coupling. Higher = less lag during fast motion. Raise this second,
    watching a fast gesture until the lag disappears.
"""

from __future__ import annotations

import math

import numpy as np


class _LowPass:
    """First-order low-pass with externally supplied smoothing factor."""

    __slots__ = ("_initialized", "_y")

    def __init__(self) -> None:
        self._y: np.ndarray | float = 0.0
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def last(self):
        return self._y

    def __call__(self, x, alpha: float):
        if not self._initialized:
            self._y = x
            self._initialized = True
        else:
            self._y = alpha * x + (1.0 - alpha) * self._y
        return self._y

    def reset(self) -> None:
        self._y = 0.0
        self._initialized = False


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuroFilter:
    """Adaptive low-pass filter. Works on scalars or arbitrary-shaped arrays.

    Parameters
    ----------
    freq:
        Nominal sample rate in Hz, used only until the first real ``dt``.
    min_cutoff:
        Cutoff frequency (Hz) at zero velocity.
    beta:
        Velocity-to-cutoff coupling coefficient.
    d_cutoff:
        Cutoff for the derivative estimate itself. Rarely needs changing.

    Examples
    --------
    >>> f = OneEuroFilter(freq=60.0, min_cutoff=1.5, beta=0.05)
    >>> _ = f(np.zeros(3), dt=1 / 60)
    >>> out = f(np.array([1.0, 0.0, 0.0]), dt=1 / 60)
    >>> bool(0.0 < out[0] < 1.0)
    True
    """

    def __init__(
        self,
        freq: float = 60.0,
        min_cutoff: float = 1.5,
        beta: float = 0.05,
        d_cutoff: float = 1.0,
    ) -> None:
        if freq <= 0:
            raise ValueError("freq must be positive")
        self.freq = float(freq)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x = _LowPass()
        self._dx = _LowPass()

    def __call__(self, x, dt: float | None = None):
        if dt is None or dt <= 0:
            dt = 1.0 / self.freq

        x_arr = np.asarray(x, dtype=np.float64) if not np.isscalar(x) else float(x)

        if self._x.initialized:
            dx = (x_arr - self._x.last) / dt
        else:
            dx = np.zeros_like(x_arr) if not np.isscalar(x_arr) else 0.0

        dx_hat = self._dx(dx, _alpha(self.d_cutoff, dt))
        speed = np.abs(dx_hat)
        cutoff = self.min_cutoff + self.beta * speed

        # Array-valued cutoff means a per-component alpha; compute elementwise so
        # a fast-moving fingertip is not smoothed at the same rate as a still one.
        if np.isscalar(x_arr):
            return self._x(x_arr, _alpha(float(cutoff), dt))

        tau = 1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-6))
        alpha = 1.0 / (1.0 + tau / max(dt, 1e-6))
        if not self._x.initialized:
            return self._x(x_arr, 1.0)
        y = alpha * x_arr + (1.0 - alpha) * self._x.last
        self._x._y = y
        return y

    def reset(self) -> None:
        self._x.reset()
        self._dx.reset()


class LandmarkSmoother:
    """One Euro filter applied to a full ``(21, 3)`` landmark array.

    Also handles short tracking dropouts by holding the last good pose for up to
    ``hold_frames``, which prevents the flower from snapping to zero when
    MediaPipe misses a frame.
    """

    def __init__(
        self,
        freq: float = 60.0,
        min_cutoff: float = 1.5,
        beta: float = 0.05,
        hold_frames: int = 6,
    ) -> None:
        self._filter = OneEuroFilter(freq=freq, min_cutoff=min_cutoff, beta=beta)
        self.hold_frames = int(hold_frames)
        self._last: np.ndarray | None = None
        self._missing = 0

    def update(self, landmarks: np.ndarray | None, dt: float | None = None) -> np.ndarray | None:
        """Return smoothed landmarks, or ``None`` if tracking is truly lost."""
        if landmarks is None:
            self._missing += 1
            if self._last is not None and self._missing <= self.hold_frames:
                return self._last
            self.reset()
            return None

        self._missing = 0
        smoothed = self._filter(np.asarray(landmarks, dtype=np.float64), dt=dt)
        self._last = np.asarray(smoothed, dtype=np.float64)
        return self._last

    def reset(self) -> None:
        self._filter.reset()
        self._last = None
        self._missing = 0
