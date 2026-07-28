"""Two-handed control: left hand drives ``grow``, right hand drives ``bloom``.

One hand cannot comfortably drive two independent continuous parameters. Any
single-hand scheme has to overload one gesture -- openness for one axis, pinch for
the other -- and the two interfere, because opening your hand changes your pinch
aperture whether you meant it to or not. Splitting across hands makes the axes
genuinely independent, which is why the reference uses two.

Three behaviours here are UX decisions rather than implementation details:

**Each hand gets its own filter and mapper state.** Sharing a One Euro filter
between hands would make one hand's motion smear into the other's value, because
the filter's velocity estimate would see the difference between two unrelated
hands as enormous speed.

**A missing hand holds its last value.** Reset-to-zero is the obvious
implementation and it is badly wrong: reaching off-screen, or briefly occluding a
hand, would collapse the plant and then re-grow it. Holding means you can set
``grow``, drop that hand, and adjust ``bloom`` with the other -- which is how
people actually want to use it.

**Handedness assignment is configurable and mirror-aware.** Because the preview
is mirrored, "left hand" means the hand that *appears* on the left of the
mirrored image, which is the user's own left hand. That is the intuitive reading,
and it is only correct because
:class:`~gesturebloom.landmarks.source.WebcamSource` already un-swaps the labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..landmarks.canonical import try_canonicalize
from ..landmarks.filters import LandmarkSmoother
from .calibration import default_ranges
from .mapper import SignalRange, raw_signals


@dataclass
class HandBinding:
    """Which raw signal from which hand drives which output parameter."""

    parameter: str
    handedness: str
    signal: str = "openness"
    gamma: float = 1.0
    invert: bool = False
    min_cutoff: float = 1.6
    beta: float = 0.04


DEFAULT_DUAL_BINDINGS: tuple[HandBinding, ...] = (
    HandBinding(parameter="grow", handedness="Left", signal="openness", gamma=1.0),
    HandBinding(parameter="bloom", handedness="Right", signal="openness", gamma=1.15),
)


@dataclass
class _HandState:
    """Per-hand filtering and last-known values."""

    smoother: LandmarkSmoother
    value_filter: object
    last_value: float = 0.0
    landmarks: np.ndarray | None = None
    tracked: bool = False
    missing_frames: int = 0


@dataclass
class DualHandController:
    """Turn a list of hand observations into ``{grow, bloom}``.

    Examples
    --------
    >>> import numpy as np
    >>> from gesturebloom.landmarks.source import HandObservation
    >>> from gesturebloom.data.recording import synthetic_recording
    >>> rec = synthetic_recording(n_frames=4, seed=1, dropout_rate=0.0)
    >>> ctl = DualHandController()
    >>> left = HandObservation(landmarks=rec.landmarks[0].astype(float), handedness="Left")
    >>> right = HandObservation(landmarks=rec.landmarks[1].astype(float), handedness="Right")
    >>> params = ctl.update([left, right], dt=1 / 60)
    >>> sorted(params)
    ['bloom', 'grow']
    >>> all(0.0 <= v <= 1.0 for v in params.values())
    True

    A hand disappearing holds its value rather than collapsing it:

    >>> _ = ctl.update([left, right], dt=1 / 60)
    >>> before = ctl.params["bloom"]
    >>> after = ctl.update([left], dt=1 / 60)["bloom"]
    >>> after == before
    True
    >>> ctl.tracked("Right")
    False
    """

    ranges: dict[str, SignalRange] = field(default_factory=default_ranges)
    bindings: tuple[HandBinding, ...] = DEFAULT_DUAL_BINDINGS
    freq: float = 60.0
    hold_frames: int = 8
    _states: dict[str, _HandState] = field(default_factory=dict, init=False, repr=False)
    params: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        from ..landmarks.filters import OneEuroFilter

        for binding in self.bindings:
            self._states[binding.handedness] = _HandState(
                smoother=LandmarkSmoother(freq=self.freq, hold_frames=self.hold_frames),
                value_filter=OneEuroFilter(
                    freq=self.freq, min_cutoff=binding.min_cutoff, beta=binding.beta
                ),
            )
            self.params[binding.parameter] = 0.0

    def update(self, observations: list, dt: float | None = None) -> dict[str, float]:
        """Advance one frame. Returns the current parameter dict.

        Parameters
        ----------
        observations:
            List of :class:`~gesturebloom.landmarks.source.HandObservation`. May
            be empty, may contain one or both hands, and may contain duplicates
            of the same handedness if the detector misjudges -- in which case the
            highest-scoring one wins, since a duplicate label usually means one
            detection is spurious.
        """
        by_hand: dict[str, object] = {}
        for obs in observations:
            existing = by_hand.get(obs.handedness)
            if existing is None or obs.score > existing.score:  # type: ignore[attr-defined]
                by_hand[obs.handedness] = obs

        for binding in self.bindings:
            state = self._states[binding.handedness]
            obs = by_hand.get(binding.handedness)

            if obs is None:
                state.missing_frames += 1
                state.tracked = False
                if state.missing_frames > self.hold_frames:
                    state.landmarks = None
                # Value deliberately held -- see the module docstring.
                continue

            state.missing_frames = 0
            smoothed = state.smoother.update(obs.landmarks, dt)  # type: ignore[attr-defined]
            if smoothed is None:
                state.tracked = False
                continue

            result = try_canonicalize(smoothed, obs.handedness)  # type: ignore[attr-defined]
            if result is None:
                state.tracked = False
                continue

            canonical, _frame = result
            state.landmarks = smoothed
            state.tracked = True

            raw = raw_signals(canonical).get(binding.signal)
            if raw is None:
                continue
            rng = self.ranges.get(binding.signal)
            t = rng.normalize(raw) if rng is not None else float(np.clip(raw, 0.0, 1.0))
            if binding.invert:
                t = 1.0 - t
            if binding.gamma != 1.0:
                t = float(np.power(max(t, 0.0), binding.gamma))
            value = float(np.clip(state.value_filter(t, dt=dt), 0.0, 1.0))  # type: ignore[operator]
            state.last_value = value
            self.params[binding.parameter] = value

        return dict(self.params)

    # ---- introspection for the overlay ---------------------------------- #
    def landmarks(self, handedness: str) -> np.ndarray | None:
        """Smoothed image-space landmarks for one hand, or ``None``."""
        state = self._states.get(handedness)
        return None if state is None else state.landmarks

    def tracked(self, handedness: str) -> bool:
        state = self._states.get(handedness)
        return bool(state and state.tracked)

    @property
    def any_tracked(self) -> bool:
        return any(s.tracked for s in self._states.values())

    def parameter_for(self, handedness: str) -> str | None:
        for binding in self.bindings:
            if binding.handedness == handedness:
                return binding.parameter
        return None

    def reset(self) -> None:
        for state in self._states.values():
            state.smoother.reset()
            state.value_filter.reset()  # type: ignore[attr-defined]
            state.last_value = 0.0
            state.landmarks = None
            state.tracked = False
            state.missing_frames = 0
        for binding in self.bindings:
            self.params[binding.parameter] = 0.0
