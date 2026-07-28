"""Onset detection: turning a probability trace into discrete gesture events.

The model emits a probability per class per frame. Naively thresholding that
trace at 0.5 produces a mess, and understanding why is most of the value of this
module:

- **Chatter.** Near the threshold, probabilities cross back and forth every
  frame, firing dozens of events for one gesture.
- **Double-fires.** A gesture with a probability dip in the middle registers as
  two gestures.
- **Class flapping.** During a transition, two classes trade the argmax for a
  few frames, so you get "swipe, pinch, swipe" for one motion.

The fix is a state machine with three mechanisms, each targeting one failure:

**Hysteresis** (two thresholds). Fire on ``enter_threshold``, but do not release
until the probability falls below the lower ``exit_threshold``. Kills chatter,
because once triggered the signal must fall substantially to reset.

**Onset persistence** (``min_frames``). Require the enter condition to hold for
``k`` consecutive frames. Kills single-frame spikes from landmark noise.

**Refractory period.** After firing, ignore new onsets of the same class for
``refractory_frames``. Kills double-fires from mid-gesture dips.

This is deliberately pure numpy with no torch dependency, so it is trivially
unit-testable against synthetic probability traces -- which is exactly how you
should tune it. Do not tune thresholds against a live webcam; record a trace
once, then iterate against it offline in a second per experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SpotterState(Enum):
    IDLE = "idle"
    ARMING = "arming"
    ACTIVE = "active"
    REFRACTORY = "refractory"


@dataclass(frozen=True)
class GestureEvent:
    """A detected gesture onset."""

    class_index: int
    frame: int
    confidence: float
    arming_frames: int
    """How many frames the enter condition held before firing. Useful as a
    quality signal: events that barely cleared ``min_frames`` are the ones to
    inspect when tuning."""


@dataclass
class SpotterConfig:
    enter_threshold: float = 0.65
    exit_threshold: float = 0.40
    min_frames: int = 3
    refractory_frames: int = 12
    background_index: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.exit_threshold < self.enter_threshold < 1.0:
            raise ValueError("require 0 < exit_threshold < enter_threshold < 1")
        if self.min_frames < 1:
            raise ValueError("min_frames must be >= 1")


class OnsetSpotter:
    """Streaming onset detector. Feed it one probability vector per frame.

    Examples
    --------
    A clean gesture fires exactly once:

    >>> import numpy as np
    >>> spotter = OnsetSpotter(SpotterConfig(min_frames=2, refractory_frames=5))
    >>> trace = [[0.9, 0.1]] * 3 + [[0.1, 0.9]] * 10 + [[0.9, 0.1]] * 10
    >>> events = [e for p in trace for e in [spotter.update(np.array(p))] if e]
    >>> len(events)
    1
    >>> events[0].class_index
    1

    Single-frame noise spikes are rejected by onset persistence:

    >>> spotter = OnsetSpotter(SpotterConfig(min_frames=3))
    >>> trace = [[0.9, 0.1], [0.1, 0.9], [0.9, 0.1], [0.1, 0.9], [0.9, 0.1]]
    >>> [e for p in trace for e in [spotter.update(np.array(p))] if e]
    []

    A mid-gesture dip does not double-fire, thanks to hysteresis:

    >>> spotter = OnsetSpotter(SpotterConfig(min_frames=2, exit_threshold=0.4))
    >>> trace = [[0.1, 0.9]] * 4 + [[0.45, 0.55]] * 3 + [[0.1, 0.9]] * 4
    >>> len([e for p in trace for e in [spotter.update(np.array(p))] if e])
    1
    """

    def __init__(self, config: SpotterConfig | None = None) -> None:
        self.config = config or SpotterConfig()
        self.state = SpotterState.IDLE
        self._frame = -1
        self._arming_class = -1
        self._arming_count = 0
        self._active_class = -1
        self._refractory_left = 0

    def update(self, probs: np.ndarray) -> GestureEvent | None:
        """Advance one frame. Returns an event on the frame an onset fires."""
        cfg = self.config
        p = np.asarray(probs, dtype=np.float64).reshape(-1)
        self._frame += 1

        if self._refractory_left > 0:
            self._refractory_left -= 1
            if self._refractory_left == 0 and self.state is SpotterState.REFRACTORY:
                self.state = SpotterState.IDLE

        candidate = int(np.argmax(p))
        conf = float(p[candidate])
        is_gesture = candidate != cfg.background_index

        if self.state is SpotterState.ACTIVE:
            # Hold until the active class decays below the exit threshold.
            if float(p[self._active_class]) < cfg.exit_threshold:
                self.state = (
                    SpotterState.REFRACTORY if self._refractory_left > 0 else SpotterState.IDLE
                )
                self._active_class = -1
            return None

        if self.state is SpotterState.REFRACTORY:
            return None

        if is_gesture and conf >= cfg.enter_threshold:
            if candidate == self._arming_class:
                self._arming_count += 1
            else:
                self._arming_class = candidate
                self._arming_count = 1
            self.state = SpotterState.ARMING

            if self._arming_count >= cfg.min_frames:
                event = GestureEvent(
                    class_index=candidate,
                    frame=self._frame,
                    confidence=conf,
                    arming_frames=self._arming_count,
                )
                self.state = SpotterState.ACTIVE
                self._active_class = candidate
                self._refractory_left = cfg.refractory_frames
                self._arming_class = -1
                self._arming_count = 0
                return event
            return None

        self._arming_class = -1
        self._arming_count = 0
        if self.state is SpotterState.ARMING:
            self.state = SpotterState.IDLE
        return None

    def reset(self) -> None:
        self.__init__(self.config)  # type: ignore[misc]


def spot_offline(probs: np.ndarray, config: SpotterConfig | None = None) -> list[GestureEvent]:
    """Run the spotter over a full ``(T, n_classes)`` trace.

    Use this for threshold tuning and for computing detection metrics. Because
    the TCN is causal, offline results here are identical to online behaviour.

    Examples
    --------
    >>> import numpy as np
    >>> tr = np.tile(np.array([[0.9, 0.1]]), (30, 1))
    >>> tr[10:20] = [0.1, 0.9]
    >>> len(spot_offline(tr))
    1
    """
    spotter = OnsetSpotter(config)
    events: list[GestureEvent] = []
    for t in range(probs.shape[0]):
        ev = spotter.update(probs[t])
        if ev is not None:
            events.append(ev)
    return events


@dataclass
class DetectionMetrics:
    """Event-level detection quality with latency.

    Frame accuracy is the wrong metric for spotting -- a model can score 95%
    frame accuracy while firing zero correct events. These are the numbers that
    belong in your README.
    """

    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    latencies: list[int] = field(default_factory=list)

    @property
    def precision(self) -> float:
        d = self.true_positives + self.false_positives
        return self.true_positives / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.true_positives + self.false_negatives
        return self.true_positives / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def median_latency_frames(self) -> float:
        return float(np.median(self.latencies)) if self.latencies else float("nan")

    def summary(self, fps: float = 60.0) -> str:
        lat_ms = self.median_latency_frames / fps * 1000.0
        return (
            f"precision={self.precision:.3f} recall={self.recall:.3f} "
            f"f1={self.f1:.3f} median_latency={self.median_latency_frames:.1f}f "
            f"({lat_ms:.0f} ms @ {fps:g} fps)"
        )


def evaluate_events(
    predicted: list[GestureEvent],
    ground_truth: list[tuple[int, int]],
    tolerance_frames: int = 15,
) -> DetectionMetrics:
    """Match predicted onsets to ground-truth onsets within a tolerance window.

    Parameters
    ----------
    predicted:
        Events from :func:`spot_offline`.
    ground_truth:
        ``(class_index, onset_frame)`` pairs.
    tolerance_frames:
        A prediction counts as a hit if it is the right class and lands within
        this many frames of the true onset. Report the tolerance you used --
        detection F1 is meaningless without it.

    Examples
    --------
    >>> ev = [GestureEvent(1, 12, 0.9, 3)]
    >>> m = evaluate_events(ev, [(1, 10)], tolerance_frames=5)
    >>> m.true_positives, m.false_positives, m.false_negatives
    (1, 0, 0)
    >>> m.latencies
    [2]
    """
    metrics = DetectionMetrics()
    unmatched = list(ground_truth)

    for ev in predicted:
        best, best_dist = None, None
        for gt in unmatched:
            gt_class, gt_frame = gt
            if gt_class != ev.class_index:
                continue
            dist = ev.frame - gt_frame
            if abs(dist) <= tolerance_frames and (best_dist is None or abs(dist) < abs(best_dist)):
                best, best_dist = gt, dist
        if best is not None:
            metrics.true_positives += 1
            metrics.latencies.append(int(best_dist))  # type: ignore[arg-type]
            unmatched.remove(best)
        else:
            metrics.false_positives += 1

    metrics.false_negatives = len(unmatched)
    return metrics
