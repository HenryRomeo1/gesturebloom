"""Onset-detection tests built from synthetic probability traces.

Each test targets one specific failure mode the state machine exists to prevent,
so a regression tells you *which* mechanism broke rather than just that
detection got worse.
"""

from __future__ import annotations

import numpy as np
import pytest

from gesturebloom.models.spotter import (
    GestureEvent,
    OnsetSpotter,
    SpotterConfig,
    evaluate_events,
    spot_offline,
)


def trace(segments: list[tuple[int, int, float]], n_classes: int = 2, length: int = 60) -> np.ndarray:
    """Build a (T, C) probability trace from (start, stop, prob) segments on class 1."""
    probs = np.zeros((length, n_classes))
    probs[:, 0] = 0.95
    probs[:, 1:] = 0.05 / max(n_classes - 1, 1)
    for start, stop, p in segments:
        probs[start:stop, 1] = p
        probs[start:stop, 0] = 1.0 - p
    return probs


def test_clean_gesture_fires_once() -> None:
    events = spot_offline(trace([(20, 35, 0.9)]), SpotterConfig(min_frames=3))
    assert len(events) == 1
    assert events[0].class_index == 1


def test_noise_spike_rejected_by_persistence() -> None:
    """A 2-frame spike must not fire when min_frames is 4."""
    events = spot_offline(trace([(20, 22, 0.95)]), SpotterConfig(min_frames=4))
    assert events == []


def test_hysteresis_prevents_double_fire_on_dip() -> None:
    """A mid-gesture dip that stays above exit_threshold must not re-trigger."""
    probs = trace([(20, 30, 0.9), (30, 33, 0.5), (33, 45, 0.9)])
    events = spot_offline(probs, SpotterConfig(enter_threshold=0.65, exit_threshold=0.4, min_frames=3))
    assert len(events) == 1


def test_refractory_blocks_immediate_retrigger() -> None:
    """Two gestures closer together than the refractory period yield one event."""
    probs = trace([(20, 28, 0.9), (30, 40, 0.9)])
    cfg = SpotterConfig(min_frames=3, refractory_frames=25, exit_threshold=0.4)
    assert len(spot_offline(probs, cfg)) == 1


def test_separated_gestures_both_fire() -> None:
    """Past the refractory period, both gestures must be detected."""
    probs = trace([(10, 20, 0.9), (60, 72, 0.9)], length=100)
    cfg = SpotterConfig(min_frames=3, refractory_frames=10, exit_threshold=0.4)
    assert len(spot_offline(probs, cfg)) == 2


def test_class_flapping_does_not_fire_wrong_class() -> None:
    """Alternating argmax between two gesture classes must not fire either."""
    probs = np.zeros((40, 3))
    for t in range(40):
        probs[t] = [0.1, 0.75, 0.15] if t % 2 else [0.1, 0.15, 0.75]
    assert spot_offline(probs, SpotterConfig(min_frames=4)) == []


def test_background_never_fires() -> None:
    probs = np.tile([[0.99, 0.01]], (200, 1))
    assert spot_offline(probs) == []


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        SpotterConfig(enter_threshold=0.3, exit_threshold=0.6)  # inverted
    with pytest.raises(ValueError):
        SpotterConfig(min_frames=0)


def test_online_matches_offline() -> None:
    """Streaming and batch must agree -- the causal-model guarantee."""
    probs = trace([(15, 30, 0.88), (60, 75, 0.92)], length=100)
    cfg = SpotterConfig(min_frames=3, refractory_frames=10)
    offline = spot_offline(probs, cfg)

    spotter = OnsetSpotter(cfg)
    online = [e for t in range(len(probs)) if (e := spotter.update(probs[t])) is not None]
    assert [(e.class_index, e.frame) for e in offline] == [(e.class_index, e.frame) for e in online]


def test_metrics_latency_and_counts() -> None:
    predicted = [GestureEvent(1, 12, 0.9, 3), GestureEvent(1, 80, 0.8, 3)]
    m = evaluate_events(predicted, [(1, 10), (1, 200)], tolerance_frames=5)
    assert (m.true_positives, m.false_positives, m.false_negatives) == (1, 1, 1)
    assert m.latencies == [2]
    assert 0.0 < m.f1 < 1.0
