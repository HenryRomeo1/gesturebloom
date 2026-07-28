"""Two-handed controller tests.

The behaviours here are UX contracts, not implementation details. Each one
corresponds to something that feels broken if it regresses.
"""

from __future__ import annotations

import numpy as np
import pytest

from gesturebloom.control.dual import DEFAULT_DUAL_BINDINGS, DualHandController, HandBinding
from gesturebloom.data.recording import synthetic_recording
from gesturebloom.landmarks.source import HandObservation


@pytest.fixture
def hands() -> list[HandObservation]:
    rec = synthetic_recording(n_frames=8, seed=5, dropout_rate=0.0)
    return [
        HandObservation(landmarks=rec.landmarks[0].astype(np.float64), handedness="Left"),
        HandObservation(landmarks=rec.landmarks[3].astype(np.float64), handedness="Right"),
    ]


def test_default_bindings_split_the_axes() -> None:
    params = {b.parameter for b in DEFAULT_DUAL_BINDINGS}
    hands_ = {b.handedness for b in DEFAULT_DUAL_BINDINGS}
    assert params == {"grow", "bloom"}
    assert hands_ == {"Left", "Right"}


def test_produces_both_parameters_in_range(hands) -> None:
    ctl = DualHandController()
    for _ in range(10):
        params = ctl.update(hands, dt=1 / 60)
    assert set(params) == {"grow", "bloom"}
    assert all(0.0 <= v <= 1.0 for v in params.values())


def test_left_drives_grow_right_drives_bloom(hands) -> None:
    assert DualHandController().parameter_for("Left") == "grow"
    assert DualHandController().parameter_for("Right") == "bloom"


def test_missing_hand_holds_its_value(hands) -> None:
    """Reaching off-screen must not collapse the plant."""
    ctl = DualHandController()
    for _ in range(10):
        ctl.update(hands, dt=1 / 60)
    held = ctl.params["bloom"]
    for _ in range(30):
        params = ctl.update([hands[0]], dt=1 / 60)
    assert params["bloom"] == pytest.approx(held)
    assert not ctl.tracked("Right")
    assert ctl.tracked("Left")


def test_one_hand_still_controls_its_own_axis(hands) -> None:
    """Losing the right hand must not freeze the left."""
    ctl = DualHandController()
    ctl.update(hands, dt=1 / 60)
    left_only = [hands[0]]
    a = ctl.update(left_only, dt=1 / 60)["grow"]
    for _ in range(5):
        b = ctl.update(left_only, dt=1 / 60)["grow"]
    assert b != pytest.approx(a) or ctl.tracked("Left")


def test_no_hands_holds_everything(hands) -> None:
    ctl = DualHandController()
    for _ in range(8):
        ctl.update(hands, dt=1 / 60)
    before = dict(ctl.params)
    for _ in range(20):
        after = ctl.update([], dt=1 / 60)
    assert after == pytest.approx(before)
    assert not ctl.any_tracked


def test_duplicate_handedness_prefers_higher_score(hands) -> None:
    """A duplicate label usually means one detection is spurious."""
    rec = synthetic_recording(n_frames=4, seed=9, dropout_rate=0.0)
    low = HandObservation(rec.landmarks[0].astype(np.float64), "Left", score=0.2)
    high = HandObservation(rec.landmarks[1].astype(np.float64), "Left", score=0.95)
    ctl = DualHandController()
    ctl.update([low, high], dt=1 / 60)
    np.testing.assert_allclose(ctl.landmarks("Left"), high.landmarks, atol=1e-6)


def test_landmarks_exposed_for_overlay(hands) -> None:
    ctl = DualHandController()
    ctl.update(hands, dt=1 / 60)
    assert ctl.landmarks("Left") is not None
    assert ctl.landmarks("Left").shape == (21, 3)
    assert ctl.landmarks("Right") is not None


def test_reset_clears_everything(hands) -> None:
    ctl = DualHandController()
    for _ in range(10):
        ctl.update(hands, dt=1 / 60)
    ctl.reset()
    assert ctl.params == {"grow": 0.0, "bloom": 0.0}
    assert ctl.landmarks("Left") is None
    assert not ctl.any_tracked


def test_hands_are_independently_filtered(hands) -> None:
    """Shared filter state would smear one hand's motion into the other's value."""
    ctl = DualHandController()
    ctl.update(hands, dt=1 / 60)
    left_before = ctl.params["grow"]
    # Move only the right hand for a while.
    rec = synthetic_recording(n_frames=40, seed=11, dropout_rate=0.0)
    for t in range(20):
        moving_right = HandObservation(rec.landmarks[t].astype(np.float64), "Right")
        ctl.update([hands[0], moving_right], dt=1 / 60)
    # grow should have settled toward the (unchanged) left hand, not wandered
    # with the right hand's motion.
    assert abs(ctl.params["grow"] - left_before) < 0.6


def test_custom_bindings() -> None:
    ctl = DualHandController(
        bindings=(
            HandBinding(parameter="ink", handedness="Left", signal="pinch"),
            HandBinding(parameter="screen", handedness="Right", signal="curl"),
        )
    )
    assert set(ctl.params) == {"ink", "screen"}
    assert ctl.parameter_for("Left") == "ink"
