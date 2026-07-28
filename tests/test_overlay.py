"""Overlay tests.

Skipped when OpenCV is absent, since it is an optional ``[live]`` dependency and
the core test suite must stay installable with numpy alone.

The emphasis here is on *not crashing* rather than pixel-exactness. Overlay code
runs every frame on data that regularly goes out of range -- a hand at the edge of
frame produces negative and >1 normalized coordinates -- and a crash there kills
the whole app. Exact pixel colours are a styling choice; robustness is not.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="opencv-python is an optional [live] dependency")

from gesturebloom.render.overlay import (  # noqa: E402
    HAND_CONNECTIONS,
    draw_hud,
    draw_param_labels,
    draw_skeleton,
    wrist_anchor_ndc,
)

PARAMS = {"grow": 0.74, "bloom": 0.68, "sway": 0.5}


@pytest.fixture
def frame() -> np.ndarray:
    return np.full((360, 640, 3), 40, dtype=np.uint8)


@pytest.fixture
def landmarks() -> np.ndarray:
    rng = np.random.default_rng(0)
    lm = rng.uniform(0.3, 0.7, size=(21, 3))
    lm[:, 2] = rng.uniform(-0.1, 0.1, size=21)
    return lm


def test_skeleton_topology() -> None:
    """Every connection must reference valid landmark indices, and no self-loops."""
    assert len(HAND_CONNECTIONS) == 21
    for a, b in HAND_CONNECTIONS:
        assert 0 <= a < 21 and 0 <= b < 21
        assert a != b
    # Every landmark should appear in at least one connection, or it renders as
    # an orphaned dot with no visible link to the hand.
    covered = {i for pair in HAND_CONNECTIONS for i in pair}
    assert covered == set(range(21))


def test_skeleton_draws_pixels(frame, landmarks) -> None:
    before = frame.copy()
    out = draw_skeleton(frame, landmarks)
    assert out is frame  # documented as in-place
    assert (frame != before).any()


def test_labels_and_hud_draw(frame, landmarks) -> None:
    draw_param_labels(frame, landmarks, PARAMS)
    draw_hud(frame, PARAMS, fps=60.0, tracking=True, backend="tasks")
    assert (frame > 240).all(axis=2).any(), "expected white text pixels"


@pytest.mark.parametrize("offset", [-8.0, -1.5, 1.5, 8.0])
def test_off_frame_landmarks_do_not_crash(frame, landmarks, offset: float) -> None:
    """A hand at or beyond the frame edge is routine, not exceptional."""
    lm = landmarks.copy()
    lm[:, :2] += offset
    draw_skeleton(frame, lm)
    draw_param_labels(frame, lm, PARAMS)


def test_degenerate_landmarks_do_not_crash(frame) -> None:
    """All-identical landmarks give zero depth span -- must not divide by zero."""
    draw_skeleton(frame, np.full((21, 3), 0.5))


def test_hud_without_tracking(frame) -> None:
    draw_hud(frame, {}, fps=None, tracking=False)
    assert (frame > 240).all(axis=2).any()


def test_hud_handles_missing_and_extreme_params(frame) -> None:
    draw_hud(frame, {"grow": 0.0, "bloom": 1.0}, fps=0.0, tracking=True)
    draw_hud(frame, {"grow": -5.0, "bloom": 99.0}, fps=1e6, tracking=True)


def test_anchor_ndc_corners() -> None:
    """Image space is y-down in [0,1]; NDC is y-up in [-1,1]."""
    lm = np.zeros((21, 3))
    for image_xy, expected in [
        ((0.5, 0.5), (0.0, 0.0)),
        ((0.0, 0.0), (-1.0, 1.0)),   # top-left -> NDC top-left
        ((1.0, 1.0), (1.0, -1.0)),   # bottom-right -> NDC bottom-right
    ]:
        lm[0, :2] = image_xy
        assert wrist_anchor_ndc(lm) == pytest.approx(expected)


def test_anchor_y_offset_applied() -> None:
    lm = np.zeros((21, 3))
    lm[0, :2] = (0.5, 0.5)
    assert wrist_anchor_ndc(lm, y_offset=-0.25)[1] == pytest.approx(-0.25)


def test_overlay_does_not_mutate_landmarks(frame, landmarks) -> None:
    """The overlay must treat landmarks as read-only; they are used downstream."""
    original = landmarks.copy()
    draw_skeleton(frame, landmarks)
    draw_param_labels(frame, landmarks, PARAMS)
    np.testing.assert_array_equal(landmarks, original)
