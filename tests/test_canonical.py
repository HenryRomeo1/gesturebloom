"""Invariance tests for canonicalization.

These are property tests, not example tests, and that distinction matters: the
whole value of canonicalization is a set of *invariances*, so the tests assert
those invariances directly over randomized input rather than checking a few
hand-picked vectors.
"""

from __future__ import annotations

import numpy as np
import pytest

from gesturebloom.landmarks.canonical import (
    MIDDLE_MCP,
    WRIST,
    canonicalize,
    feature_vector,
    try_canonicalize,
)


def random_hand(rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(21, 3))


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    ax = rng.normal(size=3)
    ax /= np.linalg.norm(ax)
    ang = rng.uniform(0, 2 * np.pi)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)


@pytest.mark.parametrize("seed", range(20))
def test_invariant_to_similarity_transform(seed: int) -> None:
    """Canonical output must be identical under rotation, scale, and translation."""
    rng = np.random.default_rng(seed)
    lm = random_hand(rng)
    c0, _ = canonicalize(lm)

    transformed = lm @ random_rotation(rng).T * rng.uniform(0.2, 8.0) + rng.uniform(-20, 20, 3)
    c1, _ = canonicalize(transformed)

    np.testing.assert_allclose(c0, c1, atol=1e-9)


@pytest.mark.parametrize("seed", range(10))
def test_fixed_points(seed: int) -> None:
    """Wrist lands at the origin, middle-MCP at (0, 1, 0), by construction."""
    c, _ = canonicalize(random_hand(np.random.default_rng(seed)))
    np.testing.assert_allclose(c[WRIST], [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(c[MIDDLE_MCP], [0, 1, 0], atol=1e-12)


@pytest.mark.parametrize("seed", range(10))
def test_frame_roundtrip(seed: int) -> None:
    """The stored HandFrame must invert the transform exactly."""
    lm = random_hand(np.random.default_rng(seed))
    c, frame = canonicalize(lm)
    np.testing.assert_allclose(frame.to_source(c), lm, atol=1e-9)


@pytest.mark.parametrize("seed", range(10))
def test_chirality_normalization(seed: int) -> None:
    """A mirrored left hand must canonicalize to the same vector as the right."""
    lm = random_hand(np.random.default_rng(seed))
    right, _ = canonicalize(lm, "Right")
    left, _ = canonicalize(lm * np.array([-1, 1, 1]), "Left")
    np.testing.assert_allclose(right, left, atol=1e-12)


def test_degenerate_inputs_rejected() -> None:
    """Degenerate poses raise, and try_canonicalize converts that to None."""
    collapsed = np.zeros((21, 3))
    with pytest.raises(ValueError):
        canonicalize(collapsed)
    assert try_canonicalize(collapsed) is None

    with pytest.raises(ValueError):
        canonicalize(np.full((21, 3), np.nan))
    with pytest.raises(ValueError):
        canonicalize(np.zeros((20, 3)))


def test_feature_vector_shape() -> None:
    c, _ = canonicalize(random_hand(np.random.default_rng(0)))
    assert feature_vector(c, drop_wrist=True).shape == (60,)
    assert feature_vector(c, drop_wrist=False).shape == (63,)
