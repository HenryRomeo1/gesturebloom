"""Geometry tests asserting the two properties the animation design depends on.

If either of these breaks, the flower stops looking like a flower -- so they are
worth encoding as tests rather than trusting to visual inspection.
"""

from __future__ import annotations

import numpy as np
import pytest

from gesturebloom.geometry.spiderlily import (
    SpiderlilyParams,
    build_spiderlily,
    ribbonize,
)
from gesturebloom.render.window import build_batch


def arclength(points: np.ndarray) -> float:
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


@pytest.mark.parametrize("bloom", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_arclength_invariant_under_bloom(bloom: float) -> None:
    """Bending must not stretch. This is why we integrate a unit tangent."""
    params = SpiderlilyParams()
    reference = arclength(build_spiderlily(1.0, 0.0, params=params, seed=5)[0].points)
    actual = arclength(build_spiderlily(1.0, bloom, params=params, seed=5)[0].points)
    assert abs(actual - reference) < 1e-3


@pytest.mark.parametrize("grow", [0.2, 0.5, 0.8, 1.0])
def test_growth_scales_arclength_linearly(grow: float) -> None:
    """Arc length must be proportional to grow -- growth extends, never scales."""
    params = SpiderlilyParams()
    full = arclength(build_spiderlily(1.0, 0.5, params=params, seed=5)[0].points)
    partial = arclength(build_spiderlily(grow, 0.5, params=params, seed=5)[0].points)
    assert abs(partial - full * grow) < 1e-2


def test_growth_preserves_base() -> None:
    """The base of the strand must not move as the tip extends."""
    params = SpiderlilyParams()
    a = build_spiderlily(0.3, 0.6, params=params, seed=2)[0].points
    b = build_spiderlily(1.0, 0.6, params=params, seed=2)[0].points
    np.testing.assert_allclose(a[0], b[0], atol=1e-6)
    # The early portion of the curve should coincide, not merely start together.
    np.testing.assert_allclose(a[:5] / 0.3, b[:5] / 1.0, atol=2e-2)


def test_strand_counts_and_roles() -> None:
    strands = build_spiderlily(1.0, 1.0, seed=0)
    assert len(strands) == 12
    assert sum(s.role == "tepal" for s in strands) == 6
    assert sum(s.role == "stamen" for s in strands) == 6


def test_all_outputs_finite_across_parameter_sweep() -> None:
    """No NaN or inf anywhere in the parameter space, including the degenerate corners."""
    for grow in np.linspace(0.0, 1.0, 11):
        for bloom in np.linspace(0.0, 1.0, 11):
            for strand in build_spiderlily(float(grow), float(bloom), seed=1):
                assert np.all(np.isfinite(strand.points))
                assert np.all(np.isfinite(strand.widths))


def test_parameters_clamped_not_exploded() -> None:
    """Out-of-range control values must clamp rather than produce garbage."""
    for grow, bloom in [(-5.0, 0.5), (0.5, -5.0), (99.0, 0.5), (0.5, 99.0)]:
        strands = build_spiderlily(grow, bloom, seed=0)
        assert all(np.all(np.isfinite(s.points)) for s in strands)


def test_ribbonize_shapes_and_uv_range() -> None:
    strand = build_spiderlily(1.0, 0.7, seed=3)[0]
    verts, uvs = ribbonize(strand)
    n = strand.points.shape[0]
    assert verts.shape == (2 * n, 3)
    assert uvs.shape == (2 * n, 2)
    assert uvs[:, 0].min() >= 0.0 and uvs[:, 0].max() <= 1.0
    assert set(np.unique(uvs[:, 1]).tolist()) == {0.0, 1.0}


def test_ribbon_does_not_pinch_when_tangent_faces_camera() -> None:
    """The cross-product fallback must keep ribbon width non-zero.

    Without the fallback, any strand segment pointing straight at the camera
    collapses to zero width and the flower appears to have holes punched in it.
    """
    strand = build_spiderlily(1.0, 1.0, seed=4)[0]
    for view in ([0, 0, 1], [0, 1, 0], [1, 0, 0]):
        verts, _ = ribbonize(strand, np.array(view, dtype=np.float32))
        widths = np.linalg.norm(verts[0::2] - verts[1::2], axis=1)
        assert widths[2:-2].min() > 1e-5, f"ribbon pinched for view {view}"


def test_build_batch_packing() -> None:
    data, counts = build_batch(build_spiderlily(1.0, 0.5, seed=0))
    assert data.shape[1] == 6  # x,y,z,u,v,role
    assert int(counts.sum()) == data.shape[0]
    assert set(np.unique(data[:, 5]).tolist()) == {0.0, 1.0}


def test_zero_growth_degenerate_case() -> None:
    """grow=0 must not crash or emit NaN -- it is the startup state every session."""
    for strand in build_spiderlily(0.0, 0.0, seed=0):
        assert np.all(np.isfinite(strand.points))
