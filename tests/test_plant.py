"""Tests for the branching plant.

The properties asserted here are the ones the composition depends on: grow must
sequence the stem before the flowers, flower count must follow branch count, and
nothing may emit NaN anywhere in the parameter space -- a single NaN vertex takes
out an entire triangle strip on the GPU, and it shows up as a flickering wedge
that is very hard to trace back to its source.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from gesturebloom.geometry.plant import (
    STEM_COMPLETE,
    PlantParams,
    build_plant,
    plant_bounds,
)
from gesturebloom.geometry.spiderlily import build_spiderlily, transform_strand
from gesturebloom.render.window import build_batch


def roles(strands) -> set[str]:
    return {s.role for s in strands}


def count_role(strands, role: str) -> int:
    return sum(1 for s in strands if s.role == role)


def test_grow_sequences_stem_then_flowers() -> None:
    """Low grow is a shoot; high grow is in flower. Two distinct readings."""
    assert roles(build_plant(0.1, 1.0)) == {"stem"}
    assert roles(build_plant(STEM_COMPLETE - 0.01, 1.0)) == {"stem"}
    assert "tepal" in roles(build_plant(1.0, 1.0))


def test_flower_count_matches_branch_count() -> None:
    params = PlantParams()
    strands = build_plant(1.0, 1.0, params=params)
    # 6 tepals + 6 stamens per flower
    assert count_role(strands, "tepal") == 6 * len(params.branches)
    assert count_role(strands, "stamen") == 6 * len(params.branches)


def test_stem_strand_count() -> None:
    """One stalk plus one strand per branch, once branches have started."""
    params = PlantParams()
    assert count_role(build_plant(1.0, 0.5, params=params), "stem") == 1 + len(params.branches)


def test_stem_grows_monotonically() -> None:
    """Revealed stem length must never decrease as grow increases."""
    lengths = []
    for grow in np.linspace(0.02, STEM_COMPLETE, 12):
        stems = [s for s in build_plant(float(grow), 0.5) if s.role == "stem"]
        total = sum(
            float(np.sum(np.linalg.norm(np.diff(s.points, axis=0), axis=1))) for s in stems
        )
        lengths.append(total)
    assert all(b >= a - 1e-4 for a, b in pairwise(lengths))


def test_flowers_scale_up_with_grow() -> None:
    """Above the stem threshold, more grow means visibly bigger flowers."""
    sizes = []
    for grow in (0.65, 0.8, 1.0):
        tepals = [s for s in build_plant(grow, 1.0, seed=1) if s.role == "tepal"]
        pts = np.vstack([s.points for s in tepals])
        sizes.append(float(np.ptp(pts, axis=0).max()))
    assert sizes[0] < sizes[1] < sizes[2]


def test_bloom_does_not_change_flower_count_or_stem() -> None:
    """bloom and grow must be independent axes."""
    a = build_plant(1.0, 0.0)
    b = build_plant(1.0, 1.0)
    assert count_role(a, "tepal") == count_role(b, "tepal")
    assert count_role(a, "stem") == count_role(b, "stem")


def test_bloom_widens_the_silhouette() -> None:
    """Opening the flowers must actually spread them."""
    def width(bloom: float) -> float:
        pts = np.vstack([s.points for s in build_plant(1.0, bloom, seed=2) if s.role == "tepal"])
        return float(np.ptp(pts[:, 0]))

    assert width(1.0) > width(0.0)


def test_all_finite_across_parameter_sweep() -> None:
    for grow in np.linspace(0.0, 1.0, 11):
        for bloom in np.linspace(0.0, 1.0, 11):
            for s in build_plant(float(grow), float(bloom), seed=3):
                assert np.all(np.isfinite(s.points)), f"NaN at grow={grow} bloom={bloom}"
                assert np.all(np.isfinite(s.widths))


@pytest.mark.parametrize("grow,bloom", [(-9.0, 0.5), (0.5, -9.0), (99.0, 0.5), (0.5, 99.0)])
def test_out_of_range_clamps(grow: float, bloom: float) -> None:
    for s in build_plant(grow, bloom):
        assert np.all(np.isfinite(s.points))


def test_zero_grow_is_safe() -> None:
    """The startup state of every session."""
    strands = build_plant(0.0, 0.0)
    assert strands
    for s in strands:
        assert np.all(np.isfinite(s.points))


def test_transform_strand_scales_points_and_widths() -> None:
    s = build_spiderlily(1.0, 1.0, seed=0)[0]
    offset = np.array([2.0, -1.0, 0.5], dtype=np.float32)
    t = transform_strand(s, scale=0.25, offset=offset)
    np.testing.assert_allclose(t.points, s.points * 0.25 + offset, atol=1e-6)
    np.testing.assert_allclose(t.widths, s.widths * 0.25, atol=1e-6)
    assert t.role == s.role


def test_flowers_sit_at_branch_tips() -> None:
    """Each flower must be positioned at its branch tip, not at the origin."""
    params = PlantParams()
    strands = build_plant(1.0, 0.3, params=params)
    tepals = [s for s in strands if s.role == "tepal"]
    bases = np.array([s.points[0] for s in tepals])
    for branch in params.branches:
        tip = np.array(branch.tip)
        assert np.linalg.norm(bases[:, :2] - tip, axis=1).min() < 0.15


def test_batch_packing_and_colours() -> None:
    data, counts = build_batch(build_plant(1.0, 0.8))
    assert data.shape[1] == 8  # x,y,z,u,v,r,g,b
    assert int(counts.sum()) == data.shape[0]
    assert np.all(np.isfinite(data))
    # Three roles -> three distinct colours in the buffer.
    assert len(np.unique(data[:, 5:8], axis=0)) == 3


def test_plant_bounds() -> None:
    lo, hi = plant_bounds(build_plant(1.0, 1.0))
    assert lo[1] < hi[1]
    assert plant_bounds([])[0].shape == (3,)
