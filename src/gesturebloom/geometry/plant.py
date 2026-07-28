"""A branching plant: an angular stem carrying a flower at each branch tip.

The single-flower version was the wrong primitive. What makes the reference
composition work is the *plant* -- a stalk that rises, forks, and presents several
blooms at different heights, so ``grow`` has somewhere to go and the silhouette
changes shape rather than merely getting bigger.

Two parameters, two distinct jobs, deliberately sequenced so they do not compete:

``grow``
    Reveals the stem by arc length, then scales the flowers in. The stem
    finishes at ``STEM_COMPLETE`` and flowers only begin appearing after that,
    which means low grow reads as "a young shoot" and high grow reads as "in
    full flower" -- two visually distinct states rather than one continuous
    scale-up.

``bloom``
    Opens the flowers, via the tepal curvature in
    :mod:`~gesturebloom.geometry.spiderlily`. Independent of grow, so you can
    hold a small plant with wide-open flowers or a tall one still in bud.

The stem is intentionally **angular** -- straight segments with kinks at the
nodes, not smooth curves. Smooth stems read as decorative vines; the kinks read
as a real plant's structural branching, and they contrast nicely against the
curved tepals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .spiderlily import SpiderlilyParams, Strand, build_spiderlily, transform_strand

#: Fraction of ``grow`` spent drawing the stem. Flowers scale in over the rest.
STEM_COMPLETE = 0.55


@dataclass
class Branch:
    """One branch: a polyline from the fork node out to a tip.

    Stored as explicit waypoints rather than generated, because the silhouette of
    the plant is a composition decision. Three hand-placed branches look
    intentional; three procedurally-spread branches look like a diagram.
    """

    waypoints: list[tuple[float, float]]
    flower_scale: float = 1.0

    @property
    def tip(self) -> tuple[float, float]:
        return self.waypoints[-1]

    def arc_length(self) -> float:
        pts = np.asarray(self.waypoints, dtype=np.float64)
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _default_branches() -> list[Branch]:
    """Three branches: low-left, tall-centre, mid-right.

    Asymmetric on purpose. A symmetric fan looks mechanical; staggered heights
    give the plant a front-to-back reading and stop the flowers from occluding
    one another.
    """
    return [
        Branch([(0.0, 0.55), (-0.42, 0.88), (-0.86, 1.12)], flower_scale=0.92),
        Branch([(0.0, 0.55), (0.02, 1.02), (0.06, 1.48)], flower_scale=1.0),
        Branch([(0.0, 0.55), (0.44, 0.92), (0.82, 1.22)], flower_scale=0.86),
    ]


@dataclass
class PlantParams:
    """Static composition of the plant."""

    stalk_base: tuple[float, float] = (0.0, -0.62)
    stalk_node: tuple[float, float] = (0.0, 0.55)
    branches: list[Branch] = field(default_factory=_default_branches)
    stem_width: float = 0.028
    stem_taper: float = 0.45
    """Stem is thickest at the base and thins toward the tips, like a real stalk."""
    flower_size: float = 0.46
    samples: int = 48
    flower_params: SpiderlilyParams | None = None

    @classmethod
    def randomized(cls, seed: int) -> PlantParams:
        """Per-plant jitter, so two plants in one scene are not identical."""
        rng = np.random.default_rng(seed)
        params = cls()
        for branch in params.branches:
            jitter = rng.normal(0.0, 0.035, size=2)
            last = branch.waypoints[-1]
            branch.waypoints[-1] = (last[0] + float(jitter[0]), last[1] + float(jitter[1]))
        params.flower_size *= 1.0 + float(rng.normal(0.0, 0.06))
        return params


def _polyline_partial(
    waypoints: list[tuple[float, float]],
    fraction: float,
    samples: int,
) -> np.ndarray:
    """Resample a 2D polyline, revealing only the first ``fraction`` of its length.

    Arc-length parameterized, so growth advances at constant speed along the path
    regardless of how the waypoints are spaced. Parameterizing by segment index
    instead would make the plant lurch through short segments and crawl through
    long ones.

    Returns
    -------
    np.ndarray
        ``(samples, 3)`` float32 points in the ``xy`` plane (``z`` = 0).
    """
    pts = np.asarray(waypoints, dtype=np.float64)
    fraction = float(np.clip(fraction, 0.0, 1.0))

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cumulative[-1])
    if total < 1e-9 or fraction < 1e-6:
        out = np.zeros((samples, 3), dtype=np.float32)
        out[:, :2] = pts[0]
        return out

    target = np.linspace(0.0, total * fraction, samples)
    x = np.interp(target, cumulative, pts[:, 0])
    y = np.interp(target, cumulative, pts[:, 1])
    out = np.zeros((samples, 3), dtype=np.float32)
    out[:, 0] = x
    out[:, 1] = y
    return out


def _stem_widths(n: int, width: float, taper: float) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)
    return (width * (1.0 - taper * t)).astype(np.float32)


def build_plant(
    grow: float,
    bloom: float,
    params: PlantParams | None = None,
    seed: int = 0,
) -> list[Strand]:
    """Build the whole plant: stem strands plus a flower per revealed branch tip.

    Parameters
    ----------
    grow, bloom:
        Both in ``[0, 1]``. See the module docstring for how they are sequenced.

    Returns
    -------
    list[Strand]
        Roles are ``"stem"``, ``"tepal"``, and ``"stamen"``. The renderer colours
        by role, so stems come out green and flowers red without the geometry
        needing to know anything about colour.

    Examples
    --------
    >>> strands = build_plant(grow=1.0, bloom=1.0)
    >>> roles = {s.role for s in strands}
    >>> sorted(roles)
    ['stamen', 'stem', 'tepal']

    At low grow the stem is partial and there are no flowers yet:

    >>> early = build_plant(grow=0.2, bloom=1.0)
    >>> {s.role for s in early}
    {'stem'}

    Flowers appear only after the stem completes:

    >>> mid = build_plant(grow=STEM_COMPLETE - 0.01, bloom=0.5)
    >>> {s.role for s in mid}
    {'stem'}
    >>> late = build_plant(grow=1.0, bloom=0.5)
    >>> sum(1 for s in late if s.role == 'tepal') > 0
    True

    Flower count scales with the number of branches:

    >>> n_flowers = len([s for s in build_plant(1.0, 1.0) if s.role == 'tepal']) // 6
    >>> n_flowers
    3
    """
    p = params or PlantParams()
    grow = float(np.clip(grow, 0.0, 1.0))
    bloom = float(np.clip(bloom, 0.0, 1.0))

    strands: list[Strand] = []

    # --- stem ------------------------------------------------------------- #
    stem_progress = float(np.clip(grow / STEM_COMPLETE, 0.0, 1.0))
    stalk_len = float(np.linalg.norm(np.subtract(p.stalk_node, p.stalk_base)))
    branch_len = max((b.arc_length() for b in p.branches), default=1.0)
    total_len = stalk_len + branch_len

    # The stalk draws first, then branches -- so the plant rises before it forks.
    stalk_share = stalk_len / max(total_len, 1e-9)
    stalk_frac = float(np.clip(stem_progress / max(stalk_share, 1e-9), 0.0, 1.0))
    branch_frac = float(
        np.clip((stem_progress - stalk_share) / max(1.0 - stalk_share, 1e-9), 0.0, 1.0)
    )

    stalk_pts = _polyline_partial([p.stalk_base, p.stalk_node], stalk_frac, p.samples)
    strands.append(
        Strand(
            points=stalk_pts,
            widths=_stem_widths(p.samples, p.stem_width, p.stem_taper),
            role="stem",
            index=0,
        )
    )

    for i, branch in enumerate(p.branches):
        if branch_frac <= 1e-6:
            continue
        pts = _polyline_partial(branch.waypoints, branch_frac, p.samples)
        strands.append(
            Strand(
                points=pts,
                widths=_stem_widths(p.samples, p.stem_width * 0.72, p.stem_taper),
                role="stem",
                index=i + 1,
            )
        )

    # --- flowers ----------------------------------------------------------- #
    # Only once the stem is complete, then scaling in over the remaining grow.
    if grow <= STEM_COMPLETE:
        return strands
    flower_t = (grow - STEM_COMPLETE) / max(1.0 - STEM_COMPLETE, 1e-9)
    # Ease-out so flowers appear decisively rather than fading in from nothing.
    flower_t = float(np.clip(flower_t, 0.0, 1.0)) ** 0.65

    flower_params = p.flower_params or SpiderlilyParams()
    for i, branch in enumerate(p.branches):
        scale = p.flower_size * branch.flower_scale * flower_t
        if scale < 1e-4:
            continue
        offset = np.array([branch.tip[0], branch.tip[1], 0.0], dtype=np.float32)
        for strand in build_spiderlily(1.0, bloom, params=flower_params, seed=seed + i * 17):
            strands.append(transform_strand(strand, scale=scale, offset=offset))

    return strands


def plant_bounds(strands: list[Strand]) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounds of all strand points. Useful for framing the camera.

    Examples
    --------
    >>> lo, hi = plant_bounds(build_plant(1.0, 1.0))
    >>> bool(lo[1] < hi[1])
    True
    """
    if not strands:
        z = np.zeros(3, dtype=np.float32)
        return z, z
    allpts = np.vstack([s.points for s in strands])
    return allpts.min(axis=0), allpts.max(axis=0)
