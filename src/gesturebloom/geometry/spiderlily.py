"""Parametric *Lycoris radiata* (red spider lily) geometry.

Why this flower: it is unusually well suited to parametric generation. Six
narrow, strongly recurved tepals and six long protruding stamens, with almost no
bulk -- the whole form is line-like, so it renders beautifully as ribbons and
needs no mesh assets.

Two ideas make the animation feel organic rather than tweened:

**Growth is an arc-length parameter, not a scale factor.**
Scaling a finished tepal up from zero looks like a zoom. Real growth extends the
tip while the base stays put. So each tepal is built by integrating a unit
tangent along arc length ``s``, and ``grow`` sets the upper limit of
integration. The base is therefore *identical* at every growth stage -- the
curve is revealed, not inflated.

**Bloom is curvature, not rotation.**
Tepals recurve by bending along their length, so bloom drives the curvature
``kappa`` of the tangent's elevation angle. Rotating a straight tepal outward
looks like a hinge; bending it looks like a flower. Because we integrate a unit
tangent, tepal length is exactly preserved under any curvature -- bending cannot
stretch it.

Coordinate convention: ``+y`` is up along the stem, ``xz`` is the horizontal
plane, and the flower's origin is the receptacle where all tepals meet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TAU = 2.0 * np.pi


@dataclass
class SpiderlilyParams:
    """Static form parameters. Not animated -- these define *this* flower.

    Randomize per instance from a seed so a bouquet of several flowers does not
    look cloned.
    """

    n_tepals: int = 6
    n_stamens: int = 6
    samples: int = 72

    tepal_length: float = 1.0
    tepal_psi0: float = 0.30
    """Initial elevation of the tepal tangent, radians from ``+y``. Small means
    tepals leave the receptacle pointing nearly straight up."""
    tepal_kappa_closed: float = 0.35
    tepal_kappa_open: float = 2.75
    """Curvature per unit arc length at bloom 0 and 1. At 2.75 the tangent turns
    ~157 degrees over the tepal, carrying the tip back down past horizontal --
    the characteristic recurve."""
    tepal_width: float = 0.045
    ripple_amp: float = 0.055
    ripple_freq: float = 2.5
    """Azimuthal waviness -- the crinkle along a spiderlily tepal edge."""

    stamen_length: float = 1.55
    stamen_psi0: float = 0.16
    stamen_kappa_closed: float = 0.20
    stamen_kappa_open: float = 0.85
    stamen_width: float = 0.010
    stamen_extend: float = 0.45
    """Fraction of stamen length held back at bloom 0. Stamens emerge *after*
    the tepals begin to open, which is what makes the bloom read as a sequence
    rather than a single blended transition."""

    azimuth_jitter: float = 0.10
    length_jitter: float = 0.09
    phase_jitter: float = 0.35

    @classmethod
    def randomized(cls, seed: int, **overrides) -> SpiderlilyParams:
        """Per-flower variation from a seed."""
        rng = np.random.default_rng(seed)
        base = cls(**overrides)
        base.tepal_length *= 1.0 + rng.normal(0.0, base.length_jitter)
        base.stamen_length *= 1.0 + rng.normal(0.0, base.length_jitter)
        base.tepal_kappa_open *= 1.0 + rng.normal(0.0, 0.07)
        base.ripple_freq *= 1.0 + rng.normal(0.0, 0.12)
        return base


def _integrate_strand(
    azimuth: float,
    length: float,
    kappa: float,
    psi0: float,
    grow: float,
    samples: int,
    ripple_amp: float = 0.0,
    ripple_freq: float = 0.0,
    ripple_phase: float = 0.0,
) -> np.ndarray:
    """Integrate a unit tangent along arc length to produce one strand.

    The tangent's elevation from ``+y`` is ``psi(s) = psi0 + kappa * s``, so the
    strand curves at a constant rate -- a clothoid-like arc in the vertical plane
    at ``azimuth``, with an optional azimuthal ripple superimposed.

    Returns
    -------
    np.ndarray
        ``(samples, 3)`` float32 polyline starting at the origin. Total length is
        exactly ``length * grow`` regardless of ``kappa``.
    """
    grow = float(np.clip(grow, 0.0, 1.0))
    s_max = max(length * grow, 1e-6)
    s = np.linspace(0.0, s_max, samples)
    ds = s_max / max(samples - 1, 1)

    psi = psi0 + kappa * s
    phi = azimuth
    if ripple_amp > 0.0 and length > 1e-9:
        phi = azimuth + ripple_amp * np.sin(TAU * ripple_freq * s / length + ripple_phase)

    sin_psi, cos_psi = np.sin(psi), np.cos(psi)
    tangent = np.stack([sin_psi * np.cos(phi), cos_psi, sin_psi * np.sin(phi)], axis=1)

    pts = np.cumsum(tangent * ds, axis=0)
    pts -= pts[0]  # anchor the base at the receptacle
    return pts.astype(np.float32)


def _taper(n: int, width: float, power: float = 0.55) -> np.ndarray:
    """Width profile: swells just past the base, tapers to a point at the tip."""
    t = np.linspace(0.0, 1.0, n)
    return (width * np.power(np.sin(np.pi * np.clip(t, 0.0, 1.0)) + 1e-6, power)).astype(np.float32)


#: Default view direction for ribbonization (camera looking down -z).
DEFAULT_VIEW_DIR = np.array([0.0, 0.0, 1.0], dtype=np.float32)


@dataclass
class Strand:
    """One tepal or stamen: a polyline with a per-vertex width and role tag."""

    points: np.ndarray  # (N, 3) float32
    widths: np.ndarray  # (N,)   float32
    role: str  # "tepal" | "stamen"
    index: int


def build_spiderlily(
    grow: float,
    bloom: float,
    params: SpiderlilyParams | None = None,
    seed: int = 0,
) -> list[Strand]:
    """Generate all strands for one flower at a given animation state.

    Parameters
    ----------
    grow:
        ``[0, 1]``. Arc-length fraction of each strand that exists.
    bloom:
        ``[0, 1]``. Recurve amount, and stamen protrusion.
    params:
        Static form parameters; randomized defaults if omitted.
    seed:
        Drives per-strand azimuth and ripple-phase jitter so the six tepals are
        not rotationally identical.

    Returns
    -------
    list[Strand]

    Examples
    --------
    >>> strands = build_spiderlily(grow=1.0, bloom=1.0)
    >>> len(strands)
    12
    >>> strands[0].points.shape
    (72, 3)

    Arc length is preserved under bloom -- bending does not stretch:

    >>> import numpy as np
    >>> def arclen(s):
    ...     return float(np.sum(np.linalg.norm(np.diff(s.points, axis=0), axis=1)))
    >>> a = arclen(build_spiderlily(1.0, 0.0, seed=3)[0])
    >>> b = arclen(build_spiderlily(1.0, 1.0, seed=3)[0])
    >>> abs(a - b) < 1e-3
    True

    Growth reveals rather than scales -- the base is unchanged:

    >>> half = build_spiderlily(0.5, 0.6, seed=1)[0].points
    >>> full = build_spiderlily(1.0, 0.6, seed=1)[0].points
    >>> bool(np.allclose(half[0], full[0], atol=1e-6))
    True
    """
    p = params or SpiderlilyParams.randomized(seed)
    rng = np.random.default_rng(seed)
    grow = float(np.clip(grow, 0.0, 1.0))
    bloom = float(np.clip(bloom, 0.0, 1.0))

    strands: list[Strand] = []

    tepal_kappa = p.tepal_kappa_closed + bloom * (p.tepal_kappa_open - p.tepal_kappa_closed)
    for i in range(p.n_tepals):
        az = TAU * i / p.n_tepals + rng.normal(0.0, p.azimuth_jitter)
        pts = _integrate_strand(
            azimuth=az,
            length=p.tepal_length,
            kappa=tepal_kappa,
            psi0=p.tepal_psi0,
            grow=grow,
            samples=p.samples,
            ripple_amp=p.ripple_amp,
            ripple_freq=p.ripple_freq,
            ripple_phase=rng.uniform(0.0, TAU) * p.phase_jitter,
        )
        strands.append(
            Strand(points=pts, widths=_taper(p.samples, p.tepal_width), role="tepal", index=i)
        )

    # Stamens lag the tepals: they only extend once bloom passes the hold-back
    # fraction, so the flower opens then pushes its stamens out.
    stamen_grow = grow * (1.0 - p.stamen_extend + p.stamen_extend * bloom)
    stamen_kappa = p.stamen_kappa_closed + bloom * (p.stamen_kappa_open - p.stamen_kappa_closed)
    for i in range(p.n_stamens):
        az = TAU * (i + 0.5) / p.n_stamens + rng.normal(0.0, p.azimuth_jitter * 1.5)
        pts = _integrate_strand(
            azimuth=az,
            length=p.stamen_length,
            kappa=stamen_kappa,
            psi0=p.stamen_psi0,
            grow=stamen_grow,
            samples=p.samples,
            ripple_amp=p.ripple_amp * 0.3,
            ripple_freq=p.ripple_freq * 0.6,
            ripple_phase=rng.uniform(0.0, TAU) * p.phase_jitter,
        )
        strands.append(
            Strand(points=pts, widths=_taper(p.samples, p.stamen_width, power=0.3), role="stamen", index=i)
        )

    return strands


def ribbonize(
    strand: Strand,
    view_dir: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Expand a strand polyline into a camera-facing triangle-strip ribbon.

    Done on the CPU because the vertex count is trivial (72 samples x 12 strands
    x 2 = ~1.7k vertices) and it keeps the shader simple enough to read. If you
    later push this to thousands of flowers, move it to a geometry or compute
    shader -- the math is unchanged.

    Returns
    -------
    vertices:
        ``(2N, 3)`` float32, interleaved left/right pairs for a triangle strip.
    uvs:
        ``(2N, 2)`` float32. ``u`` is normalized arc length along the strand
        (drives the tip-to-base color gradient), ``v`` is ``0``/``1`` across the
        ribbon width (drives the soft edge falloff).
    """
    pts = strand.points
    n = pts.shape[0]
    if n < 2:
        empty = np.zeros((0, 3), dtype=np.float32)
        return empty, np.zeros((0, 2), dtype=np.float32)

    tangents = np.gradient(pts, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangents = tangents / np.maximum(norms, 1e-8)

    v = np.asarray(DEFAULT_VIEW_DIR if view_dir is None else view_dir, dtype=np.float32)
    v = v / max(float(np.linalg.norm(v)), 1e-8)
    side = np.cross(tangents, v)
    side_norm = np.linalg.norm(side, axis=1, keepdims=True)
    # Where the tangent is parallel to the view direction the cross product
    # collapses; fall back to a fixed axis so the ribbon does not pinch shut.
    fallback = np.cross(tangents, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    fb_norm = np.linalg.norm(fallback, axis=1, keepdims=True)
    side = np.where(side_norm < 1e-4, fallback / np.maximum(fb_norm, 1e-8), side / np.maximum(side_norm, 1e-8))

    half = strand.widths[:, None] * 0.5
    left = pts - side * half
    right = pts + side * half

    vertices = np.empty((2 * n, 3), dtype=np.float32)
    vertices[0::2] = left
    vertices[1::2] = right

    u = np.linspace(0.0, 1.0, n, dtype=np.float32)
    uvs = np.empty((2 * n, 2), dtype=np.float32)
    uvs[0::2, 0] = u
    uvs[1::2, 0] = u
    uvs[0::2, 1] = 0.0
    uvs[1::2, 1] = 1.0
    return vertices, uvs
