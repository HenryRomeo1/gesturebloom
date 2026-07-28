"""Canonicalization of MediaPipe hand landmarks.

The single largest accuracy win in this pipeline. Raw MediaPipe landmarks are
expressed in image-normalized coordinates, so an identical hand pose produces
wildly different vectors depending on where the hand sits in frame, how far it
is from the camera, and how it is rotated. A classifier trained on raw
landmarks spends most of its capacity learning to undo those nuisance factors.

We remove them analytically instead:

1. **Translation** -- re-origin at the wrist.
2. **Scale** -- divide by the wrist -> middle-MCP bone length, a rigid bone that
   does not change with pose. This is the "hand span" unit; every downstream
   distance is expressed in it, which is what makes thresholds transferable
   between users and camera distances.
3. **Rotation** -- build a right-handed orthonormal frame from the palm and
   project into it. After this, middle-MCP always sits at (0, 1, 0) and the
   palm normal always points along +z, regardless of hand orientation.
4. **Chirality** -- mirror left hands into right-hand form. This doubles the
   effective dataset size and lets one model serve both hands.

The transform is exactly invertible given the stored :class:`HandFrame`, which
matters for rendering: geometry is authored in canonical space and pushed back
into image space for compositing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- MediaPipe HandLandmarker index constants -------------------------------
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

N_LANDMARKS = 21
FEATURE_DIM = N_LANDMARKS * 3

FINGERTIPS = (THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP)
FINGER_MCPS = (THUMB_MCP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP)

#: (mcp, pip, dip, tip) chains for the four non-thumb fingers.
FINGER_CHAINS = (
    (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
)

_EPS = 1e-8


@dataclass(frozen=True)
class HandFrame:
    """The rigid transform that was removed during canonicalization.

    Attributes
    ----------
    origin:
        Wrist position in the source coordinate system, shape ``(3,)``.
    scale:
        Hand-span scalar (wrist -> middle-MCP distance) in source units.
    basis:
        ``(3, 3)`` row-stacked orthonormal basis ``[x, y, z]`` expressed in
        source coordinates. Maps source -> canonical as ``basis @ v``.
    mirrored:
        Whether an x-flip was applied to normalize chirality.
    """

    origin: np.ndarray
    scale: float
    basis: np.ndarray
    mirrored: bool

    def to_source(self, pts: np.ndarray) -> np.ndarray:
        """Push canonical-space points back into source coordinates."""
        pts = np.atleast_2d(np.asarray(pts, dtype=np.float64))
        out = (pts @ self.basis) * self.scale + self.origin
        if self.mirrored:
            out = out.copy()
            out[:, 0] *= -1.0
        return out


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < _EPS:
        raise ValueError("degenerate vector; cannot normalize")
    return v / n


def canonicalize(
    landmarks: np.ndarray,
    handedness: str = "Right",
) -> tuple[np.ndarray, HandFrame]:
    """Map raw landmarks into the canonical hand frame.

    Parameters
    ----------
    landmarks:
        ``(21, 3)`` array of landmark positions in any consistent coordinate
        system (image-normalized or metric world both work).
    handedness:
        ``"Right"`` or ``"Left"``. Left hands are mirrored into right-hand form.

    Returns
    -------
    canonical:
        ``(21, 3)`` array. Wrist at origin, middle-MCP at ``(0, 1, 0)``, palm
        normal along ``+z``.
    frame:
        The :class:`HandFrame` that was factored out.

    Raises
    ------
    ValueError
        If the pose is geometrically degenerate (collapsed hand span, or a palm
        so edge-on that the frame cannot be resolved). Callers should treat this
        as a dropped frame rather than a fatal error -- see
        :func:`try_canonicalize`.
    """
    lm = np.asarray(landmarks, dtype=np.float64)
    if lm.shape != (N_LANDMARKS, 3):
        raise ValueError(f"expected (21, 3) landmarks, got {lm.shape}")
    if not np.all(np.isfinite(lm)):
        raise ValueError("landmarks contain non-finite values")

    mirrored = handedness.lower().startswith("l")
    if mirrored:
        lm = lm.copy()
        lm[:, 0] *= -1.0

    origin = lm[WRIST].copy()
    p = lm - origin

    scale = float(np.linalg.norm(p[MIDDLE_MCP]))
    if scale < 1e-6:
        raise ValueError("degenerate hand span")
    p = p / scale

    # y: along the palm, wrist -> middle MCP. Rigid, pose-independent.
    y = _unit(p[MIDDLE_MCP])
    # x: across the palm, pinky -> index. Gram-Schmidt against y.
    across = p[INDEX_MCP] - p[PINKY_MCP]
    x_raw = across - np.dot(across, y) * y
    if float(np.linalg.norm(x_raw)) < 1e-4:
        raise ValueError("palm too edge-on to resolve frame")
    x = _unit(x_raw)
    z = np.cross(x, y)  # palm normal, completes a right-handed frame

    basis = np.stack([x, y, z], axis=0)
    canonical = p @ basis.T
    return canonical, HandFrame(origin=origin, scale=scale, basis=basis, mirrored=mirrored)


def try_canonicalize(
    landmarks: np.ndarray, handedness: str = "Right"
) -> tuple[np.ndarray, HandFrame] | None:
    """Non-raising variant. Returns ``None`` on degenerate input."""
    try:
        return canonicalize(landmarks, handedness)
    except ValueError:
        return None


def feature_vector(canonical: np.ndarray, drop_wrist: bool = True) -> np.ndarray:
    """Flatten canonical landmarks into a model input vector.

    The wrist is identically zero after canonicalization and the middle-MCP is
    identically ``(0, 1, 0)``, so both carry no information. Dropping the wrist
    yields 60 dims; we keep middle-MCP for index alignment simplicity.
    """
    c = np.asarray(canonical, dtype=np.float32)
    if drop_wrist:
        c = c[1:]
    return c.reshape(-1)


def augment(
    canonical: np.ndarray,
    rng: np.random.Generator,
    rot_sigma_deg: float = 8.0,
    scale_sigma: float = 0.06,
    jitter_sigma: float = 0.012,
) -> np.ndarray:
    """Train-time augmentation applied *in canonical space*.

    Canonicalization removes global pose, so augmentation here simulates
    landmark estimation error and residual frame instability rather than
    re-introducing the nuisance factors we just removed. Keep the rotation
    sigma small: large rotations would teach the model to accept poses the
    canonicalizer can never actually emit.
    """
    c = np.asarray(canonical, dtype=np.float64).copy()

    ax = rng.normal(size=3)
    ax /= max(float(np.linalg.norm(ax)), _EPS)
    ang = np.deg2rad(rng.normal(0.0, rot_sigma_deg))
    K = np.array(
        [[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]], dtype=np.float64
    )
    R = np.eye(3) + np.sin(ang) * K + (1.0 - np.cos(ang)) * (K @ K)  # Rodrigues
    c = c @ R.T

    c *= 1.0 + rng.normal(0.0, scale_sigma)
    c += rng.normal(0.0, jitter_sigma, size=c.shape)
    return c.astype(np.float32)
