"""Hand skeleton and HUD overlays, drawn onto the camera frame with OpenCV.

Why OpenCV and not GLSL: the camera frame is already a numpy array passing
through this process, and OpenCV's line/circle/text primitives are hardware-fast
and take five minutes to get right. Drawing 2D annotations in GL would mean a
text-rendering stack, a font atlas, and a second shader pipeline for what is
fundamentally a debug HUD. Wrong place to spend complexity.

The division of labour ends up clean:

- **OpenCV** draws everything in *image space* (skeleton, labels, tracking state).
  Cheap, legible, easy to change.
- **GLSL** draws the flower in *scene space*, with the bloom pass. Expensive,
  pretty, worth the shader.

The composite is then just: camera+annotations as an opaque background, flower
additively on top. The flower glows; the camera feed does not, which is what
keeps it readable.
"""

from __future__ import annotations

import numpy as np

from ..landmarks.canonical import (
    INDEX_TIP,
    THUMB_TIP,
    WRIST,
)

#: MediaPipe's 21-landmark skeleton, as index pairs.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    # thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # middle
    (9, 10), (10, 11), (11, 12),
    # ring
    (13, 14), (14, 15), (15, 16),
    # pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # palm bridge -- without these the fingers look detached from the hand
    (5, 9), (9, 13), (13, 17),
)

FINGERTIPS = (4, 8, 12, 16, 20)

# BGR, because OpenCV. Mint green reads well over most skin tones and
# backgrounds, and stays distinct from the flower's red.
SKELETON_COLOR = (150, 255, 120)
JOINT_COLOR = (255, 255, 255)
FINGERTIP_COLOR = (120, 255, 220)
LABEL_COLOR = (255, 255, 255)
LABEL_SHADOW = (0, 0, 0)


def draw_skeleton(
    frame: np.ndarray,
    landmarks: np.ndarray,
    thickness: int = 2,
    joint_radius: int = 3,
) -> np.ndarray:
    """Draw the hand skeleton onto an RGB frame, in place.

    Parameters
    ----------
    frame:
        ``(H, W, 3)`` uint8 RGB image. Modified in place and returned.
    landmarks:
        ``(21, 3)`` landmarks in **image-normalized** coordinates -- i.e. the raw
        MediaPipe output, *not* canonicalized. Canonical coordinates have had
        position and scale deliberately removed, so they cannot be drawn back
        onto the image; this function needs the pre-canonical values.

    Notes
    -----
    Depth (``z``) modulates line thickness slightly, which gives a cheap sense of
    the hand tilting toward and away from the camera.
    """
    import cv2

    h, w = frame.shape[:2]
    pts = np.empty((21, 2), dtype=np.int32)
    pts[:, 0] = np.clip(landmarks[:, 0] * w, -1e4, 1e4).astype(np.int32)
    pts[:, 1] = np.clip(landmarks[:, 1] * h, -1e4, 1e4).astype(np.int32)

    z = landmarks[:, 2]
    z_span = float(np.ptp(z)) or 1.0

    for a, b in HAND_CONNECTIONS:
        depth = (z[a] + z[b]) * 0.5
        near = 1.0 - float(np.clip((depth - z.min()) / z_span, 0.0, 1.0))
        t = max(1, round(thickness * (0.6 + 0.8 * near)))
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), SKELETON_COLOR, t, cv2.LINE_AA)

    for i in range(21):
        if i in FINGERTIPS:
            cv2.circle(frame, tuple(pts[i]), joint_radius + 2, FINGERTIP_COLOR, -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, tuple(pts[i]), joint_radius, JOINT_COLOR, -1, cv2.LINE_AA)

    return frame


def _label(frame: np.ndarray, text: str, xy: tuple[int, int], scale: float = 0.55) -> None:
    """Text with a dark outline, so it stays legible over any background."""
    import cv2

    x, y = xy
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, (x, y), font, scale, LABEL_SHADOW, 3, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, LABEL_COLOR, 1, cv2.LINE_AA)


def draw_param_labels(
    frame: np.ndarray,
    landmarks: np.ndarray,
    params: dict[str, float],
) -> np.ndarray:
    """Float the live control values next to the fingers that drive them.

    This is the single most useful debugging affordance in the project. Seeing
    ``bloom`` pinned at 0.00 while you pinch tells you instantly whether the
    problem is tracking, calibration, or mapping -- something a number printed in
    the terminal cannot do, because you cannot watch your hand and the terminal
    at the same time.
    """
    h, w = frame.shape[:2]

    def at(idx: int, dx: int = 14, dy: int = -8) -> tuple[int, int]:
        return (
            int(np.clip(landmarks[idx, 0] * w + dx, 4, w - 130)),
            int(np.clip(landmarks[idx, 1] * h + dy, 18, h - 8)),
        )

    # grow is driven by openness (whole hand), bloom by pinch aperture --
    # so anchor each label to the anatomy responsible for it.
    if "grow" in params:
        _label(frame, f"grow {params['grow']:.2f}", at(INDEX_TIP))
    if "bloom" in params:
        _label(frame, f"bloom {params['bloom']:.2f}", at(THUMB_TIP, dx=14, dy=18))
    return frame


def draw_hud(
    frame: np.ndarray,
    params: dict[str, float],
    fps: float | None = None,
    tracking: bool = True,
    backend: str = "",
) -> np.ndarray:
    """Corner HUD: bars for each parameter, plus fps and tracking state.

    Bars rather than only numbers, because a bar shows you the *range* you are
    actually reaching. If your bloom bar never gets past a third, you need
    calibration -- which is invisible from the number alone.
    """
    import cv2

    h = frame.shape[0]
    x0, y0 = 14, 26
    bar_w, bar_h, gap = 150, 10, 24

    if not tracking:
        _label(frame, "no hand detected", (x0, y0), scale=0.6)
        return frame

    for i, (name, value) in enumerate(sorted(params.items())):
        y = y0 + i * gap
        _label(frame, f"{name:<6} {value:.2f}", (x0, y), scale=0.5)
        bx = x0 + 116
        cv2.rectangle(frame, (bx, y - 9), (bx + bar_w, y - 9 + bar_h), (60, 60, 60), -1)
        fill = int(bar_w * float(np.clip(value, 0.0, 1.0)))
        if fill > 0:
            cv2.rectangle(frame, (bx, y - 9), (bx + fill, y - 9 + bar_h), (120, 255, 220), -1)

    footer = []
    if fps is not None:
        footer.append(f"{fps:5.1f} fps")
    if backend:
        footer.append(backend)
    if footer:
        _label(frame, "  ".join(footer), (x0, h - 14), scale=0.45)

    return frame


def wrist_anchor_ndc(landmarks: np.ndarray, y_offset: float = 0.0) -> tuple[float, float]:
    """Convert the wrist position to normalized device coordinates for the flower.

    Image space is ``x`` right, ``y`` down, both in ``[0, 1]``. NDC is ``x`` right,
    ``y`` **up**, both in ``[-1, 1]`` -- so the ``y`` axis flips. Forgetting that
    flip is the classic bug here, and it presents as a flower that moves the
    wrong way vertically, which is confusing rather than obviously broken.

    Examples
    --------
    >>> import numpy as np
    >>> lm = np.zeros((21, 3)); lm[0] = [0.5, 0.5, 0.0]
    >>> wrist_anchor_ndc(lm)
    (0.0, 0.0)
    >>> lm[0] = [1.0, 0.0, 0.0]
    >>> wrist_anchor_ndc(lm)
    (1.0, 1.0)
    """
    x = float(landmarks[WRIST, 0]) * 2.0 - 1.0
    y = (1.0 - float(landmarks[WRIST, 1])) * 2.0 - 1.0
    return (x, y + y_offset)
