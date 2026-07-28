"""Landmark sources: live webcam via MediaPipe, or replay from a recording.

Both satisfy the same iterator protocol -- ``(landmarks | None, handedness, dt)``
-- which is what makes ``--replay`` a one-flag substitution everywhere. Any code
that branches on which source it has is a bug.

MediaPipe and OpenCV are optional dependencies: ``pip install 'gesturebloom[live]'``.
The replay source needs only numpy, so CI and tests never touch them.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ..data.recording import Recording, replay

Frame = tuple[np.ndarray | None, str, float]


class LandmarkSource(Protocol):
    """Anything that can yield timestamped hand landmarks."""

    def frames(self) -> Iterator[Frame]: ...
    def close(self) -> None: ...
    @property
    def nominal_fps(self) -> float: ...


@dataclass
class ReplaySource:
    """Deterministic playback of a ``.npz`` recording."""

    recording: Recording
    realtime: bool = False
    loop: bool = False

    @classmethod
    def from_path(cls, path: str | Path, realtime: bool = False, loop: bool = False) -> ReplaySource:
        return cls(recording=Recording.load(path), realtime=realtime, loop=loop)

    def frames(self) -> Iterator[Frame]:
        while True:
            yield from replay(self.recording, realtime=self.realtime)
            if not self.loop:
                return

    def close(self) -> None:
        return None

    @property
    def nominal_fps(self) -> float:
        return self.recording.fps


class WebcamSource:
    """Live capture through MediaPipe HandLandmarker.

    Two configuration notes that matter more than they look:

    ``model_complexity`` / ``num_hands``
        Tracking two hands roughly doubles landmark cost, which is the dominant
        stage in the latency budget. Default to one hand and only raise it when
        the interaction actually needs both.

    ``min_tracking_confidence``
        Raising it produces cleaner landmarks but more dropped frames. Because
        :class:`~gesturebloom.landmarks.filters.LandmarkSmoother` holds the last
        good pose across short dropouts, you can afford a higher threshold here
        than you would without it -- prefer clean-or-absent over noisy-and-present.
    """

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 1280,
        height: int = 720,
        target_fps: int = 60,
        num_hands: int = 1,
        min_detection_confidence: float = 0.6,
        min_tracking_confidence: float = 0.6,
        mirror: bool = True,
    ) -> None:
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "opencv-python and mediapipe are required for live capture. "
                "Install with: pip install 'gesturebloom[live]'"
            ) from exc

        self._cv2 = cv2
        self.mirror = mirror
        self._cap = cv2.VideoCapture(camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, target_fps)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open camera index {camera_index}")

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._target_fps = float(target_fps)

    def frames(self) -> Iterator[Frame]:
        cv2 = self._cv2
        last = time.perf_counter()
        while True:
            ok, bgr = self._cap.read()
            if not ok:
                return
            if self.mirror:
                bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = self._hands.process(rgb)

            now = time.perf_counter()
            dt = now - last
            last = now

            if not result.multi_hand_landmarks:
                yield None, "Right", dt
                continue

            hand_lms = result.multi_hand_landmarks[0]
            label = "Right"
            if result.multi_handedness:
                label = result.multi_handedness[0].classification[0].label
                # MediaPipe labels handedness for the *un-mirrored* image, so a
                # mirrored preview reverses it. Getting this wrong silently halves
                # your effective dataset by mislabeling chirality.
                if self.mirror:
                    label = "Left" if label == "Right" else "Right"

            lm = np.array([[p.x, p.y, p.z] for p in hand_lms.landmark], dtype=np.float64)
            yield lm, label, dt

    def close(self) -> None:
        self._cap.release()
        self._hands.close()

    @property
    def nominal_fps(self) -> float:
        return self._target_fps


def open_source(
    replay_path: str | Path | None = None,
    camera_index: int = 0,
    realtime: bool = False,
    loop: bool = False,
    **webcam_kwargs,
) -> LandmarkSource:
    """Factory: replay if a path is given, otherwise live webcam.

    Every CLI command routes through here, so ``--replay`` works uniformly.
    """
    if replay_path is not None:
        return ReplaySource.from_path(replay_path, realtime=realtime, loop=loop)
    return WebcamSource(camera_index=camera_index, **webcam_kwargs)
