"""Landmark sources: live webcam via MediaPipe, or replay from a recording.

Both satisfy the same iterator protocol -- ``(landmarks | None, handedness, dt)``
-- which is what makes ``--replay`` a one-flag substitution everywhere. Any code
that branches on which source it has is a bug.

MediaPipe API versions
----------------------
MediaPipe changed its Python interface in a breaking way. The legacy
``mediapipe.solutions.hands.Hands`` interface was removed in MediaPipe 1.0 and
replaced by the Tasks API (``mediapipe.tasks.python.vision.HandLandmarker``).

This module supports both, preferring Tasks when available, because pinning to
one means the repo breaks for anyone on the other side of that version boundary.
The two backends are isolated behind :class:`_HandBackend` so the capture loop
never knows which one it has.

The Tasks API needs a model file on disk (the legacy API bundled its weights).
:func:`ensure_hand_model` downloads and caches it to
``~/.cache/gesturebloom/hand_landmarker.task`` on first use.

MediaPipe and OpenCV are optional dependencies: ``pip install 'gesturebloom[live]'``.
The replay source needs only numpy, so CI and tests never touch them.
"""

from __future__ import annotations

import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ..data.recording import Recording, replay

Frame = tuple[np.ndarray | None, str, float]

#: Google's published float16 hand landmarker bundle for the Tasks API.
HAND_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def model_cache_path() -> Path:
    """Where the Tasks model bundle is cached.

    Respects ``XDG_CACHE_HOME``; falls back to ``~/.cache``. Override entirely
    with the ``GESTUREBLOOM_HAND_MODEL`` environment variable, which is also the
    escape hatch for air-gapped machines.
    """
    override = os.environ.get("GESTUREBLOOM_HAND_MODEL")
    if override:
        return Path(override).expanduser()
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "gesturebloom" / "hand_landmarker.task"


def ensure_hand_model(path: Path | None = None, url: str = HAND_LANDMARKER_URL) -> Path:
    """Return a local path to the hand landmarker bundle, downloading if needed.

    Raises
    ------
    RuntimeError
        With manual-download instructions if the fetch fails. A network error
        here is common (corporate proxies, offline machines) and deserves an
        actionable message rather than a bare traceback.
    """
    target = Path(path) if path is not None else model_cache_path()
    if target.exists() and target.stat().st_size > 1000:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading hand landmarker model (~7 MB) to {target} ...")
    tmp = target.with_suffix(".partial")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as fh:
            while chunk := response.read(1 << 16):
                fh.write(chunk)
        tmp.replace(target)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the hand landmarker model: {exc}\n\n"
            f"Download it manually:\n"
            f"  1. Open this URL in a browser:\n     {url}\n"
            f"  2. Save the file to:\n     {target}\n\n"
            f"Or point at an existing copy:\n"
            f"  export GESTUREBLOOM_HAND_MODEL=/path/to/hand_landmarker.task"
        ) from exc

    print(f"Saved {target} ({target.stat().st_size / 1e6:.1f} MB)")
    return target


class LandmarkSource(Protocol):
    """Anything that can yield timestamped hand landmarks."""

    def frames(self) -> Iterator[Frame]: ...
    def close(self) -> None: ...
    @property
    def last_frame(self) -> np.ndarray | None: ...
    @property
    def nominal_fps(self) -> float: ...


@dataclass
class ReplaySource:
    """Deterministic playback of a ``.npz`` recording."""

    recording: Recording
    realtime: bool = False
    loop: bool = False

    @classmethod
    def from_path(
        cls, path: str | Path, realtime: bool = False, loop: bool = False
    ) -> ReplaySource:
        return cls(recording=Recording.load(path), realtime=realtime, loop=loop)

    def frames(self) -> Iterator[Frame]:
        while True:
            yield from replay(self.recording, realtime=self.realtime)
            if not self.loop:
                return

    def close(self) -> None:
        return None

    @property
    def last_frame(self) -> np.ndarray | None:
        """Replay has no video, only landmarks. The app falls back to a plain
        dark background, which is why ``--replay`` still renders."""
        return None

    @property
    def nominal_fps(self) -> float:
        return self.recording.fps


# --------------------------------------------------------------------------- #
# MediaPipe backends
# --------------------------------------------------------------------------- #
class _HandBackend(Protocol):
    """Adapter over one MediaPipe API generation."""

    name: str

    def process(self, rgb: np.ndarray, timestamp_ms: int) -> tuple[np.ndarray | None, str]: ...
    def close(self) -> None: ...


class _TasksBackend:
    """MediaPipe >= 0.10 Tasks API (``vision.HandLandmarker``).

    Uses ``RunningMode.VIDEO``, which requires strictly increasing timestamps and
    in exchange gives temporal tracking between frames -- markedly more stable
    than per-frame IMAGE mode, and that stability matters more here than raw
    accuracy, because jitter is what you actually see in the render.
    """

    name = "tasks"

    def __init__(
        self,
        num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        model_path: Path | None = None,
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        model = ensure_hand_model(model_path)
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, rgb: np.ndarray, timestamp_ms: int) -> tuple[np.ndarray | None, str]:
        # VIDEO mode rejects non-increasing timestamps with an opaque error, so
        # enforce monotonicity here rather than trusting wall-clock arithmetic.
        if timestamp_ms <= self._last_ts:
            timestamp_ms = self._last_ts + 1
        self._last_ts = timestamp_ms

        image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(rgb, dtype=np.uint8),
        )
        result = self._detector.detect_for_video(image, timestamp_ms)

        if not result.hand_landmarks:
            return None, "Right"

        landmarks = result.hand_landmarks[0]
        label = "Right"
        if result.handedness and result.handedness[0]:
            label = result.handedness[0][0].category_name

        lm = np.array([[p.x, p.y, p.z] for p in landmarks], dtype=np.float64)
        return lm, label

    def close(self) -> None:
        self._detector.close()


class _LegacyBackend:
    """MediaPipe < 1.0 ``solutions.hands.Hands`` API.

    Kept so the repo still runs for anyone on an older pinned MediaPipe.
    """

    name = "legacy-solutions"

    def __init__(
        self,
        num_hands: int,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        model_path: Path | None = None,
    ) -> None:
        import mediapipe as mp

        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def process(self, rgb: np.ndarray, timestamp_ms: int) -> tuple[np.ndarray | None, str]:
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None, "Right"

        hand = result.multi_hand_landmarks[0]
        label = "Right"
        if result.multi_handedness:
            label = result.multi_handedness[0].classification[0].label

        lm = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float64)
        return lm, label

    def close(self) -> None:
        self._hands.close()


def _make_backend(
    num_hands: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    model_path: Path | None = None,
) -> _HandBackend:
    """Pick the best available MediaPipe API generation.

    Tasks first: it is the supported path going forward, and it is the only one
    that exists in MediaPipe 1.0+.
    """
    try:
        import mediapipe  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required for live capture. "
            "Install with: pip install 'gesturebloom[live]'"
        ) from exc

    try:
        from mediapipe.tasks.python import vision  # noqa: F401
    except ImportError:
        pass
    else:
        return _TasksBackend(
            num_hands, min_detection_confidence, min_tracking_confidence, model_path
        )

    try:
        return _LegacyBackend(
            num_hands, min_detection_confidence, min_tracking_confidence, model_path
        )
    except AttributeError as exc:
        raise RuntimeError(
            "This MediaPipe build exposes neither the Tasks API "
            "(mediapipe.tasks.python.vision) nor the legacy solutions API "
            "(mediapipe.solutions.hands). Try: pip install -U mediapipe"
        ) from exc


class WebcamSource:
    """Live capture through MediaPipe, on whichever API version is installed.

    Two configuration notes that matter more than they look:

    ``num_hands``
        Tracking two hands roughly doubles landmark cost, the dominant stage in
        the latency budget. Default to one and raise it only when the
        interaction actually needs both.

    ``min_tracking_confidence``
        Raising it gives cleaner landmarks but more dropped frames. Because
        :class:`~gesturebloom.landmarks.filters.LandmarkSmoother` holds the last
        good pose across short dropouts, you can afford a higher threshold than
        you otherwise could -- prefer clean-or-absent over noisy-and-present.
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
        model_path: Path | None = None,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "opencv-python is required for live capture. "
                "Install with: pip install 'gesturebloom[live]'"
            ) from exc

        self._cv2 = cv2
        self.mirror = mirror
        self._cap = cv2.VideoCapture(camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, target_fps)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Could not open camera index {camera_index}.\n"
                f"On macOS, grant camera access: System Settings > Privacy & "
                f"Security > Camera > enable your terminal app, then fully quit "
                f"and reopen it."
            )

        self._backend = _make_backend(
            num_hands, min_detection_confidence, min_tracking_confidence, model_path
        )
        print(f"MediaPipe backend: {self._backend.name}")
        self._target_fps = float(target_fps)
        self._last_frame: np.ndarray | None = None

    @property
    def last_frame(self) -> np.ndarray | None:
        """The most recent RGB frame, for compositing behind the flower.

        Exposed as a property rather than added to the yielded tuple so that
        ReplaySource and every existing test keep working unchanged -- the frame
        is genuinely optional data that only one consumer wants.
        """
        return self._last_frame

    @property
    def backend_name(self) -> str:
        return self._backend.name

    def frames(self) -> Iterator[Frame]:
        cv2 = self._cv2
        start = time.perf_counter()
        last = start
        while True:
            ok, bgr = self._cap.read()
            if not ok:
                return
            if self.mirror:
                bgr = cv2.flip(bgr, 1)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self._last_frame = rgb

            now = time.perf_counter()
            lm, label = self._backend.process(rgb, int((now - start) * 1000.0))
            dt = now - last
            last = now

            if lm is None:
                yield None, "Right", dt
                continue

            # MediaPipe labels handedness for the *un-mirrored* image, so a
            # mirrored preview reverses it. Getting this wrong silently halves
            # your effective dataset by mislabeling chirality.
            if self.mirror:
                label = "Left" if label == "Right" else "Right"
            yield lm, label, dt

    def close(self) -> None:
        self._cap.release()
        self._backend.close()

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
