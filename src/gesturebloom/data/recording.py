"""The ``.npz`` landmark recording format, and deterministic replay.

This module is the reason the rest of the repo is testable, and it is worth
putting near the top of the README because almost no gesture-recognition repo
has it.

The problem with webcam-driven projects is that nothing is reproducible. A bug
that appears at a particular hand angle cannot be re-triggered on demand. Tests
cannot run in CI. Two runs of the same "experiment" differ because your hand
differed. Threshold tuning becomes an afternoon of waving at a laptop.

The fix is to make the landmark stream a first-class serializable artifact.
Capture once, then everything downstream -- canonicalization, control mapping,
model inference, spotting, rendering -- runs from the file, identically, every
time. The ``--replay`` flag on every CLI command exists because of this module,
and CI runs the full pipeline with no camera and no display.

Format
------
Uncompressed ``.npz`` (uncompressed because we want mmap-speed random access and
these files are small -- a minute at 60 fps is about 1 MB):

``landmarks``    ``(T, 21, 3)`` float32. NaN-filled rows mean tracking was lost.
``handedness``   ``(T,)`` int8. ``0`` right, ``1`` left, ``-1`` no hand.
``timestamps``   ``(T,)`` float64, seconds since recording start.
``labels``       ``(T,)`` int64 frame-wise class, ``0`` = background.
``label_names``  ``(n_classes,)`` unicode.
``meta``         JSON string: fps, source resolution, model version, notes.

NaN rather than a separate mask because it propagates loudly. A silently-zeroed
dropped frame becomes a hand at the origin and produces a plausible-looking
wrong answer; NaN makes the bug obvious at the first arithmetic operation.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FORMAT_VERSION = 1
HANDEDNESS_RIGHT = 0
HANDEDNESS_LEFT = 1
HANDEDNESS_NONE = -1


@dataclass
class Recording:
    """An in-memory landmark recording."""

    landmarks: np.ndarray  # (T, 21, 3) float32, NaN where tracking lost
    handedness: np.ndarray  # (T,) int8
    timestamps: np.ndarray  # (T,) float64
    labels: np.ndarray  # (T,) int64
    label_names: list[str] = field(default_factory=lambda: ["background"])
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        T = self.landmarks.shape[0]
        if self.landmarks.shape[1:] != (21, 3):
            raise ValueError(f"landmarks must be (T, 21, 3), got {self.landmarks.shape}")
        for name in ("handedness", "timestamps", "labels"):
            arr = getattr(self, name)
            if arr.shape != (T,):
                raise ValueError(f"{name} must have shape ({T},), got {arr.shape}")

    def __len__(self) -> int:
        return int(self.landmarks.shape[0])

    @property
    def fps(self) -> float:
        """Measured frame rate from timestamps, not the nominal capture rate.

        Always report the measured value -- webcams routinely deliver 24 fps while
        claiming 30, and a latency budget computed against the nominal rate is
        wrong.
        """
        if len(self) < 2:
            return float(self.meta.get("fps", 60.0))
        dt = float(np.median(np.diff(self.timestamps)))
        return 1.0 / dt if dt > 0 else float(self.meta.get("fps", 60.0))

    @property
    def tracking_ratio(self) -> float:
        """Fraction of frames with a valid hand. Below ~0.9, fix your lighting
        before you touch a hyperparameter."""
        if len(self) == 0:
            return 0.0
        valid = ~np.isnan(self.landmarks).any(axis=(1, 2))
        return float(valid.mean())

    def valid_mask(self) -> np.ndarray:
        return ~np.isnan(self.landmarks).any(axis=(1, 2))

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(self.meta)
        meta.setdefault("format_version", FORMAT_VERSION)
        meta.setdefault("fps", self.fps)
        np.savez(
            p,
            landmarks=self.landmarks.astype(np.float32),
            handedness=self.handedness.astype(np.int8),
            timestamps=self.timestamps.astype(np.float64),
            labels=self.labels.astype(np.int64),
            label_names=np.array(self.label_names, dtype=object),
            meta=np.array(json.dumps(meta)),
        )
        return p if p.suffix == ".npz" else p.with_suffix(".npz")

    @classmethod
    def load(cls, path: str | Path) -> Recording:
        with np.load(Path(path), allow_pickle=True) as z:
            meta = json.loads(str(z["meta"].item()))
            version = int(meta.get("format_version", 0))
            if version > FORMAT_VERSION:
                raise ValueError(
                    f"recording format version {version} is newer than supported "
                    f"version {FORMAT_VERSION}; upgrade gesturebloom"
                )
            return cls(
                landmarks=z["landmarks"].astype(np.float32),
                handedness=z["handedness"].astype(np.int8),
                timestamps=z["timestamps"].astype(np.float64),
                labels=z["labels"].astype(np.int64),
                label_names=[str(s) for s in z["label_names"].tolist()],
                meta=meta,
            )

    def slice(self, start: int, stop: int) -> Recording:
        return Recording(
            landmarks=self.landmarks[start:stop],
            handedness=self.handedness[start:stop],
            timestamps=self.timestamps[start:stop],
            labels=self.labels[start:stop],
            label_names=list(self.label_names),
            meta=dict(self.meta),
        )

    def onsets(self) -> list[tuple[int, int]]:
        """Ground-truth ``(class_index, onset_frame)`` pairs from frame labels.

        An onset is any transition into a non-background class. This is what you
        feed to :func:`~gesturebloom.models.spotter.evaluate_events`.

        Examples
        --------
        >>> r = synthetic_recording(n_frames=20, seed=0)
        >>> r.labels[:] = 0
        >>> r.labels[5:10] = 1
        >>> r.labels[14:18] = 2
        >>> r.onsets()
        [(1, 5), (2, 14)]
        """
        out: list[tuple[int, int]] = []
        prev = 0
        for t, lab in enumerate(self.labels.tolist()):
            if lab != 0 and lab != prev:
                out.append((int(lab), t))
            prev = lab
        return out


class RecordingWriter:
    """Accumulates frames during capture, then writes one ``.npz``.

    Append-only and allocation-free per frame (Python lists, converted once at
    the end), so recording does not perturb the latency you are trying to measure.
    """

    def __init__(self, label_names: list[str] | None = None, meta: dict | None = None) -> None:
        self._landmarks: list[np.ndarray] = []
        self._handedness: list[int] = []
        self._timestamps: list[float] = []
        self._labels: list[int] = []
        self.label_names = label_names or ["background"]
        self.meta = meta or {}

    def add(
        self,
        landmarks: np.ndarray | None,
        timestamp: float,
        handedness: str | int = HANDEDNESS_NONE,
        label: int = 0,
    ) -> None:
        if landmarks is None:
            frame = np.full((21, 3), np.nan, dtype=np.float32)
            hand = HANDEDNESS_NONE
        else:
            frame = np.asarray(landmarks, dtype=np.float32).reshape(21, 3)
            if isinstance(handedness, str):
                hand = HANDEDNESS_LEFT if handedness.lower().startswith("l") else HANDEDNESS_RIGHT
            else:
                hand = int(handedness)
        self._landmarks.append(frame)
        self._handedness.append(hand)
        self._timestamps.append(float(timestamp))
        self._labels.append(int(label))

    def __len__(self) -> int:
        return len(self._landmarks)

    def finish(self) -> Recording:
        if not self._landmarks:
            raise ValueError("no frames recorded")
        return Recording(
            landmarks=np.stack(self._landmarks),
            handedness=np.array(self._handedness, dtype=np.int8),
            timestamps=np.array(self._timestamps, dtype=np.float64),
            labels=np.array(self._labels, dtype=np.int64),
            label_names=list(self.label_names),
            meta=dict(self.meta),
        )


def replay(recording: Recording, realtime: bool = False) -> Iterator[tuple[np.ndarray | None, str, float]]:
    """Yield ``(landmarks_or_None, handedness_str, dt)`` frame by frame.

    Matches the interface of the live capture source exactly, which is what lets
    every downstream component be swapped between live and replay with one flag.

    Parameters
    ----------
    realtime:
        Sleep to match the original timestamps. Off by default so tests and
        benchmarks run at full speed.

    Examples
    --------
    >>> r = synthetic_recording(n_frames=5, seed=1)
    >>> frames = list(replay(r))
    >>> len(frames)
    5
    >>> frames[0][0].shape
    (21, 3)
    """
    import time

    ts = recording.timestamps
    for t in range(len(recording)):
        frame = recording.landmarks[t]
        valid = not bool(np.isnan(frame).any())
        hand = "Left" if recording.handedness[t] == HANDEDNESS_LEFT else "Right"
        dt = float(ts[t] - ts[t - 1]) if t > 0 else 1.0 / max(recording.fps, 1e-6)
        if realtime and dt > 0:
            time.sleep(dt)
        yield (frame.astype(np.float64) if valid else None, hand, dt)


def synthetic_recording(
    n_frames: int = 240,
    fps: float = 60.0,
    seed: int = 0,
    dropout_rate: float = 0.02,
) -> Recording:
    """Generate a plausible synthetic recording for tests and CI.

    Builds a hand skeleton with jointed fingers whose flexion oscillates, so the
    derived control signals actually vary the way real ones do. Not a substitute
    for real data when training -- but exactly what you want for testing plumbing,
    and it lets CI exercise the full pipeline with no camera.

    Examples
    --------
    >>> r = synthetic_recording(n_frames=100, seed=7)
    >>> len(r)
    100
    >>> bool(0.0 < r.tracking_ratio <= 1.0)
    True
    >>> round(r.fps)
    60
    """
    rng = np.random.default_rng(seed)
    writer = RecordingWriter(
        label_names=["background", "pinch", "spread"],
        meta={"synthetic": True, "seed": seed, "fps": fps},
    )

    mcp_x = np.linspace(-0.35, 0.35, 4)
    for t in range(n_frames):
        phase = 2.0 * np.pi * t / max(n_frames, 1)
        flex = 0.5 + 0.5 * np.sin(phase * 3.0)  # 0 = extended, 1 = curled

        lm = np.zeros((21, 3), dtype=np.float64)
        lm[0] = [0.0, 0.0, 0.0]  # wrist
        lm[9] = [0.0, 1.0, 0.0]  # middle MCP defines the hand-span unit

        # Thumb: swings across the palm with flexion.
        thumb_ang = 0.9 - 0.6 * flex
        for i, k in enumerate((1, 2, 3, 4), start=1):
            r = 0.22 * i
            lm[k] = [-np.sin(thumb_ang) * r * 1.6, np.cos(thumb_ang) * r, 0.05 * i * flex]

        # Four fingers: MCP fixed, phalanges bend progressively.
        for f, (mcp, pip, dip, tip) in enumerate(
            ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20))
        ):
            base = np.array([mcp_x[f], 0.95, 0.0])
            lm[mcp] = base
            seg, ang = 0.30, 0.0
            pos = base.copy()
            for j, k in enumerate((pip, dip, tip)):
                ang += flex * (0.55 + 0.12 * j)
                pos = pos + np.array([0.0, np.cos(ang) * seg, np.sin(ang) * seg])
                lm[k] = pos
                seg *= 0.82

        lm += rng.normal(0.0, 0.004, size=lm.shape)  # landmark estimation noise
        # Random global pose -- canonicalization must undo all of this.
        ang = rng.uniform(-0.4, 0.4)
        R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1]])
        lm = lm @ R.T * rng.uniform(0.8, 1.25) + rng.uniform(-0.3, 0.3, size=3)

        dropped = rng.random() < dropout_rate
        label = 1 if flex > 0.85 else (2 if flex < 0.15 else 0)
        writer.add(
            None if dropped else lm,
            timestamp=t / fps,
            handedness="Left" if rng.random() < 0.3 else "Right",
            label=0 if dropped else label,
        )

    return writer.finish()
