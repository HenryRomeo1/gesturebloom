"""Windowing, splitting, and augmentation for the two training tasks.

Two subtleties here are the difference between honest numbers and inflated ones.

**Windows overlap, so random splitting leaks.** If you cut a recording into
32-frame windows with a stride of 4 and then shuffle-split, adjacent windows
share 28 of 32 frames -- your test set is almost literally in your training set,
and your reported accuracy is fiction. Splits must be made at the **recording**
level, before windowing. :func:`split_recordings` does that, and
:class:`WindowDataset` refuses to be built from a mixed pool.

**Class balance must be measured on windows, not frames.** A gesture occupying
5% of frames can occupy 30% of windows, because any window overlapping it counts.
Compute weights from the windowed label distribution you actually train on.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..landmarks.canonical import augment, feature_vector, try_canonicalize
from .recording import Recording

HANDEDNESS_LEFT = 1


@dataclass
class WindowSpec:
    length: int = 32
    """Window length in frames. Must be <= the TCN receptive field, or the model
    cannot see the whole window and you are wasting data."""
    stride: int = 4
    min_valid_ratio: float = 0.75
    """Discard windows where tracking was lost for more than a quarter of frames."""


def canonical_features(recording: Recording, drop_wrist: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Canonicalize an entire recording into a feature matrix.

    Returns
    -------
    features:
        ``(T, D)`` float32, NaN rows where canonicalization failed.
    valid:
        ``(T,)`` bool.
    """
    T = len(recording)
    dim = (20 if drop_wrist else 21) * 3
    feats = np.full((T, dim), np.nan, dtype=np.float32)
    valid = np.zeros(T, dtype=bool)

    for t in range(T):
        lm = recording.landmarks[t]
        if np.isnan(lm).any():
            continue
        hand = "Left" if recording.handedness[t] == HANDEDNESS_LEFT else "Right"
        res = try_canonicalize(lm.astype(np.float64), hand)
        if res is None:
            continue
        feats[t] = feature_vector(res[0], drop_wrist=drop_wrist)
        valid[t] = True
    return feats, valid


def split_recordings(
    paths: list[str | Path],
    val_fraction: float = 0.2,
    test_fraction: float = 0.15,
    seed: int = 0,
) -> dict[str, list[Path]]:
    """Split at the recording level to prevent window overlap leakage.

    Record each pose in several separate takes so this split is meaningful: if
    all your data for a class lives in one file, that class ends up entirely in
    one split.

    Examples
    --------
    >>> s = split_recordings([f"r{i}.npz" for i in range(10)], seed=1)
    >>> sorted(s)
    ['test', 'train', 'val']
    >>> len(s['train']) + len(s['val']) + len(s['test'])
    10
    >>> set(s['train']) & set(s['val'])
    set()
    """
    rng = np.random.default_rng(seed)
    items = [Path(p) for p in paths]
    order = rng.permutation(len(items))
    n_test = round(len(items) * test_fraction)
    n_val = round(len(items) * val_fraction)
    idx_test = order[:n_test]
    idx_val = order[n_test : n_test + n_val]
    idx_train = order[n_test + n_val :]
    return {
        "train": [items[i] for i in idx_train],
        "val": [items[i] for i in idx_val],
        "test": [items[i] for i in idx_test],
    }


class PoseDataset:
    """Per-frame samples for the static pose classifier.

    Examples
    --------
    >>> from gesturebloom.data.recording import synthetic_recording
    >>> ds = PoseDataset([synthetic_recording(n_frames=120, seed=2)], n_classes=3)
    >>> x, y = ds[0]
    >>> x.shape, int(y) in (0, 1, 2)
    ((60,), True)
    >>> len(ds) > 100
    True
    """

    def __init__(
        self,
        recordings: list[Recording],
        n_classes: int,
        train: bool = True,
        seed: int = 0,
    ) -> None:
        self.n_classes = n_classes
        self.train = train
        self._rng = np.random.default_rng(seed)

        feats: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        for rec in recordings:
            f, valid = canonical_features(rec)
            feats.append(f[valid])
            labels.append(rec.labels[valid])
        self.features = np.concatenate(feats).astype(np.float32)
        self.labels = np.concatenate(labels).astype(np.int64)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.int64]:
        x = self.features[i]
        if self.train:
            x = augment(x.reshape(-1, 3), self._rng).reshape(-1)
        return x.astype(np.float32), self.labels[i]

    def class_counts(self) -> np.ndarray:
        return np.bincount(self.labels, minlength=self.n_classes)


class WindowDataset:
    """Sliding windows for the temporal spotter.

    Each item is ``(features (L, D), soft_targets (L, C))``. Targets are soft --
    see :func:`~gesturebloom.models.temporal.soft_onset_targets`.

    Examples
    --------
    >>> from gesturebloom.data.recording import synthetic_recording
    >>> recs = [synthetic_recording(n_frames=200, seed=s) for s in (1, 2)]
    >>> ds = WindowDataset(recs, n_classes=3, spec=WindowSpec(length=32, stride=8))
    >>> x, y = ds[0]
    >>> x.shape, y.shape
    ((32, 60), (32, 3))
    >>> bool(abs(y.sum(axis=1).mean() - 1.0) < 1e-5)
    True
    """

    def __init__(
        self,
        recordings: list[Recording],
        n_classes: int,
        spec: WindowSpec | None = None,
        train: bool = True,
        ramp: int = 3,
        seed: int = 0,
    ) -> None:
        from ..models.temporal import soft_onset_targets  # local: avoids torch at import

        self.spec = spec or WindowSpec()
        self.n_classes = n_classes
        self.train = train
        self._rng = np.random.default_rng(seed)

        self._x: list[np.ndarray] = []
        self._y: list[np.ndarray] = []
        L, S = self.spec.length, self.spec.stride

        for rec in recordings:
            feats, valid = canonical_features(rec)
            targets = soft_onset_targets(rec.labels, n_classes=n_classes, ramp=ramp)
            for start in range(0, max(len(rec) - L + 1, 0), S):
                sl = slice(start, start + L)
                v = valid[sl]
                if v.mean() < self.spec.min_valid_ratio:
                    continue
                window = feats[sl].copy()
                # Hold-last-value interpolation across dropped frames, matching
                # what LandmarkSmoother does at inference. Train/test parity in
                # the *dropout handling* matters as much as in the features.
                for t in range(L):
                    if not v[t]:
                        window[t] = window[t - 1] if t > 0 else 0.0
                self._x.append(window)
                self._y.append(targets[sl])

        if not self._x:
            raise ValueError("no valid windows produced; check tracking quality")
        self.x = np.stack(self._x).astype(np.float32)
        self.y = np.stack(self._y).astype(np.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        x, y = self.x[i], self.y[i]
        if self.train:
            x = self._augment_window(x)
        return x.astype(np.float32), y

    def _augment_window(self, x: np.ndarray) -> np.ndarray:
        L, D = x.shape
        out = x.reshape(L, D // 3, 3).copy()
        for t in range(L):
            out[t] = augment(out[t], self._rng, rot_sigma_deg=5.0, scale_sigma=0.04)
        out = out.reshape(L, D)
        # Time warp: resample the window at a slightly different rate so the
        # model does not memorize one gesture speed.
        if self._rng.random() < 0.5:
            factor = float(self._rng.uniform(0.85, 1.18))
            src = np.clip(np.linspace(0, (L - 1) * factor, L), 0, L - 1)
            lo = np.floor(src).astype(int)
            hi = np.minimum(lo + 1, L - 1)
            w = (src - lo)[:, None]
            out = out[lo] * (1 - w) + out[hi] * w
        return out

    def window_class_counts(self) -> np.ndarray:
        """Class counts over the *windowed* targets -- the distribution you train on."""
        hard = self.y.argmax(axis=-1).reshape(-1)
        return np.bincount(hard, minlength=self.n_classes)
