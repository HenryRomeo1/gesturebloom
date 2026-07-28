"""End-to-end pipeline tests that run with no camera, no GPU, and no display.

This file is the payoff for the recording format. It exercises capture-to-render
plumbing in CI, which is normally the untested part of a project like this.
"""

from __future__ import annotations

import numpy as np
import pytest

from gesturebloom.control.calibration import (
    CalibrationCollector,
    CalibrationProfile,
    default_ranges,
)
from gesturebloom.control.mapper import ControlMapper, frame_signals, raw_signals
from gesturebloom.data.recording import Recording, replay, synthetic_recording
from gesturebloom.geometry.spiderlily import build_spiderlily
from gesturebloom.landmarks.canonical import try_canonicalize
from gesturebloom.landmarks.filters import LandmarkSmoother, OneEuroFilter
from gesturebloom.render.window import build_batch


def run_pipeline(recording: Recording, ranges=None) -> list[dict[str, float]]:
    smoother = LandmarkSmoother(freq=recording.fps)
    mapper = ControlMapper(ranges=ranges or default_ranges(), freq=recording.fps)
    out = []
    for lm, hand, dt in replay(recording):
        smoothed = smoother.update(lm, dt)
        if smoothed is None:
            continue
        res = try_canonicalize(smoothed, hand)
        if res is None:
            continue
        canonical, frame = res
        out.append(mapper.update(canonical, dt=dt, basis=frame.basis))
    return out


def test_recording_roundtrip(tmp_path) -> None:
    rec = synthetic_recording(n_frames=120, seed=4)
    path = rec.save(tmp_path / "take.npz")
    loaded = Recording.load(path)
    np.testing.assert_array_equal(
        np.nan_to_num(rec.landmarks, nan=-1.0), np.nan_to_num(loaded.landmarks, nan=-1.0)
    )
    np.testing.assert_array_equal(rec.labels, loaded.labels)
    assert loaded.label_names == rec.label_names
    assert loaded.meta["seed"] == 4


def test_dropouts_marked_with_nan_not_zero() -> None:
    """Dropped frames must be NaN. Zeros would read as a hand at the origin."""
    rec = synthetic_recording(n_frames=400, seed=11, dropout_rate=0.15)
    dropped = np.isnan(rec.landmarks).any(axis=(1, 2))
    assert dropped.sum() > 0
    assert not np.any(rec.landmarks[dropped] == 0.0)
    assert rec.tracking_ratio < 1.0


def test_pipeline_produces_bounded_params() -> None:
    params = run_pipeline(synthetic_recording(n_frames=300, seed=6))
    assert len(params) > 200
    for p in params:
        assert set(p) == {"grow", "bloom", "sway"}
        for name, value in p.items():
            assert 0.0 <= value <= 1.0, f"{name}={value} out of range"
            assert np.isfinite(value)


def test_pipeline_is_deterministic() -> None:
    """Same recording in, identical parameters out -- every time."""
    rec = synthetic_recording(n_frames=200, seed=8)
    a, b = run_pipeline(rec), run_pipeline(rec)
    assert len(a) == len(b)
    for pa, pb in zip(a, b, strict=True):
        for k in pa:
            assert pa[k] == pytest.approx(pb[k], abs=1e-12)


def test_signals_actually_vary() -> None:
    """Guards against a silently-constant control signal.

    A mapper bug that pins a parameter to 0 passes every bounds check while making
    the project completely non-interactive. Assert variance explicitly.
    """
    params = run_pipeline(synthetic_recording(n_frames=400, seed=9))
    for key in ("grow", "bloom"):
        values = np.array([p[key] for p in params])
        assert values.std() > 0.02, f"{key} barely moves (std={values.std():.4f})"
        assert values.max() - values.min() > 0.15


def test_smoothing_reduces_jitter() -> None:
    """The filter must measurably reduce frame-to-frame variation."""
    rng = np.random.default_rng(0)
    clean = np.sin(np.linspace(0, 6 * np.pi, 400))
    noisy = clean + rng.normal(0, 0.25, clean.shape)
    f = OneEuroFilter(freq=60.0, min_cutoff=1.0, beta=0.01)
    filtered = np.array([f(float(v), dt=1 / 60) for v in noisy])

    raw_jitter = np.abs(np.diff(noisy)).mean()
    filt_jitter = np.abs(np.diff(filtered)).mean()
    assert filt_jitter < raw_jitter * 0.5


def test_smoother_holds_through_short_dropout() -> None:
    """Short tracking gaps hold the last pose; long gaps release to None."""
    smoother = LandmarkSmoother(freq=60.0, hold_frames=5)
    lm = synthetic_recording(n_frames=10, seed=0, dropout_rate=0.0).landmarks[0].astype(np.float64)
    assert smoother.update(lm, 1 / 60) is not None
    for _ in range(5):
        assert smoother.update(None, 1 / 60) is not None
    assert smoother.update(None, 1 / 60) is None


def test_calibration_profile_roundtrip(tmp_path) -> None:
    rec = synthetic_recording(n_frames=300, seed=12)
    collector = CalibrationCollector()
    for lm, hand, _dt in replay(rec):
        if lm is None:
            continue
        res = try_canonicalize(lm, hand)
        if res is None:
            continue
        raw = raw_signals(res[0])
        raw.update(frame_signals(res[1].basis))
        collector.add(raw)

    profile = collector.finish(note="test")
    path = tmp_path / "cal.json"
    profile.save(path)
    loaded = CalibrationProfile.load(path)
    for name, rng in profile.ranges.items():
        assert loaded.ranges[name].lo == pytest.approx(rng.lo)
        assert loaded.ranges[name].hi == pytest.approx(rng.hi)


def test_calibration_version_mismatch_rejected() -> None:
    """A stale profile must fail loudly rather than be misinterpreted."""
    with pytest.raises(ValueError, match="version"):
        CalibrationProfile.from_dict({"version": 999, "ranges": {}})


def test_calibration_widens_usable_range() -> None:
    """Calibrated ranges should use more of [0, 1] than the generic defaults."""
    rec = synthetic_recording(n_frames=400, seed=13)
    collector = CalibrationCollector()
    for lm, hand, _dt in replay(rec):
        if lm is None:
            continue
        res = try_canonicalize(lm, hand)
        if res is not None:
            collector.add(raw_signals(res[0]))
    calibrated = collector.finish().ranges

    default_spread = np.std([p["grow"] for p in run_pipeline(rec, default_ranges())])
    cal_spread = np.std([p["grow"] for p in run_pipeline(rec, calibrated)])
    assert cal_spread >= default_spread * 0.9


def test_geometry_to_vertex_buffer_end_to_end() -> None:
    params = run_pipeline(synthetic_recording(n_frames=120, seed=14))
    for p in params[::10]:
        data, counts = build_batch(build_spiderlily(p["grow"], p["bloom"], seed=0))
        assert np.all(np.isfinite(data))
        assert int(counts.sum()) == data.shape[0]


def test_onsets_extracted_from_labels() -> None:
    rec = synthetic_recording(n_frames=300, seed=15)
    onsets = rec.onsets()
    assert all(cls != 0 for cls, _ in onsets)
    assert all(0 <= frame < len(rec) for _, frame in onsets)
    assert onsets == sorted(onsets, key=lambda x: x[1])
