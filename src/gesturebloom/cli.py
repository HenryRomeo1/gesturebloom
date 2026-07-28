"""Command-line interface.

Uses ``argparse`` rather than click/typer so the base install has no CLI
dependency at all -- the package installs with numpy and pyyaml only.

Every command that consumes landmarks accepts ``--replay PATH``, and the ones
that produce visuals accept ``--headless``. That is what makes the whole project
runnable and testable without a camera or a display.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import AppConfig, load_config


def _add_source_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--replay", type=Path, default=None, help="Play a recorded .npz instead of the webcam")
    p.add_argument("--camera", type=int, default=0, help="Camera index for live capture")
    p.add_argument("--config", type=Path, default=None, help="YAML config path")
    p.add_argument("--realtime", action="store_true", help="Replay at the original frame rate")


# --------------------------------------------------------------------------- #
# record
# --------------------------------------------------------------------------- #
def cmd_record(args: argparse.Namespace) -> int:
    """Capture a labelled landmark recording from the webcam."""
    from .data.recording import RecordingWriter
    from .landmarks.source import open_source

    labels = ["background", *args.labels]
    print(f"Recording to {args.out}")
    print(f"Classes: {', '.join(f'{i}={n}' for i, n in enumerate(labels))}")
    print(f"Press keys 0-{len(labels) - 1} to set the active label, 'q' to stop.\n")

    source = open_source(replay_path=args.replay, camera_index=args.camera)
    writer = RecordingWriter(label_names=labels, meta={"source": "webcam", "note": args.note})
    active = 0
    elapsed = 0.0

    try:
        for lm, hand, dt in source.frames():
            elapsed += dt
            writer.add(lm, timestamp=elapsed, handedness=hand, label=active)
            if len(writer) % 30 == 0:
                print(f"\r{len(writer)} frames | {elapsed:5.1f}s | label={labels[active]}   ", end="")
            if args.max_frames and len(writer) >= args.max_frames:
                break
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        source.close()

    rec = writer.finish()
    path = rec.save(args.out)
    print(f"\nSaved {len(rec)} frames to {path}")
    print(f"Tracking ratio: {rec.tracking_ratio:.1%} | measured fps: {rec.fps:.1f}")
    if rec.tracking_ratio < 0.9:
        print("WARNING: tracking below 90%. Improve lighting or move closer before training.")
    return 0


# --------------------------------------------------------------------------- #
# calibrate
# --------------------------------------------------------------------------- #
def cmd_calibrate(args: argparse.Namespace) -> int:
    """Run the guided per-user calibration and write a profile."""
    from .control.calibration import CALIBRATION_SEQUENCE, CalibrationCollector
    from .control.mapper import frame_signals, raw_signals
    from .landmarks.canonical import try_canonicalize
    from .landmarks.source import open_source

    source = open_source(replay_path=args.replay, camera_index=args.camera, realtime=args.realtime)
    collector = CalibrationCollector()
    fps = source.nominal_fps

    print("Calibration: follow each prompt.\n")
    frames = source.frames()
    for step in CALIBRATION_SEQUENCE:
        print(f"  -> {step.prompt}")
        target = int(step.seconds * fps)
        collected = 0
        for lm, hand, _dt in frames:
            if lm is not None:
                res = try_canonicalize(lm, hand)
                if res is not None:
                    canonical, hand_frame = res
                    raw = raw_signals(canonical)
                    raw.update(frame_signals(hand_frame.basis))
                    collector.add(raw)
                    collected += 1
            if collected >= target:
                break
        print(f"     collected {collected} samples")

    source.close()
    profile = collector.finish(note=args.note)
    profile.save(args.out)
    print(f"\nSaved calibration to {args.out}")
    for name, rng in sorted(profile.ranges.items()):
        print(f"  {name:10s} [{rng.lo:+.3f}, {rng.hi:+.3f}]")
    return 0


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace) -> int:
    """Drive the flower from a landmark source."""
    from .control.calibration import CalibrationProfile, default_ranges
    from .control.mapper import ControlMapper
    from .landmarks.canonical import try_canonicalize
    from .landmarks.filters import LandmarkSmoother
    from .landmarks.source import open_source

    config: AppConfig = load_config(args.config)
    ranges = (
        CalibrationProfile.load(args.calibration).ranges
        if args.calibration
        else default_ranges()
    )
    if not args.calibration:
        print("No --calibration given; using default ranges. Run `gesturebloom calibrate` for a better feel.")

    source = open_source(
        replay_path=args.replay,
        camera_index=args.camera,
        realtime=args.realtime,
        loop=args.loop,
    )
    smoother = LandmarkSmoother(
        freq=source.nominal_fps,
        min_cutoff=config.smoothing.min_cutoff,
        beta=config.smoothing.beta,
    )
    mapper = ControlMapper(ranges=ranges, freq=source.nominal_fps)

    if not args.headless:
        # Windowed mode inverts control: the windowing library owns the loop, so
        # we hand everything over rather than driving it from here.
        try:
            from .render.app import run_windowed
            from .render.window import RenderConfig

            return run_windowed(
                source=source,
                smoother=smoother,
                mapper=mapper,
                render_config=RenderConfig(
                    width=config.render.width,
                    height=config.render.height,
                    bloom_passes=config.render.bloom_passes,
                    bloom_strength=config.render.bloom_strength,
                ),
                flower_seed=config.flower_seed,
                record_frames=args.record_frames,
                max_frames=args.max_frames,
                vsync=config.render.vsync,
            )
        except ImportError as exc:
            print(f"Rendering unavailable ({exc}); falling back to --headless.")

    traces: list[dict] = []
    n = 0
    try:
        for lm, hand, dt in source.frames():
            smoothed = smoother.update(lm, dt)
            if smoothed is None:
                continue
            res = try_canonicalize(smoothed, hand)
            if res is None:
                continue
            canonical, hand_frame = res
            params = mapper.update(canonical, dt=dt, basis=hand_frame.basis)
            if args.dump_params:
                traces.append({"frame": n, **{k: round(v, 5) for k, v in params.items()}})
            if args.headless and n % 30 == 0:
                print(f"\rframe {n:5d} | grow {params['grow']:.3f} | bloom {params['bloom']:.3f}   ", end="")
            n += 1
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        source.close()

    print(f"\nProcessed {n} frames")
    if args.dump_params:
        Path(args.dump_params).write_text(json.dumps(traces, indent=1), encoding="utf-8")
        print(f"Wrote parameter trace to {args.dump_params}")
    return 0


# --------------------------------------------------------------------------- #
# train
# --------------------------------------------------------------------------- #
def cmd_train(args: argparse.Namespace) -> int:
    """Train the pose classifier or the temporal spotter."""
    from .training import train_pose, train_temporal

    paths = sorted(Path(args.data).glob("*.npz"))
    if not paths:
        print(f"No .npz recordings found in {args.data}", file=sys.stderr)
        return 1
    print(f"Found {len(paths)} recordings")

    if args.task == "pose":
        return train_pose(paths, out=args.out, epochs=args.epochs, seed=args.seed)
    return train_temporal(paths, out=args.out, epochs=args.epochs, seed=args.seed)


# --------------------------------------------------------------------------- #
# bench
# --------------------------------------------------------------------------- #
def cmd_bench(args: argparse.Namespace) -> int:
    """Measure per-stage latency and print a markdown table."""
    from .bench.latency import benchmark_pipeline

    timer = benchmark_pipeline(
        n_frames=args.frames, seed=args.seed, include_inference=args.with_inference
    )
    table = timer.markdown_table(title=args.title or "Pipeline latency")
    print(table)
    if args.out:
        Path(args.out).write_text(table + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}")
    return 0


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def cmd_export(args: argparse.Namespace) -> int:
    """Export a trained checkpoint to ONNX."""
    from .models.export import export_onnx

    path = export_onnx(args.checkpoint, args.out, task=args.task, opset=args.opset)
    print(f"Exported {args.task} model to {path}")
    return 0


# --------------------------------------------------------------------------- #
# tune
# --------------------------------------------------------------------------- #
def cmd_tune(args: argparse.Namespace) -> int:
    """Grid-search spotter thresholds against a recorded probability trace.

    This is the workflow the recording format exists to enable: hundreds of
    threshold combinations evaluated in seconds, offline, reproducibly -- instead
    of waving at a webcam and guessing.
    """
    from .models.spotter import SpotterConfig, evaluate_events, spot_offline

    probs = np.load(args.probs)["probs"]
    truth = [tuple(x) for x in json.loads(Path(args.truth).read_text(encoding="utf-8"))]

    best = None
    print(f"{'enter':>6} {'exit':>6} {'min_f':>6} {'F1':>7} {'prec':>7} {'rec':>7} {'lat_f':>7}")
    for enter in np.arange(0.5, 0.91, 0.05):
        for exit_ in np.arange(0.25, float(enter) - 0.05, 0.05):
            for min_frames in (2, 3, 4, 5):
                cfg = SpotterConfig(
                    enter_threshold=float(enter),
                    exit_threshold=float(exit_),
                    min_frames=min_frames,
                    refractory_frames=args.refractory,
                )
                m = evaluate_events(spot_offline(probs, cfg), truth, tolerance_frames=args.tolerance)
                score = m.f1
                if best is None or score > best[0]:
                    best = (score, cfg, m)
                    print(
                        f"{enter:6.2f} {exit_:6.2f} {min_frames:6d} {m.f1:7.3f} "
                        f"{m.precision:7.3f} {m.recall:7.3f} {m.median_latency_frames:7.1f}"
                    )

    if best is None:
        print("no configuration evaluated", file=sys.stderr)
        return 1
    score, cfg, m = best
    print(f"\nBest: {m.summary(fps=args.fps)}")
    print(f"  enter={cfg.enter_threshold:.2f} exit={cfg.exit_threshold:.2f} min_frames={cfg.min_frames}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gesturebloom",
        description="Gesture-driven procedural spiderlily. Real-time hand landmarks -> GPU render.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("record", help="Capture a labelled landmark recording")
    _add_source_args(p)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--labels", nargs="+", default=["pinch", "spread", "swipe"])
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--note", default="")
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("calibrate", help="Guided per-user range calibration")
    _add_source_args(p)
    p.add_argument("--out", type=Path, default=Path("calibration.json"))
    p.add_argument("--note", default="")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("run", help="Drive the flower live or from a recording")
    _add_source_args(p)
    p.add_argument("--calibration", type=Path, default=None)
    p.add_argument("--headless", action="store_true", help="No window; print parameters")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--dump-params", type=Path, default=None)
    p.add_argument(
        "--record-frames",
        type=Path,
        default=None,
        help="Dump each rendered frame as PNG into this directory (for making a GIF)",
    )
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("train", help="Train the pose classifier or temporal spotter")
    p.add_argument("task", choices=["pose", "temporal"])
    p.add_argument("--data", type=Path, required=True, help="Directory of .npz recordings")
    p.add_argument("--out", type=Path, default=Path("checkpoints"))
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("bench", help="Per-stage latency table")
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--with-inference", action="store_true")
    p.add_argument("--title", default="")
    p.add_argument("--out", type=Path, default=None)
    p.set_defaults(func=cmd_bench)

    p = sub.add_parser("export", help="Export a checkpoint to ONNX")
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--task", choices=["pose", "temporal"], default="pose")
    p.add_argument("--opset", type=int, default=17)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("gl-probe", help="Verify the GL setup and compile all shaders")
    p.set_defaults(func=lambda _a: __import__(
        "gesturebloom.render.app", fromlist=["probe_gl"]
    ).probe_gl())

    p = sub.add_parser("tune", help="Grid-search spotter thresholds offline")
    p.add_argument("--probs", type=Path, required=True, help=".npz with a 'probs' (T, C) array")
    p.add_argument("--truth", type=Path, required=True, help="JSON list of [class, frame] onsets")
    p.add_argument("--tolerance", type=int, default=15)
    p.add_argument("--refractory", type=int, default=12)
    p.add_argument("--fps", type=float, default=60.0)
    p.set_defaults(func=cmd_tune)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
