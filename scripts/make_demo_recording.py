#!/usr/bin/env python3
"""Generate the example recording used by `make demo` and the README quickstart.

Exists so a first-time visitor can run the full pipeline in under a minute with
no webcam, no GPU, and no training. Lowering that barrier is the single highest
-leverage thing you can do for a portfolio repo -- most people who open it will
never plug in a camera.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from gesturebloom.data.recording import synthetic_recording


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("data/examples/demo.npz"))
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    rec = synthetic_recording(n_frames=args.frames, seed=args.seed, dropout_rate=0.02)
    path = rec.save(args.out)
    print(f"Wrote {path} ({len(rec)} frames, {rec.tracking_ratio:.1%} tracked, {rec.fps:.0f} fps)")
    print(f"Ground-truth onsets: {rec.onsets()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
