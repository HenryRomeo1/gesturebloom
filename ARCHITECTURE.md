# Architecture

## Data flow

```
                  ┌─────────────────────────────────────────┐
                  │  LandmarkSource (one interface)          │
                  │  WebcamSource  │  ReplaySource (.npz)    │
                  └────────────────┬────────────────────────┘
                                   │ (landmarks|None, handedness, dt)
                                   ▼
                  ┌─────────────────────────────────────────┐
                  │  LandmarkSmoother   One Euro + dropout   │
                  │                     hold                 │
                  └────────────────┬────────────────────────┘
                                   ▼
                  ┌─────────────────────────────────────────┐
                  │  canonicalize()  →  (canonical, frame)   │
                  │  translate · scale · rotate · mirror     │
                  └──────┬──────────────────────┬───────────┘
                         │                      │
              canonical  │                      │  frame.basis
              landmarks  │                      │  (global rotation,
                         │                      │   deliberately factored out
                         ▼                      ▼   then read back)
        ┌────────────────────────┐   ┌──────────────────────────┐
        │  Models                 │   │  ControlMapper            │
        │  PoseMLP    (frame)     │   │  raw_signals →            │
        │  GestureTCN (window)    │   │  SignalRange.normalize →   │
        │        ↓                │   │  gamma → One Euro          │
        │  OnsetSpotter           │   └───────────┬──────────────┘
        │  (hysteresis FSM)       │               │
        └────────────┬───────────┘               │
                     │ GestureEvent              │ {grow, bloom, sway}
                     │ (discrete)                │ (continuous, [0,1])
                     └───────────┬───────────────┘
                                 ▼
                  ┌─────────────────────────────────────────┐
                  │  build_spiderlily(grow, bloom)           │
                  │  arc-length integration of unit tangent  │
                  └────────────────┬────────────────────────┘
                                   ▼
                  ┌─────────────────────────────────────────┐
                  │  ribbonize → build_batch  (numpy)        │  ← test seam
                  ├─────────────────────────────────────────┤
                  │  BloomRenderer (moderngl)                │
                  │  scene → bright/blur ×2 → ACES composite │
                  └─────────────────────────────────────────┘
```

## Module boundaries

| Module | Depends on | Notes |
|---|---|---|
| `landmarks/canonical` | numpy | Pure functions. No state, no I/O. |
| `landmarks/filters` | numpy | Stateful, but state is explicit and resettable. |
| `landmarks/source` | mediapipe, opencv *(optional)* | Replay path is numpy-only. |
| `control/*` | numpy | **Shared across projects.** Emits named scalars. |
| `models/spotter` | numpy | Deliberately torch-free so it is trivially testable. |
| `models/{pose_mlp,temporal}` | torch *(optional)* | Import raises a helpful message. |
| `geometry/spiderlily` | numpy | Pure. Fully property-tested. |
| `render/window` | moderngl *(optional)* | `build_batch` is the numpy test seam. |
| `data/*` | numpy | `dataset` imports torch lazily, inside a function. |

### Two boundaries worth explaining

**`build_batch` is the render test seam.** Everything above the GL boundary is
pure numpy and fully tested headlessly — vertex packing, UV ranges, the
ribbon-pinch fallback. Only actual GL calls are untested, which is the correct
place to stop: testing GL requires a display, and a display requirement in CI
means the tests don't run.

**`OnsetSpotter` has no torch dependency.** The state machine is the piece you
tune most often and the piece most likely to harbour subtle bugs, so it takes
plain probability arrays. Consequence: it's tested against synthetic traces
covering each failure mode, and `gesturebloom tune` grid-searches hundreds of
configurations in seconds with no model loaded.

## Why the dependencies are optional

Base install is `numpy` + `pyyaml`. A visitor who wants to read the geometry or
run the tests shouldn't download a 2 GB torch wheel, and someone deploying
inference shouldn't need mediapipe.

Each optional import raises a message naming the extra to install:

```python
raise ImportError(
    "PyTorch is required for gesturebloom.models. "
    "Install with: pip install 'gesturebloom[train]'"
) from exc
```

This is also what lets CI run the full suite on a bare runner.

## Coordinate systems

Three, and confusing them is the most likely source of a subtle bug.

1. **Image-normalized** — MediaPipe output. `x, y ∈ [0,1]`, `z` is relative depth
   in roughly the same scale as `x`. Origin top-left. Handedness labels refer to
   the *un-mirrored* image, so a mirrored preview reverses them.
2. **Canonical hand** — wrist at origin, middle-MCP at `(0,1,0)`, palm normal
   along `+z`, all distances in hand-span units. Model inputs and control
   measures live here.
3. **Flower** — `+y` up the stem, origin at the receptacle, tepal length ≈ 1.
   Independent of hand space; the mapping between them is only the scalar
   parameters, which is what keeps the geometry reusable.

`HandFrame` stores the exact transform between (1) and (2) and inverts it to
machine precision, so canonical geometry can be pushed back into image space for
compositing over the camera feed.

## Design decisions

**Analytic invariance over learned invariance.** Canonicalization removes
rotation, scale, translation, and chirality in closed form rather than relying on
augmentation to teach them. The model gets smaller, the data requirement drops
sharply, and the invariance is *exact* rather than approximate — and testable as
a property.

**Causal-only temporal modelling.** No bidirectional layers, no future context,
no non-causal padding anywhere. This costs some accuracy versus an offline model
and buys the guarantee that reported metrics describe deployed behaviour. The
`test_online_matches_offline` test asserts it.

**Regression for continuous, classification for discrete, and never confuse
them.** `grow`/`bloom` are regressions. Gesture identity is classification. A
system that classifies `bloom` into buckets steps visibly; one that regresses
gesture identity produces meaningless interpolations between gestures.

**NaN for missing data, never zero.** A zeroed dropped frame is
indistinguishable from a real hand at the origin and yields a plausible wrong
answer. NaN propagates loudly and surfaces at the first arithmetic operation.

**Config typos raise.** A silently-ignored misspelled key costs an hour of
concluding a parameter has no effect. `load_config` rejects unknown keys at every
level.

**Generated tables, not typed ones.** The README latency table comes from
`make bench`. Hand-maintained performance numbers go stale immediately and are
then quietly wrong, which is worse than absent.

## Extending to a second project

`control/` was factored to be reusable. To bind the same hand signals to a
different renderer, supply new `ControlBinding`s and consume the output dict:

```python
from gesturebloom.control.mapper import ControlBinding, ControlMapper

bindings = {
    "ink_offset":   ControlBinding(signal="pinch",    gamma=1.4),
    "screen_angle": ControlBinding(signal="roll",     gamma=1.0),
    "stipple_rate": ControlBinding(signal="openness", gamma=0.8),
}
mapper = ControlMapper(ranges=profile.ranges, bindings=bindings)
params = mapper.update(canonical, dt=dt, basis=frame.basis)
```

Calibration profiles, filtering, and the recording format transfer unchanged.
The abstraction sits at "named normalized scalars" precisely so this works
without touching `control/` at all.
