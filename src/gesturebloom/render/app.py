"""Windowed application loop.

The piece that inverts control. Everywhere else in this package *you* drive the
loop by pulling frames from a :class:`~gesturebloom.landmarks.source.LandmarkSource`.
A windowing library drives the loop itself and calls you back, so this module
adapts between the two: it holds the frame iterator and pulls exactly one frame
per rendered frame.

That coupling is deliberate. The alternative -- a capture thread feeding a queue --
sounds better and is worse: it decouples capture rate from render rate, which
means a slow render silently accumulates a queue of stale landmarks and you get
input lag that grows over time and disappears when you profile it. Pulling one
frame per render keeps latency bounded and observable. If capture becomes the
bottleneck, you see it as dropped frame rate, which is honest.

Requires the ``render`` extra: ``pip install 'gesturebloom[render]'``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def run_windowed(
    source,
    smoother,
    mapper,
    render_config,
    flower_seed: int = 7,
    record_frames: Path | None = None,
    max_frames: int = 0,
    vsync: bool = True,
) -> int:
    """Open a window and drive the flower until the source ends or the user quits.

    Parameters
    ----------
    source:
        Any :class:`~gesturebloom.landmarks.source.LandmarkSource` -- webcam or
        replay. The window does not know or care which.
    smoother:
        A :class:`~gesturebloom.landmarks.filters.LandmarkSmoother`.
    mapper:
        A :class:`~gesturebloom.control.mapper.ControlMapper`.
    render_config:
        A :class:`~gesturebloom.render.window.RenderConfig`.
    record_frames:
        If given, dump each rendered frame as a PNG into this directory. Convert
        to a GIF afterwards with ffmpeg -- see the README. Recording slows the
        render considerably, so never benchmark with it on.
    max_frames:
        Stop after this many frames. ``0`` means run until the source ends.

    Keys
    ----
    ``ESC`` / ``Q``  quit
    ``SPACE``        pause the landmark stream (freezes the flower, keeps rendering)
    ``R``            reset the filters and the mapper
    ``S``            save a single PNG screenshot
    """
    try:
        import moderngl_window as mglw
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "moderngl-window is required for the windowed app. "
            "Install with: pip install 'gesturebloom[render]'"
        ) from exc

    from ..geometry.spiderlily import build_spiderlily
    from ..landmarks.canonical import try_canonicalize
    from .window import BloomRenderer

    if record_frames is not None:
        record_frames = Path(record_frames)
        record_frames.mkdir(parents=True, exist_ok=True)

    class BloomApp(mglw.WindowConfig):
        gl_version = (3, 3)
        title = "GestureBloom"
        window_size = (render_config.width, render_config.height)
        aspect_ratio = None  # let the window resize freely
        resizable = True

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            # Pass the window's existing context rather than creating a second
            # one -- two contexts in one thread is a class of bug that manifests
            # as a black screen with no error.
            self.renderer = BloomRenderer(render_config, ctx=self.ctx)
            self.frames = source.frames()
            self.params = {"grow": 0.0, "bloom": 0.0, "sway": 0.5}
            self.n_frames = 0
            self.n_tracked = 0
            self.paused = False
            self.finished = False
            print("Running. ESC/Q quit | SPACE pause | R reset | S screenshot")

        # ---- frame pump ---------------------------------------------------- #
        def _pull(self) -> None:
            try:
                lm, hand, dt = next(self.frames)
            except StopIteration:
                self.finished = True
                return

            self.n_frames += 1
            smoothed = smoother.update(lm, dt)
            if smoothed is None:
                return
            result = try_canonicalize(smoothed, hand)
            if result is None:
                return
            canonical, hand_frame = result
            self.n_tracked += 1
            self.params = mapper.update(canonical, dt=dt, basis=hand_frame.basis)

        # ---- render callback ----------------------------------------------- #
        def on_render(self, time: float, frametime: float) -> None:
            if not self.paused:
                self._pull()

            strands = build_spiderlily(
                self.params["grow"], self.params["bloom"], seed=flower_seed
            )
            # sway maps [0,1] -> a gentle +/- 0.6 rad orbit
            self.renderer.draw(strands, self.params, spin=self.params.get("sway", 0.5) * 1.2 - 0.6)

            if record_frames is not None:
                self._dump_frame()

            if self.finished or (max_frames and self.n_frames >= max_frames):
                self.wnd.close()

        # moderngl-window renamed callbacks in 2.4.5. Aliasing both means the
        # app works across versions instead of failing with a silent no-op
        # render, which is a genuinely confusing way to lose an hour.
        render = on_render

        def _dump_frame(self) -> None:
            try:
                from PIL import Image
            except ImportError:
                return
            w, h = self.wnd.buffer_size
            raw = self.wnd.fbo.read(components=3, alignment=1)
            img = Image.frombytes("RGB", (w, h), raw).transpose(Image.FLIP_TOP_BOTTOM)
            img.save(record_frames / f"frame_{self.n_frames:05d}.png")

        # ---- input --------------------------------------------------------- #
        def on_key_event(self, key, action, modifiers) -> None:
            keys = self.wnd.keys
            if action != keys.ACTION_PRESS:
                return
            if key in (keys.ESCAPE, keys.Q):
                self.wnd.close()
            elif key == keys.SPACE:
                self.paused = not self.paused
                print(f"{'paused' if self.paused else 'resumed'}")
            elif key == keys.R:
                smoother.reset()
                mapper.reset()
                print("filters reset")
            elif key == keys.S:
                out = Path(f"screenshot_{self.n_frames:05d}.png")
                try:
                    from PIL import Image

                    w, h = self.wnd.buffer_size
                    raw = self.wnd.fbo.read(components=3, alignment=1)
                    Image.frombytes("RGB", (w, h), raw).transpose(
                        Image.FLIP_TOP_BOTTOM
                    ).save(out)
                    print(f"saved {out}")
                except ImportError:
                    print("pillow not installed; cannot save screenshots")

        key_event = on_key_event  # version compatibility

    try:
        mglw.run_window_config(BloomApp, args=["--vsync", "1" if vsync else "0"])
    except SystemExit:
        pass
    finally:
        source.close()

    return 0


def probe_gl() -> int:
    """Print GL/driver info and confirm a context can be created.

    Run this first when the window is black or refuses to open. It separates
    "my GL setup is broken" from "my code is broken", which are very different
    afternoons.
    """
    try:
        import moderngl
    except ImportError:
        print("moderngl not installed. pip install 'gesturebloom[render]'")
        return 1

    try:
        ctx = moderngl.create_standalone_context(require=330)
    except Exception as exc:
        print(f"FAILED to create a GL 3.3 context: {exc}")
        print("\nLikely causes:")
        print("  - headless machine with no GPU driver (try a local desktop)")
        print("  - remote/SSH session without X forwarding")
        print("  - very old integrated GPU without GL 3.3 support")
        return 1

    print(f"GL version : {ctx.info['GL_VERSION']}")
    print(f"Renderer   : {ctx.info['GL_RENDERER']}")
    print(f"Vendor     : {ctx.info['GL_VENDOR']}")
    print(f"GLSL       : {ctx.info['GL_SHADING_LANGUAGE_VERSION']}")
    print(f"Max texture: {ctx.info['GL_MAX_TEXTURE_SIZE']}")

    # Compile the real shaders. Driver-specific GLSL rejections are common and
    # this is where you want to find out, not inside the render loop.
    from .window import SHADER_DIR

    for name in ("strand", "blur", "composite"):
        try:
            ctx.program(
                vertex_shader=(SHADER_DIR / f"{name}.vert").read_text(encoding="utf-8"),
                fragment_shader=(SHADER_DIR / f"{name}.frag").read_text(encoding="utf-8"),
            )
            print(f"shader {name:10s} OK")
        except Exception as exc:
            print(f"shader {name:10s} FAILED: {exc}")
            return 1

    # Exercise the numpy->GL seam once with real flower data.
    from ..geometry.spiderlily import build_spiderlily
    from .window import build_batch

    data, counts = build_batch(build_spiderlily(1.0, 0.7))
    print(f"vertex batch: {data.shape[0]} verts across {len(counts)} strands")
    assert np.all(np.isfinite(data))
    print("\nGL setup looks good.")
    return 0
