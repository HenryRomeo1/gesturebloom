"""moderngl renderer: strands -> screen.

Kept deliberately thin. The interesting work in this project is the ML and the
geometry, and a renderer that tries to be a framework obscures both. This is one
shader pair, one dynamic vertex buffer, and a two-pass bloom.

**Why a dynamic VBO rather than instancing or a compute shader.** The whole
flower is about 1,700 vertices. Re-uploading that every frame costs tens of
microseconds -- far below the landmark inference cost, and utterly invisible in the
latency budget. Optimizing it would be optimizing the wrong stage. If you scale
to a field of hundreds of flowers, move :func:`~gesturebloom.geometry.spiderlily.ribbonize`
into a geometry shader; until then, clarity wins.

**Why the glow is a real two-pass blur.** Additive-blended wide lines are the
cheap approximation and they look like additive-blended wide lines. A separable
Gaussian on a half-resolution bright-pass costs about 0.3 ms at 1080p and is the
single largest visual-quality difference in the project.

Optional dependency. ``pip install 'gesturebloom[render]'``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..geometry.spiderlily import Strand, ribbonize

SHADER_DIR = Path(__file__).parent / "shaders"


@dataclass
class RenderConfig:
    width: int = 1280
    height: int = 720
    bloom_passes: int = 2
    bloom_strength: float = 0.85
    bloom_threshold: float = 0.55
    background: tuple[float, float, float] = (0.02, 0.01, 0.04)
    tepal_color: tuple[float, float, float] = (0.96, 0.14, 0.22)
    stamen_color: tuple[float, float, float] = (1.0, 0.82, 0.35)
    camera_distance: float = 3.4
    fov_degrees: float = 42.0
    camera_dim: float = 0.62
    """How much to dim the camera feed. Below ~0.7 the flower reads clearly on
    top; at 1.0 it competes with a bright, busy video background and loses."""
    anchor_y_offset: float = -0.18
    """Pushes the flower slightly below the wrist so it appears to grow *out of*
    the hand rather than through it."""


def _perspective(fov_deg: float, aspect: float, near: float = 0.1, far: float = 100.0) -> np.ndarray:
    f = 1.0 / np.tan(np.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= max(float(np.linalg.norm(s)), 1e-8)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[:3, 3] = -m[:3, :3] @ eye
    return m


def build_batch(
    strands: list[Strand],
    view_dir: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack all strands into one interleaved vertex buffer with primitive restarts.

    Returns
    -------
    vertex_data:
        ``(N, 6)`` float32: ``x, y, z, u, v, role`` where ``role`` is 0 for tepal
        and 1 for stamen -- passed as a vertex attribute so a single draw call
        handles both, with the fragment shader picking the colour.
    counts:
        ``(n_strands,)`` int32 vertex count per strand, for
        ``glMultiDrawArrays``-style batched triangle-strip drawing.

    Examples
    --------
    >>> from gesturebloom.geometry.spiderlily import build_spiderlily
    >>> data, counts = build_batch(build_spiderlily(1.0, 0.7))
    >>> data.shape[1], len(counts)
    (6, 12)
    >>> int(counts.sum()) == data.shape[0]
    True
    """
    vd = np.array([0.0, 0.0, 1.0], dtype=np.float32) if view_dir is None else view_dir
    chunks: list[np.ndarray] = []
    counts: list[int] = []
    for s in strands:
        verts, uvs = ribbonize(s, vd)
        if verts.shape[0] == 0:
            continue
        role = np.full((verts.shape[0], 1), 1.0 if s.role == "stamen" else 0.0, dtype=np.float32)
        chunks.append(np.hstack([verts, uvs, role]).astype(np.float32))
        counts.append(verts.shape[0])
    if not chunks:
        return np.zeros((0, 6), dtype=np.float32), np.zeros(0, dtype=np.int32)
    return np.vstack(chunks), np.array(counts, dtype=np.int32)


class BloomRenderer:
    """Owns the GL context, programs, and framebuffers.

    Not constructed in tests -- :func:`build_batch` is the seam, and it is pure
    numpy. Everything above the GL boundary is verified headlessly.
    """

    def __init__(
        self,
        config: RenderConfig | None = None,
        standalone: bool = False,
        ctx=None,
    ) -> None:
        try:
            import moderngl
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "moderngl is required for rendering. Install with: "
                "pip install 'gesturebloom[render]'"
            ) from exc

        self.config = config or RenderConfig()
        # Reuse the window's context when one is supplied. Creating a second
        # context in the same thread yields a black screen with no error.
        if ctx is not None:
            self.ctx = ctx
        elif standalone:
            self.ctx = moderngl.create_standalone_context(require=330)
        else:
            self.ctx = moderngl.create_context(require=330)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.strand_program = self._load_program("strand")
        self.blur_program = self._load_program("blur")
        self.composite_program = self._load_program("composite")
        self.background_program = self._load_program("background")
        self._camera_tex = None
        # 1x1 black stand-in, bound when there is no camera frame. Sampling an
        # unbound texture unit is undefined behaviour and shows up as garbage or
        # a driver crash, so always bind something valid.
        self._empty_tex = self.ctx.texture((1, 1), 3, data=b"\x00\x00\x00")

        self._vbo = self.ctx.buffer(reserve=4 * 6 * 8192, dynamic=True)
        self._vao = self.ctx.vertex_array(
            self.strand_program, [(self._vbo, "3f 2f 1f", "in_position", "in_uv", "in_role")]
        )
        self._make_targets()
        self._quad = self._make_fullscreen_quad()
        # Built once. Creating a VAO per frame leaks GL objects until the driver
        # stalls -- it presents as a slow framerate decay over several minutes,
        # which is a miserable bug to track down after the fact.
        self._composite_vao = self.ctx.vertex_array(
            self.composite_program, [(self._quad[1], "2f 2f", "in_position", "in_uv")]
        )
        self._background_vao = self.ctx.vertex_array(
            self.background_program, [(self._quad[1], "2f 2f", "in_position", "in_uv")]
        )

    def _load_program(self, name: str):
        vert = (SHADER_DIR / f"{name}.vert").read_text(encoding="utf-8")
        frag = (SHADER_DIR / f"{name}.frag").read_text(encoding="utf-8")
        return self.ctx.program(vertex_shader=vert, fragment_shader=frag)

    def _make_targets(self) -> None:
        c = self.config
        self.scene_tex = self.ctx.texture((c.width, c.height), 4, dtype="f2")
        self.scene_fbo = self.ctx.framebuffer(color_attachments=[self.scene_tex])
        hw, hh = max(c.width // 2, 1), max(c.height // 2, 1)
        self.ping_tex = self.ctx.texture((hw, hh), 4, dtype="f2")
        self.pong_tex = self.ctx.texture((hw, hh), 4, dtype="f2")
        for t in (self.scene_tex, self.ping_tex, self.pong_tex):
            t.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self.ping_fbo = self.ctx.framebuffer(color_attachments=[self.ping_tex])
        self.pong_fbo = self.ctx.framebuffer(color_attachments=[self.pong_tex])

    def _make_fullscreen_quad(self):
        quad = np.array(
            [-1, -1, 0, 0, 1, -1, 1, 0, -1, 1, 0, 1, 1, 1, 1, 1], dtype=np.float32
        )
        vbo = self.ctx.buffer(quad.tobytes())
        return self.ctx.vertex_array(self.blur_program, [(vbo, "2f 2f", "in_position", "in_uv")]), vbo

    def _upload_camera(self, frame: np.ndarray):
        """Upload an RGB frame, (re)allocating the texture if the size changed."""
        h, w = frame.shape[:2]
        if self._camera_tex is None or self._camera_tex.size != (w, h):
            if self._camera_tex is not None:
                self._camera_tex.release()
            self._camera_tex = self.ctx.texture((w, h), 3, dtype="f1")
            self._camera_tex.filter = (self.ctx.LINEAR, self.ctx.LINEAR)
        self._camera_tex.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
        return self._camera_tex

    def _mvp(self, spin: float = 0.0) -> np.ndarray:
        c = self.config
        eye = np.array(
            [
                np.sin(spin) * c.camera_distance,
                0.9,
                np.cos(spin) * c.camera_distance,
            ],
            dtype=np.float32,
        )
        view = _look_at(eye, np.array([0.0, 0.85, 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32))
        proj = _perspective(c.fov_degrees, c.width / c.height)
        return (proj @ view).astype(np.float32)

    def draw(
        self,
        strands: list[Strand],
        params: dict[str, float],
        spin: float = 0.0,
        background: np.ndarray | None = None,
        anchor: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        """Render one frame. Call inside your windowing library's draw callback.

        Parameters
        ----------
        background:
            Optional ``(H, W, 3)`` uint8 RGB camera frame, already annotated with
            the skeleton overlay. Drawn opaque behind the flower.
        anchor:
            Screen-space NDC offset for the flower, so it can track the hand.
        """
        import moderngl

        data, counts = build_batch(strands)
        c = self.config

        # Flower renders into a TRANSPARENT buffer. This is the important part:
        # if the camera feed were in this buffer, the bloom bright-pass would
        # glow the whole video and the result looks like a smeared mess.
        self.scene_fbo.use()
        self.ctx.clear(0.0, 0.0, 0.0, 0.0)
        if data.shape[0]:
            needed = data.nbytes
            if needed > self._vbo.size:
                self._vbo.orphan(needed * 2)
            self._vbo.write(data.tobytes())
            self.strand_program["u_mvp"].write(self._mvp(spin).T.tobytes())
            self.strand_program["u_anchor"].value = (float(anchor[0]), float(anchor[1]))
            self.strand_program["u_tepal_color"].value = c.tepal_color
            self.strand_program["u_stamen_color"].value = c.stamen_color
            self.strand_program["u_bloom"].value = float(params.get("bloom", 0.0))
            offset = 0
            for n in counts.tolist():
                self._vao.render(moderngl.TRIANGLE_STRIP, vertices=n, first=offset)
                offset += n

        # Separable Gaussian on a half-res bright pass.
        vao, _ = self._quad
        src = self.scene_tex
        for i in range(max(c.bloom_passes, 1)):
            for horizontal in (True, False):
                fbo = self.ping_fbo if horizontal else self.pong_fbo
                fbo.use()
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                src.use(0)
                self.blur_program["u_source"].value = 0
                self.blur_program["u_horizontal"].value = horizontal
                self.blur_program["u_threshold"].value = c.bloom_threshold if i == 0 else 0.0
                self.blur_program["u_texel"].value = (1.0 / src.width, 1.0 / src.height)
                vao.render(moderngl.TRIANGLE_STRIP)
                src = self.ping_tex if horizontal else self.pong_tex

        self.ctx.screen.use()
        self.ctx.clear(*c.background, 1.0)

        cam_tex = (
            self._upload_camera(background) if background is not None else self._empty_tex
        )

        self.scene_tex.use(0)
        self.pong_tex.use(1)
        cam_tex.use(2)
        self.composite_program["u_scene"].value = 0
        self.composite_program["u_bloom_tex"].value = 1
        self.composite_program["u_background"].value = 2
        self.composite_program["u_strength"].value = c.bloom_strength
        self.composite_program["u_has_background"].value = (
            c.camera_dim if background is not None else 0.0
        )
        self._composite_vao.render(moderngl.TRIANGLE_STRIP)

    def release(self) -> None:
        for obj in (self.scene_fbo, self.ping_fbo, self.pong_fbo, self._vbo, self._empty_tex):
            obj.release()
        if self._camera_tex is not None:
            self._camera_tex.release()
