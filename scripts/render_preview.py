#!/usr/bin/env python3
"""Render a bloom progression to SVG, with no dependencies beyond numpy.

Why SVG and not a screenshot: this runs anywhere, including CI and headless
machines, so the README always has a visual even before the GL renderer is
working. It also renders the *actual geometry module* rather than a mock, so if
the parametric form regresses, the preview visibly changes.

GitHub renders static SVG in markdown, so the output drops straight into the
README.

Usage
-----
    python scripts/render_preview.py --out assets/geometry_preview.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gesturebloom.geometry.plant import build_plant
from gesturebloom.geometry.spiderlily import ribbonize

# Matches the shader palette in render/shaders/strand.frag
TEPAL = "#f5243a"
TEPAL_TIP = "#ff9c8a"
STAMEN = "#ffd15a"
STEM = "#4df25c"
BACKGROUND = "#07040d"
ROLE_FILL = {"tepal": TEPAL, "stamen": STAMEN, "stem": STEM}


def project(points: np.ndarray, tilt: float = 0.42) -> tuple[np.ndarray, np.ndarray]:
    """Orthographic projection with a slight elevation.

    Returns ``(screen_xy, depth)``. Depth is returned separately so strands can
    be sorted back-to-front -- without that, front tepals get drawn behind rear
    ones and the flower reads as a flat tangle.
    """
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    cos_t, sin_t = np.cos(tilt), np.sin(tilt)
    up = y * cos_t + z * sin_t
    depth = -y * sin_t + z * cos_t
    return np.stack([x, up], axis=1), depth


def strand_polygon(strand, scale: float, cx: float, cy: float) -> str:
    """Build a closed SVG polygon from a ribbon: left edge out, right edge back."""
    verts, _ = ribbonize(strand, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    if verts.shape[0] < 4:
        return ""
    xy, _ = project(verts)
    left, right = xy[0::2], xy[1::2]
    outline = np.vstack([left, right[::-1]])
    pts = " ".join(f"{cx + p[0] * scale:.2f},{cy - p[1] * scale:.2f}" for p in outline)
    return pts


def render_flower(grow: float, bloom: float, cx: float, cy: float, scale: float, seed: int) -> list[str]:
    """Emit SVG elements for the whole plant, depth-sorted back to front."""
    strands = build_plant(grow, bloom, seed=seed)

    ordered = []
    for strand in strands:
        _, depth = project(strand.points)
        ordered.append((float(np.mean(depth)), strand))
    ordered.sort(key=lambda item: item[0])  # far strands first

    out: list[str] = []
    for mean_depth, strand in ordered:
        pts = strand_polygon(strand, scale, cx, cy)
        if not pts:
            continue
        fill = ROLE_FILL.get(strand.role, TEPAL)
        opacity = 0.95
        if strand.role == "tepal":
            # Rear tepals sit slightly darker, which reads as depth without
            # needing real lighting.
            fill = TEPAL_TIP if mean_depth > 0 else TEPAL
            opacity = 0.92 if mean_depth > 0 else 0.78
        glow = 1.0 + 0.6 * bloom
        out.append(
            f'    <polygon points="{pts}" fill="{fill}" fill-opacity="{opacity:.2f}" '
            f'stroke="{fill}" stroke-width="{0.6 * glow:.2f}" stroke-opacity="0.5"/>'
        )
    return out


def build_svg(stages: int = 5, seed: int = 7, panel_w: int = 210, panel_h: int = 290) -> str:
    width = panel_w * stages
    scale = panel_w * 0.24
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {panel_h}" '
        f'width="{width}" height="{panel_h}" role="img" '
        f'aria-label="Branching spiderlily plant, grow and bloom from 0 to 1">',
        f'  <rect width="{width}" height="{panel_h}" fill="{BACKGROUND}"/>',
    ]

    for i in range(stages):
        t = i / max(stages - 1, 1)
        grow = 0.25 + 0.75 * t
        bloom = t
        cx = panel_w * (i + 0.5)
        cy = panel_h * 0.80
        parts.append(f'  <g id="stage-{i}">')
        parts.extend(render_flower(grow, bloom, cx, cy, scale, seed))
        parts.append(
            f'    <text x="{cx:.1f}" y="{panel_h - 14}" fill="#8b8296" '
            f'font-family="ui-monospace,SFMono-Regular,Menlo,monospace" font-size="11" '
            f'text-anchor="middle">grow {grow:.2f}  bloom {bloom:.2f}</text>'
        )
        parts.append("  </g>")

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("assets/geometry_preview.svg"))
    ap.add_argument("--stages", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    svg = build_svg(stages=args.stages, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg, encoding="utf-8")
    print(f"Wrote {args.out} ({len(svg) / 1024:.1f} KB, {args.stages} stages)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
