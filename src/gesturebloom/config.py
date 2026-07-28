"""Typed configuration with YAML loading.

Plain dataclasses rather than pydantic: the base install stays at numpy + pyyaml,
and for a config this shape the validation pydantic buys you is a few lines of
``__post_init__``. Reach for pydantic when configs get nested and
externally-supplied; not here.

Unknown keys raise rather than being ignored. A typo'd key that silently does
nothing is one of the most expensive classes of bug in a tuning-heavy project --
you spend an hour concluding a parameter has no effect.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class SmoothingConfig:
    min_cutoff: float = 1.5
    beta: float = 0.05
    d_cutoff: float = 1.0
    hold_frames: int = 6

    def __post_init__(self) -> None:
        if self.min_cutoff <= 0:
            raise ValueError("min_cutoff must be positive")


@dataclass
class RenderSettings:
    width: int = 1280
    height: int = 720
    bloom_passes: int = 2
    bloom_strength: float = 0.85
    vsync: bool = True
    camera_dim: float = 0.62
    anchor_y_offset: float = -0.30


@dataclass
class SpotterSettings:
    enter_threshold: float = 0.65
    exit_threshold: float = 0.40
    min_frames: int = 3
    refractory_frames: int = 12


@dataclass
class TrainSettings:
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    window_length: int = 32
    window_stride: int = 4
    label_ramp: int = 3


@dataclass
class AppConfig:
    flower_seed: int = 7
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    render: RenderSettings = field(default_factory=RenderSettings)
    spotter: SpotterSettings = field(default_factory=SpotterSettings)
    train: TrainSettings = field(default_factory=TrainSettings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_SECTIONS = {
    "smoothing": SmoothingConfig,
    "render": RenderSettings,
    "spotter": SpotterSettings,
    "train": TrainSettings,
}


def _build(cls, payload: dict[str, Any]):
    valid = {f.name for f in fields(cls)}
    unknown = set(payload) - valid
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**payload)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load a YAML config, or return defaults if ``path`` is ``None``.

    Examples
    --------
    >>> cfg = load_config(None)
    >>> cfg.smoothing.min_cutoff
    1.5
    >>> cfg.render.width
    1280
    """
    if path is None:
        return AppConfig()

    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    sections = {}
    for key, cls in _SECTIONS.items():
        if key in payload:
            sections[key] = _build(cls, payload.pop(key) or {})

    valid_top = {f.name for f in fields(AppConfig)} - set(_SECTIONS)
    unknown = set(payload) - valid_top
    if unknown:
        raise ValueError(f"unknown top-level config keys: {sorted(unknown)}")
    return AppConfig(**payload, **sections)
