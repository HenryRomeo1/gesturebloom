"""Config loading tests, with emphasis on failing loudly on typos."""

from __future__ import annotations

import pytest

from gesturebloom.config import AppConfig, load_config


def test_defaults() -> None:
    cfg = load_config(None)
    assert isinstance(cfg, AppConfig)
    assert cfg.smoothing.min_cutoff == 1.5
    assert cfg.spotter.exit_threshold < cfg.spotter.enter_threshold


def test_loads_shipped_default_yaml() -> None:
    from pathlib import Path

    path = Path(__file__).parent.parent / "configs" / "default.yaml"
    if not path.exists():
        pytest.skip("configs/default.yaml not present")
    cfg = load_config(path)
    assert cfg.train.window_length == 32


def test_partial_override(tmp_path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("flower_seed: 42\nsmoothing:\n  beta: 0.2\n")
    cfg = load_config(p)
    assert cfg.flower_seed == 42
    assert cfg.smoothing.beta == 0.2
    assert cfg.smoothing.min_cutoff == 1.5  # untouched default


def test_typo_in_section_key_raises(tmp_path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("smoothing:\n  min_cuttoff: 2.0\n")
    with pytest.raises(ValueError, match="min_cuttoff"):
        load_config(p)


def test_typo_in_top_level_key_raises(tmp_path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("flower_sed: 3\n")
    with pytest.raises(ValueError, match="flower_sed"):
        load_config(p)


def test_invalid_value_raises(tmp_path) -> None:
    p = tmp_path / "c.yaml"
    p.write_text("smoothing:\n  min_cutoff: -1.0\n")
    with pytest.raises(ValueError):
        load_config(p)
