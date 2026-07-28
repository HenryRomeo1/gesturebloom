.PHONY: install test lint fmt bench preview demo clean

install:
	pip install -e '.[dev]'

install-all:
	pip install -e '.[all,dev]'

test:
	pytest tests/ -v

cov:
	pytest tests/ --cov=gesturebloom --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/gesturebloom

fmt:
	ruff format src/ tests/
	ruff check --fix src/ tests/

# Regenerates the latency table in the README. Do not hand-edit that table.
bench:
	gesturebloom bench --frames 1200 --title "Pipeline latency" --out docs/latency.md

# Regenerates the README preview image from the geometry module.
preview:
	python scripts/render_preview.py --out assets/geometry_preview.svg

# Full pipeline, no camera, no GPU -- the 30-second first-run experience.
demo:
	python scripts/make_demo_recording.py --out data/examples/demo.npz
	gesturebloom run --replay data/examples/demo.npz --headless --max-frames 300

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
