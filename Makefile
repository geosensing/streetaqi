.PHONY: install lint test clean

install:
	uv sync --all-extras

lint:
	uv run ruff check src/
	uv run ruff format --check src/

format:
	uv run ruff check --fix src/
	uv run ruff format src/

test:
	uv run pytest

clean:
	rm -rf dist/ build/ *.egg-info/ .pytest_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
