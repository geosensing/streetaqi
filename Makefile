.PHONY: install lint format type-check docstrings dependencies dead-code test docs build ci clean

install:
	uv sync --all-groups --all-extras

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

type-check:
	uv run pyright

docstrings:
	uv run pydoclint src

dependencies:
	uv run deptry .

dead-code:
	uv run vulture

test:
	uv run pytest --cov=streetaqi --cov-report=term-missing

docs:
	uv run sphinx-build -W --keep-going -b html docs docs/_build/html

build:
	uv build
	uvx twine check dist/*

ci: lint type-check docstrings dependencies dead-code test docs build

clean:
	rm -rf .coverage .pytest_cache .ruff_cache build dist docs/_build htmlcov output
