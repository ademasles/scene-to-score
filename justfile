default:
    @just --list

# Create or update the virtual environment from uv.lock
sync:
    uv sync

test:
    uv run pytest

lint:
    uv run ruff check .
    uv run ruff format --check .

fmt:
    uv run ruff format .
    uv run ruff check --fix .

typecheck:
    uv run basedpyright

# All checks run before a commit or a PR
check: lint typecheck test

# GPU availability + micro-benchmark
gpu:
    uv run python scripts/gpu_check.py

# Register this project's venv as a Jupyter kernel
kernel:
    uv run python -m ipykernel install --user --name scene-to-score

# Download the OSSL-v2 index and a seeded subset of movies; Hydra overrides go after the
# recipe name, e.g. `just download data.n_train_movies=50`
download *ARGS:
    uv run python -m scene_to_score.data.download {{ARGS}}
