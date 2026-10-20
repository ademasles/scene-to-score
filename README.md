# scene-to-score

[![CI](https://github.com/ademasles/scene-to-score/actions/workflows/ci.yml/badge.svg)](https://github.com/ademasles/scene-to-score/actions/workflows/ci.yml)

Predict the music of a film scene in latent space, then generate it.

Multimodal (video to music) project of the Advanced Deep Learning course. The data comes from
OSSL-v2 (Kim et al., 2026): 34,343 clips from 1,812 public-domain films, with their music
separated from the dialogue, under a CC BY 4.0 licence.

The work plan is in [docs/plan.md](docs/plan.md).

## Getting started

Requirements: Linux, macOS on Apple silicon or WSL2, with git, [uv](https://docs.astral.sh/uv/)
and [just](https://just.systems) (see [Installing the tools](#installing-the-tools)). An NVIDIA
GPU speeds up compute; 8 GB is enough for development.

```bash
just sync        # create .venv from uv.lock
just gpu         # check that PyTorch sees the GPU
just check       # ruff + basedpyright + pytest
just download    # OSSL-v2 index, 20 training films and 5 test films (about 3.4 GB)
uv run pre-commit install   # once after cloning
```

Downloading everything takes 180 GB; keep that for the tech lab:

```bash
just download data.n_train_movies=null data.n_test_movies=null
```

## Installing the tools

Once per machine, no admin rights needed. Python does not need to be installed: uv downloads the
version pinned in `.python-version`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv, macOS and Linux
uv tool install rust-just                         # just, distributed through uv
```

On macOS with Homebrew, `brew install uv just` does the same. Then open a new terminal and check
with `uv --version` and `just --version`.

Limits depending on the machine:

- PyTorch is only published for macOS on Apple silicon (M1 and later), on macOS 14 or newer. On
  an Intel Mac, work on a Linux machine or on the tech lab.
- On a Mac there is no CUDA: code runs on the CPU. Heavy compute happens on the tech lab.
- On Linux with an NVIDIA GPU, you need a CUDA 13 compatible driver; `just gpu` checks it.

`just` is only a shortcut: each recipe in the `justfile` is a `uv run` command that can be typed
directly, for example `uv run pytest`.

## Layout

| Path | Contents |
|---|---|
| `src/scene_to_score/data/` | OSSL-v2 index (`ossl.py`) and download (`download.py`) |
| `src/scene_to_score/seed.py` | Seeding and device selection |
| `configs/` | Hydra configuration: paths, data |
| `scripts/` | One-off scripts (`gpu_check.py`) |
| `notebooks/` | Exploration; `just kernel` registers the environment with Jupyter |
| `tests/` | pytest |
| `docs/` | Project plan |
| `data/`, `outputs/` | Data and experiment outputs, ignored by git |

## Data

- Data goes to `data/`. On another machine, `export STS_DATA_DIR=/path/to/data`.
- The dataset version is pinned in `configs/data/ossl_v2.yaml` (revision of 13 August 2026).
- Clip ids look like `<film>_<index>_<start>_<end>`, with start and end in hundredths of a
  second.
- The train/test split is by film: no film appears on both sides.
- Hugging Face rate-limits anonymous downloads; `hf auth login` lifts that limit. The token stays
  out of the repository.

## Conventions

- Dependencies: `uv add <package>`, dev tools: `uv add --dev <package>`. Never `pip install`.
- PyTorch comes from the CUDA index declared in `pyproject.toml`.
- Every experiment is seeded and configured through Hydra; overrides go on the command line, for
  example `just download data.n_train_movies=50`.
- No data, checkpoints or secrets in git; `gitleaks` runs as a pre-commit hook.
- The pre-commit hooks also keep `uv.lock` up to date (`uv-lock`) and strip notebook outputs
  (`nbstripout`).
- Continuous integration reruns `just check` and every hook on each push to `main` and on each
  pull request.
- Code, comments and commit messages in English, following Conventional Commits.

## Licences

The code is released under the MIT licence (see [LICENSE](LICENSE)).

The OSSL-v2 data is distributed under CC BY 4.0. Cite Kim, Novack, McAuley and Dong (2026),
*Dialogue-Aware Video-to-Music Generation Using Public Domain Film Collections*,
arXiv:2608.11576.
