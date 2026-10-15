# scene-to-score

[![CI](https://github.com/ademasles/scene-to-score/actions/workflows/ci.yml/badge.svg)](https://github.com/ademasles/scene-to-score/actions/workflows/ci.yml)

Prédire la musique d'une scène de film dans l'espace latent, puis la générer.

Projet multimodal (vidéo vers musique) du cours de deep learning avancé. Les données viennent
d'OSSL-v2 (Kim et al., 2026) : 34 343 extraits de 1 812 films du domaine public, avec leur
musique séparée du dialogue, sous licence CC BY 4.0.

Le plan de travail est dans [docs/plan.md](docs/plan.md).

## Démarrage

Prérequis : Linux, macOS sur puce Apple ou WSL2, avec git, [uv](https://docs.astral.sh/uv/) et
[just](https://just.systems) (voir [Installation des outils](#installation-des-outils)). Un GPU
NVIDIA accélère les calculs ; 8 Go suffisent pour développer.

```bash
just sync        # crée .venv depuis uv.lock
just gpu         # vérifie que PyTorch voit le GPU
just check       # ruff + basedpyright + pytest
just download    # index OSSL-v2, 20 films d'entraînement et 5 de test (environ 3,4 Go)
uv run pre-commit install   # une fois après le clone
```

Tout télécharger représente 180 Go, à réserver au tech lab :

```bash
just download data.n_train_movies=null data.n_test_movies=null
```

## Installation des outils

Une seule fois par machine, sans droits administrateur. Python n'a pas besoin d'être installé :
uv télécharge la version fixée dans `.python-version`.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # uv, macOS et Linux
uv tool install rust-just                         # just, distribué par uv
```

Sur macOS avec Homebrew, `brew install uv just` fait la même chose. Ouvrir ensuite un nouveau
terminal et vérifier avec `uv --version` et `just --version`.

Limites selon la machine :

- PyTorch n'est publié pour macOS que sur puce Apple (M1 et suivantes), sous macOS 14 ou plus
  récent. Sur un Mac Intel, travailler sur une machine Linux ou au tech lab.
- Sur Mac, il n'y a pas de CUDA : le code tourne sur le processeur. Les gros calculs se font au
  tech lab.
- Sous Linux avec un GPU NVIDIA, il faut un pilote compatible CUDA 13 ; `just gpu` le vérifie.

`just` n'est qu'un raccourci : chaque recette du `justfile` est une commande `uv run` que l'on
peut taper directement, par exemple `uv run pytest`.

## Organisation

| Chemin | Contenu |
|---|---|
| `src/scene_to_score/data/` | Index OSSL-v2 (`ossl.py`) et téléchargement (`download.py`) |
| `src/scene_to_score/seed.py` | Graines et choix du périphérique |
| `configs/` | Configuration Hydra : chemins, données |
| `scripts/` | Scripts ponctuels (`gpu_check.py`) |
| `notebooks/` | Exploration ; `just kernel` enregistre l'environnement dans Jupyter |
| `tests/` | pytest |
| `docs/` | Plan du projet |
| `data/`, `outputs/` | Données et sorties d'expériences, ignorées par git |

## Données

- Les données vont dans `data/`. Sur une autre machine, `export STS_DATA_DIR=/chemin/vers/data`.
- La version du jeu est figée dans `configs/data/ossl_v2.yaml` (révision du 13 août 2026).
- Les identifiants d'extraits ont la forme `<film>_<index>_<début>_<fin>`, avec le début et la
  fin en centièmes de seconde.
- Le découpage entraînement/test est fait par film : aucun film n'apparaît des deux côtés.
- Hugging Face limite le débit des téléchargements anonymes ; `hf auth login` lève cette limite.
  Le jeton reste hors du dépôt.

## Conventions

- Dépendances : `uv add <paquet>`, outils de développement : `uv add --dev <paquet>`. Jamais
  `pip install`.
- PyTorch vient de l'index CUDA déclaré dans `pyproject.toml`.
- Toute expérience est seedée et configurée par Hydra ; les surcharges passent en ligne de
  commande, par exemple `just download data.n_train_movies=50`.
- Ni données, ni points de contrôle, ni secrets dans git ; `gitleaks` tourne en pre-commit.
- Les hooks pre-commit tiennent aussi `uv.lock` à jour (`uv-lock`) et vident les sorties des
  notebooks (`nbstripout`).
- L'intégration continue relance `just check` et tous les hooks à chaque push sur `main` et à
  chaque pull request.
- Code, commentaires et messages de commit en anglais, au format des commits conventionnels.

## Licences

Le code est sous licence MIT (voir [LICENSE](LICENSE)).

Les données OSSL-v2 sont distribuées sous CC BY 4.0. Citer Kim, Novack, McAuley et Dong (2026),
*Dialogue-Aware Video-to-Music Generation Using Public Domain Film Collections*,
arXiv:2608.11576.
