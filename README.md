# ML Cookbook Project

[![CI](https://github.com/tuanmp/ml-cookbook-project/actions/workflows/ci.yml/badge.svg)](https://github.com/tuanmp/ml-cookbook-project/actions/workflows/ci.yml)

Repository / Depot: [https://github.com/tuanmp/ml-cookbook-project](https://github.com/tuanmp/ml-cookbook-project)

Template de prototypage ML avec PyTorch + Lightning, base sur `uv`.
ML prototyping template with PyTorch + Lightning, powered by `uv`.

## Objectif / Goal

Ce projet sert de socle pour:
This project is a starter kit to:

1. valider rapidement une boucle d entrainement avec donnees factices,
1. quickly validate a training loop with dummy data,
1. iterer sur l architecture du modele,
1. iterate on model architecture,
1. basculer vers des donnees reelles sans casser la plomberie.
1. switch to real data without breaking the training plumbing.

### CAST: Cross-Attention Seed Transformer

Un modele de tracking de particules chargees base sur l extension de seeds par cross-attention.
A charged particle tracking model based on seed extension via cross-attention.

- Dataset: [ColliderML](https://huggingface.co/datasets/CERN/ColliderML-Release-1) (`ttbar_pu200`)
- Seeds: triplets de hits des couches internes du detecteur a pixels
- Seeds: triplets of hits from innermost pixel detector layers
- Modele: seeds (queries) attendent tous les hits (keys/values) via cross-attention
- Model: seeds (queries) attend to all hits (keys/values) via cross-attention
- Loss: InfoNCE multi-positif sur la matrice d assignation seed→hit
- Loss: multi-positive InfoNCE over seed→hit assignment matrix

```bash
uv run python cast_train.py --config configs/cast_default.yaml
```

## Quick Start

```bash
make sync
make test
make train
```

Equivalent sans Makefile:
Equivalent without Makefile:

```bash
uv sync --group dev
uv run pytest
uv run python train.py --config configs/default.yaml
```

## Structure Essentielle / Core Structure

```text
.
├── pyproject.toml
├── uv.lock
├── train.py                        # Dummy prototype entrypoint
├── cast_train.py                   # CAST seed-extension entrypoint
├── configs/
│   ├── default.yaml                # Dummy prototype config
│   └── cast_default.yaml           # CAST experiment config
├── src/ml_cookbook/
│   ├── train.py
│   ├── data/dummy_datamodule.py
│   ├── models/prototype_model.py
│   └── utils/repro.py
├── src/seed_extension/             # CAST package
│   ├── train.py                    # CAST training orchestration
│   ├── data/
│   │   ├── __init__.py             # Monkey-patch ColliderML features
│   │   └── dataset.py              # SeedExtensionDataset
│   └── models/cast/
│       ├── embedders.py            # FourierEncode, HitEmbedder, SeedEmbedder
│       ├── attention.py            # CrossAttentionDecoder, SeedSelfAttention
│       ├── encoders.py             # IdentityEncoder (pluggable)
│       ├── loss.py                 # Multi-positive InfoNCE loss
│       ├── metrics.py              # Binned efficiency/purity metrics
│       └── model.py                # CASTModel LightningModule
└── tests/
    ├── test_shapes.py
    ├── test_train_smoke.py
    ├── test_cast_shapes.py         # CAST component shape tests
    ├── test_cast_loss.py           # Loss function tests
    ├── test_cast_metrics.py        # Metrics tests
    ├── test_cast_dataset.py        # Dataset smoke test
    └── test_cast_smoke.py          # 1-epoch CAST training smoke test
```

## Documentation

- Guide utilisateur complet: `cookbook.md`
- User guide: `cookbook.md`
- Config experiment: `configs/default.yaml`
- Experiment config: `configs/default.yaml`
- MCP guide and boilerplate: `mcp-playbook.md`
- Guide MCP et boilerplate: `mcp-playbook.md`
- MCP Context7 example: `mcp-examples/context7.md`
- Exemple MCP Context7: `mcp-examples/context7.md`

## Copilot Behavior / Comportement Copilot

- Repository instructions file: `.github/copilot-instructions.md`
- Fichier d instructions du repository: `.github/copilot-instructions.md`
- Keep this file updated when project workflow, quality gates, or conventions change.
- Mettez ce fichier a jour lorsque le workflow, les controles qualite, ou les conventions changent.

## Copilot Agents / Agents Copilot

- Agent files folder: `.github/agents/`
- Dossier des agents: `.github/agents/`
- `Code Review`: PR review, bug/regression and risk analysis.
- `Docs Writer`: README/cookbook updates in French + English.
- `Test Triage`: diagnose failing tests and CI failures quickly.
- `Experiment Setup`: prepare reproducible experiment configs.
- `Dependency Upgrade Planner`: safe dependency upgrades with `uv`.
- `Performance Profiler`: bottleneck analysis and measurable speedups.

## Commandes Utiles / Useful Commands

- `make sync`: installe et verrouille l environnement via uv
- `make sync`: install and lock the environment via uv
- `make test`: lance la suite de tests
- `make test`: run the test suite
- `make train`: execute un entrainement complet avec la config par defaut
- `make train`: run full training with the default config
- `make lint`: verification syntaxique rapide
- `make lint`: quick syntax check
- `uv run python cast_train.py --config configs/cast_default.yaml`: CAST training on ColliderML
- `uv run pytest tests/test_cast_*.py -v`: CAST test suite
