# Agent Instructions for ML Cookbook Project

These instructions define how AI coding assistants should behave in this repository.

## Project Context

- Stack: Python, PyTorch, Lightning, uv.
- Dependency management: use `uv` with `pyproject.toml` and `uv.lock`.
- Training entrypoint: `train.py` and `src/ml_cookbook/train.py`.
- Config source of truth: `configs/default.yaml`.
- Baseline tests: `tests/test_shapes.py` and `tests/test_train_smoke.py`.

## Global Behavior

- Prefer small, incremental changes over broad rewrites.
- Preserve existing architecture and naming unless explicitly asked to refactor.
- Keep code and docs bilingual-friendly (French + English) when editing user-facing docs.
- Keep comments short and useful; avoid obvious comments.
- Do not introduce heavy dependencies unless clearly justified.

## Python and Tooling Rules

- Use `uv` commands in examples and scripts:
  - `uv sync --group dev`
  - `uv run pytest`
  - `uv run python train.py --config configs/default.yaml`
- Do not reintroduce `requirements.txt` as primary dependency source.
- Keep code compatible with Python 3.11+.
- Prefer type hints for public functions and module boundaries.

## ML / Lightning Best Practices

- Preserve reproducibility:
  - Respect seed handling in `src/ml_cookbook/utils/repro.py`.
  - Keep deterministic options configurable from YAML.
- Keep data/model contract explicit:
  - Inputs: batch-first tensors.
  - Targets: class indices (`torch.long`) for classification.
- For model changes, keep logging of train/val/test loss and accuracy unless task says otherwise.
- Keep callbacks sensible by default (checkpoint, early stopping, lr monitor).

## Testing and Quality Gates

- Any change affecting data pipeline, model forward, or train flow should include/adjust tests.
- Run and keep passing at minimum:
  - `make lint`
  - `make test`
- Prefer adding fast smoke/sanity checks instead of long integration tests.

## CI and Automation

- Keep CI workflow in `.github/workflows/ci.yml` aligned with local commands.
- If adding a new mandatory check locally, add it to CI too.
- Ensure CI remains fast and deterministic.

## Documentation Expectations

- Update docs when behavior, commands, or structure changes.
- Keep `README.md` concise and onboarding-focused.
- Keep `cookbook.md` as the detailed practical guide.

## What to Avoid

- Large unrelated formatting changes.
- Silent behavior changes without config updates.
- Hardcoding machine-specific paths.
- Breaking current command surface (`make sync`, `make test`, `make train`, `make lint`).

## Available Subagents

This project provides specialized subagents in `.opencode/agents/`:

| Agent | Purpose |
|---|---|
| `code-review` | Strict PR review: correctness, regressions, data/model contracts, reproducibility, test gaps. Read-only. |
| `docs-writer` | Documentation updates for README/cookbook/setup in French + English. |
| `test-triage` | Diagnose failing tests (pytest, CI, smoke training). Root cause + minimal fix. |
| `experiment-setup` | Create/adjust experiment configs, trainer/model/data parameters for reproducible ML runs. |
| `dependency-upgrade` | Plan and apply safe dependency upgrades, refresh `uv.lock`, reduce upgrade risk. |
| `performance-profiler` | Profile slow training/inference, find bottlenecks, propose measured optimizations. |
