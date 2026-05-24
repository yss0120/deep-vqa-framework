# Contributing to deep-vqa-framework

Thank you for your interest! Please follow these guidelines.

---

## 1. Environment Setup
We use `uv` for dependency management. Please ensure you have `uv` installed, then run:

```bash
make setup
uv sync
```

---

## 2. Coding Standards

We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting. Before committing your code, please run:

```bash
make validate
make fmt
```

---

## 3. Development Workflow

* **Branching**: Please use descriptive branch names (e.g., `feature/add-new-backbone` or `fix/gpu-oom`).
* **Commits**: Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification.
* **Testing**: Before submitting a PR, ensure `make test` (the smoke test) passes without errors.

---

## 4. Dependencies

* Core: `[project.dependencies]`
* Dev: `[project.optional-dependencies]`
* Run `uv sync` after changes

---

## 5. Reporting Issues

When reporting a bug, please include:

* The error message and stack trace.
* Your environment details (OS, GPU, PyTorch version).
* A minimal reproducible example.
