# Contributing to wakesmith

Contributions that improve detection quality, language coverage, reproducibility or
documentation are welcome. Please discuss substantial changes in an issue first.

## Development setup

wakesmith uses [uv](https://docs.astral.sh/uv/) and requires Linux on x86-64.

```bash
git clone https://github.com/yannikheizmann/wakesmith.git
cd wakesmith
uv sync --frozen --extra dev --extra docs
```

Training needs an NVIDIA GPU; the tests do not.

## Code conventions

- Each package exposes its public API from `__init__.py`; implementation modules are private
  (`_module.py`). Import across packages only through public APIs.
- Settings live in typed argument groups under `wakesmith.config.args`, composed by the root
  `Args`. A new setting changes the arguments' digest, so existing runs are never silently
  reused.
- Every stage must stay resumable: write outputs atomically and keep results a pure function
  of the arguments and seed.
- Features must remain identical to openWakeWord's streaming runtime (see
  `tests/features/test_features.py`).
- Use British English in prose and documentation, imperative docstring summaries for
  functions and noun phrases for classes. Avoid comments that restate the code.

## Validation

```bash
uv run --frozen ruff format --check .
uv run --frozen ruff check .
uv run --frozen pytest -q --cov=wakesmith
uv run --frozen mkdocs build --strict
uv build
```

Changes to synthesis, screening, features, training or evaluation should report the effect
on a short run (see "A faster trial run" in the documentation) in the pull request.

## Pull requests

Explain the problem and the chosen solution, keep changes focused, add tests and
documentation, and make sure continuous integration passes. By contributing you agree that
your contribution is licensed under the [Apache License 2.0](LICENSE).
