(contributing-tests)=
# Testing

## Running the suite

The test suite is `pytest`-driven and lives under `tests/`. Run the
full suite:

```shell
pytest -v tests/
```

Run a single file directly:

```shell
pytest -v tests/unit/test_factory.py
```

The `tests/` tree has five top-level buckets — `unit/`, `models/`,
`regression/`, `verification/`, and `aoti/`. The
[pytest usage guide](https://docs.pytest.org/en/stable/how-to/usage.html)
documents the rich selector syntax (`-k expr`, `--lf`, parametrized
test IDs, marker filters, …).

The AOTI compile suite under `tests/aoti/` triggers an Inductor
compile per scenario (noticeably slower than the rest of the suite)
and runs by default; skip it with `pytest --ignore=tests/aoti tests/`
when you want the fast subset for an inner edit loop.

VS Code users can drive the suite through the
[Python extension](https://github.com/Microsoft/vscode-python); set
`python.testing.pytestEnabled` to `true` and point `pytestArgs` at
`${workspaceFolder}/tests`.

## Coverage

Python branch + line coverage is measured by `coverage.py` via the
`pytest-cov` plugin (both pinned in the `[dev]` extras). Configuration
lives in `pyproject.toml` under `[tool.coverage.run]` /
`[tool.coverage.report]`; the package source is `neml2` and the runner
uses branch coverage on top of line coverage.

Local workflow:

```shell
pytest --cov tests/unit tests/models tests/aoti          # terminal summary
pytest --cov --cov-report=html tests/unit tests/models tests/aoti  # → htmlcov/index.html
pytest -n auto --cov tests/unit tests/models tests/aoti  # parallel (xdist-safe)
```

`tests/regression/` and `tests/verification/` are deliberately omitted
— their end-to-end runs duplicate coverage the unit and model suites
already provide while multiplying the run time. The `coverage` CI job in
`.github/workflows/python.yaml` runs the same subset and uploads the
raw report as a workflow artifact.
