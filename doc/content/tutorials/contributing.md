@insert-title:tutorials-contributing

[TOC]

## Model development

Although NEML2 comes with a large collection of modular building blocks for composing material models, it is sometimes necessary to write your own material models (and integrate them with existing NEML2 models). The [extension](#tutorials-extension) tutorial set demonstrates how a custom model can be implemented within the NEML2 framework and provides an in-depth explanation for each step in the development process.

## C++ backend {#testing-cpp}

By default when `NEML2_TESTS` is set to `ON`, three test suites are built under the specified build directory:

- `tests/unit/unit_tests`: Collection of tests to ensure individual objects are working correctly.
- `tests/regression/regression_tests`: Collection of tests to avoid regression.
- `tests/verification/verification_tests`: Collection of verification problems.

For Visual Studio Code users, the [C++ TestMate](https://github.com/matepek/vscode-catch2-test-adapter) extension can be used to automatically discover and run tests.

When `NEML2_WORK_DISPATCHER` is set to `ON`, an additional test suite is built:

- `test/dispatchers/dispatcher_tests`: Collection of unit tests for the work dispatcher.

### Catch tests {#testing-catch-tests}

A Catch test refers to a test directly written in C++ source code within the Catch2 framework. It offers the highest level of flexibility, but requires more effort to set up. To understand how a Catch2 test works, please refer to the [official Catch2 documentation](https://github.com/catchorg/Catch2/blob/v2.x/docs/tutorial.md).

### Unit tests {#testing-unit-tests}

A model unit test examines the outputs of a `Model` given a predefined set of inputs. Model unit tests can be directly designed using the input file syntax with the `ModelUnitTest` type. A variety of checks can be turned on and off based on input file options. To list a few: `check_first_derivatives` compares the implemented first order derivatives of the model against finite-differencing results, and the test is marked as passing only if the two derivatives are within tolerances specified with `derivative_abs_tol` and `derivative_rel_tol`.

All input files for model unit tests should be stored inside `tests/unit/models`. Every input file with the `.i` extension will be automatically discovered and executed. To run all the model unit tests, use the following commands
```
./build/dev/unit/unit_tests models
```

To run a specific model unit test, use the `-c` command line option followed by the relative location of the input file, i.e.
```
./build/dev/unit/unit_tests models -c solid_mechanics/LinearIsotropicElasticity.i
```

### Regression tests {#testing-regression-tests}

A model regression test runs a `Model` using a user specified driver. The results are compared against a predefined reference (stored on the disk checked into the repository). The test passes only if the current results are the same as the predefined reference (again within specified tolerances). The regression tests ensure the consistency of implementations across commits. Currently, `TransientRegression` is the only supported type of regression test.

Each input file for model regression tests should be stored inside a separate folder inside `tests/regression`. Every input file with the `.i` extension will be automatically discovered and executed. To run all the model regression tests, use the `regression_tests` executable followed by the physics module, i.e.
```
./build/dev/regression/regression_tests "solid mechanics"
```
To run a specific model regression test, use the `-c` command line option followed by the relative location of the input file, i.e.
```
./build/dev/regression/regression_tests "solid mechanics" -c viscoplasticity/chaboche/model.i
```
Note that the regression test expects an option `reference` which specifies the relative location to the reference solution.

### Verification tests {#testing-verification-tests}

The model verification test is similar to the model regression test in terms of workflow. The difference is the a verification test defines the reference solution using NEML, the predecessor of NEML2. Since NEML was developed with strict software assurance, the verification tests ensure that the migration from NEML to NEML2 does not cause any regression in software quality.

Each input file for model verification tests should be stored inside a separate folder inside `tests/verification`. Every input file with the `.i` extension will be automatically discovered and executed. To run all the model verification tests, use the `verification_tests` executable followed by the physics module, i.e.
```
../build/dev/verification/verification_tests "solid mechanics"
```

To run a specific model verification test, use the `-c` command line option followed by the relative location of the input file, i.e.
```
./build/dev/verification/verification_tests "solid mechanics" -c chaboche/chaboche.i
```
The regression test compares variables (specified using the `variables` option) against reference values (specified using the `references` option). The reference variables can be read using input objects with type `VTestTimeSeries`.

### Command-line arguments

The above test suites, including unit tests, regression tests, verification tests, and dispatcher tests, support a variety of configuration options that can be controlled from the command line. See [Catch2 command line](https://github.com/catchorg/Catch2/blob/devel/docs/command-line.md) documentation for a list of useful command-line options. In addition, NEML2 test suites support the following options:
- `-p`, `--path`: working directory. This argument is optional and is only needed when the default guessing mechanism fails to locate the test input files.
- `-d`, `--devices`: list of additional (non-CPU) devices to test.

## Python package

### Setup {#testing-python}

A collection of tests are available under `python/tests` to ensure the NEML2 Python package is working correctly.

First install the package with development dependencies:
```shell
pip install ".[dev]" -v
```

For Visual Studio Code users, the [Python](https://github.com/Microsoft/vscode-python) extension can be used to automatically discover and run tests. In the extension settings, the "Pytest Enabled" variable shall be set to true. In addition, "pytestArgs" shall provide the location of tests, i.e. "${workspaceFolder}/python/tests". The `settings.json` file shall contain the following entries:
```json
{
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "${workspaceFolder}/python/tests"
  ],
}
```

### Python stubs {#testing-python-stubs}

NEML2's Python API is primarily exposed through compiled bindings. To make type information and function signatures visible to IDEs, static analyzers, and documentation tooling, stub files must be generated.

Use the `neml2-stub` command line tool:
```shell
neml2-stub
```

Generating stubs is recommended after changing Python bindings and before building documentation.

### pytest {#testing-pytest}

The Python tests use the [pytest](https://docs.pytest.org/en/stable/index.html) framework. To run all tests:

```
pytest -v python/tests
```

To run a specific test case, use

```
pytest -v "python/tests/test_Model.py::test_forward"
```
which runs the function named `test_forward` defined in the `python/tests/test_Model.py` file.

## Jupyter notebooks

NEML2 tracks notebooks with a paired-text workflow so notebook changes remain reviewable and reproducible.

1. Notebook files (`*.ipynb`) are deliberately marked as binary in `.gitattributes`, so they are not the primary code-review surface.
2. Each notebook is paired with a MyST markdown file via `jupytext` (`.jupytext.toml`), and the paired markdown is the review surface.
3. Developers should edit notebooks directly, not the paired markdown files.
4. After modifying a notebook, all code cells must be executed. The documentation build flow (`doc/scripts/examples.py`) checks for unexecuted cells and fails if any are found.
5. Install pre-commit hooks so notebook/markdown pairs are synced automatically on commit:
   ```shell
   pre-commit install
   ```
   The repository hook `jupytext-sync` keeps paired files in sync for notebook examples.

## Documentation

It is of paramount importance to write documentation as the library is being developed. While NEML2 supports both Doxygen-style in-code documentation mechanisms and runtime syntax documentation mechanisms, it is still sometimes necessary to write standalone, self-contained documentation.

The documentation workflow is driven by scripts in `doc/scripts`:
1. `doc/scripts/examples.py`: builds/runs tutorial examples and prepares tutorial outputs.
2. `doc/scripts/genhtml.py`: syncs content, extracts syntax pages, and generates HTML documentation.

A typical local workflow is:
```shell
pip install ".[dev]" -v
neml2-stub
./doc/scripts/examples.py --log-level INFO --cmake-configure-args="-GNinja"
./doc/scripts/genhtml.py --log-level INFO
```
The documentation scripts require a non-editable `neml2` installation in `site-packages`.

Once built, preview the site locally:
```shell
firefox build/doc/build/html/index.html
```

For live preview during editing:
```shell
./doc/scripts/genhtml.py --serve
```
This starts a local server (default `http://127.0.0.1:8000/`) and automatically rebuilds docs when tracked content changes.

## Code formatting and static analysis

The C++ source code is formatted using `clang-format`. A `.clang-format` file is provided at the repository root specifying the formatting requirements. When using an IDE providing plugins or extensions to format C++ source code, it is important to
1. Point the plugin/extension to use the `.clang-format` file located at NEML2's repository root.
2. Associate file extensions `.h` and `.cxx` with C++.

The Python scripts must be formatted using `black`. Formatting requirements are specified under the `[black]` section in `pyproject.toml`. All pull requests will be run through `clang-format` and `black` to ensure formatting consistency.

The "cc" preset (or the `CMAKE_EXPORT_COMPILE_COMMANDS` configure option) can be used to generate the compilation database `compile_commands.json`. A symbolic link to `compile_commands.json` will be created at the project root. The compilation database is needed by many static analysis tools. Visual Studio Code users are encouraged to use the [clangd](https://github.com/clangd/vscode-clangd) extension. For C++ linting, a `.clang-tidy` file is provided at the repository root to specify expected checks. Python linting is not currently enforced.
