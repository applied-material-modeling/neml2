@insert-title:tutorials-contributing

[TOC]

## Model development

Although NEML2 comes with a large collection modular building blocks for composing material models, it is sometimes necessary to write your own material models (and integrate them with existing NEML2 models). The [extension](#tutorials-extension) tutorial set demonstrates how a custom model can be implemented within the NEML2 framework and provides in-depth explanation for every step throughout the development process.

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

A model unit test examines the outputs of a `Model` given a predefined set of inputs. Model unit tests can be directly designed using the input file syntax with the `ModelUnitTest` type. A variety of checks can be turned on and off based on input file options. To list a few: `check_first_derivatives` compares the implemented first order derivatives of the model against finite-differencing results, and the test is marked as passing only if the two derivatives are within tolerances specified with `derivative_abs_tol` and `derivative_rel_tol`; if `check_cuda` is set to `true`, all checks are repeated a second time on GPU (if available).

All input files for model unit tests should be stored inside `tests/unit/models`. Every input file with the `.i` extension will be automatically discovered and executed. To run all the model unit tests, use the following commands
```
cd tests
../build/dev/unit/unit_tests models
```

To run a specific model unit test, use the `-c` command line option followed by the relative location of the input file, i.e.
```
cd tests
../build/dev/unit/unit_tests models -c solid_mechanics/LinearIsotropicElasticity.i
```

### Regression tests {#testing-regression-tests}

A model regression test runs a `Model` using a user specified driver. The results are compared against a predefined reference (stored on the disk checked into the repository). The test passes only if the current results are the same as the predefined reference (again within specified tolerances). The regression tests ensure the consistency of implementations across commits. Currently, `TransientRegression` is the only supported type of regression test.

Each input file for model regression tests should be stored inside a separate folder inside `tests/regression`. Every input file with the `.i` extension will be automatically discovered and executed. To run all the model regression tests, use the `regression_tests` executable followed by the physics module, i.e.
```
cd tests
../build/dev/regression/regression_tests "solid mechanics"
```
To run a specific model regression test, use the `-c` command line option followed by the relative location of the input file, i.e.
```
cd tests
../build/dev/regression/regression_tests "solid mechanics" -c viscoplasticity/chaboche/model.i
```
Note that the regression test expects an option `reference` which specifies the relative location to the reference solution.

### Verification tests {#testing-verification-tests}

The model verification test is similar to the model regression test in terms of workflow. The difference is the a verification test defines the reference solution using NEML, the predecessor of NEML2. Since NEML was developed with strict software assurance, the verification tests ensure that the migration from NEML to NEML2 does not cause any regression in software quality.

Each input file for model verification tests should be stored inside a separate folder inside `tests/verification`. Every input file with the `.i` extension will be automatically discovered and executed. To run all the model verification tests, use the `verification_tests` executable followed by the physics module, i.e.
```
cd tests
../build/dev/verification/verification_tests "solid mechanics"
```

To run a specific model verification test, use the `-c` command line option followed by the relative location of the input file, i.e.
```
cd tests
../build/dev/verification/verification_tests "solid mechanics" -c chaboche/chaboche.i
```
The regression test compares variables (specified using the `variables` option) against reference values (specified using the `references` option). The reference variables can be read using input objects with type `VTestTimeSeries`.

## Python package

### Setup {#testing-python}

A collection of tests are available under `python/tests` to ensure the NEML2 Python package is working correctly. For Visual Studio Code users, the [Python](https://github.com/Microsoft/vscode-python) extension can be used to automatically discover and run tests. In the extension settings, the "Pytest Enabled" variable shall be set to true. In addition, "pytestArgs" shall provide the location of tests, i.e. "${workspaceFolder}/python/tests". The `settings.json` file shall contain the following entries:
```json
{
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": [
    "${workspaceFolder}/python/tests"
  ],
}
```

If the Python bindings are built (with `NEML2_PYBIND` set to `ON`) but are not installed to the site-packages directory (i.e. during development), pytest will not be able to import the %neml2 package unless the environment variable `PYTHONPATH` is modified according to the specified build directory. For Visual Studio Code users, create a `.env` file in the repository's root and include an entry `PYTHONPATH=build/dev/python` (assuming the build directory is `build/dev` which is the default from CMake presets), and the Python extension will be able to import the NEML2 Python package.

### pytest {#testing-pytest}

The Python tests use the [pytest](https://docs.pytest.org/en/stable/index.html) framework. To run tests using commandline, invoke `pytest` with the correct `PYTHONPATH`, i.e.

```
PYTHONPATH=build/dev/python pytest python/tests
```

To run a specific test case, use

```
PYTHONPATH=build/dev/python pytest "python/tests/test_Model.py::test_forward"
```
which runs the function named `test_forward` defined in the `python/tests/test_Model.py` file.

@insert-page-navigation

## Documentation

It is of paramount importance to write documentation as the library is being developed. While NEML2 supports both Doxygen-style in-code documentation mechanisms and runtime syntax documentation mechanisms, it is still sometimes necessary to write standalone, self-contained documentation.

To this end, the "dev" configure preset and the "dev-doc" build preset (see [build customization](@ref build-customization)) can be used to generate and render the documentation locally:
```
cmake --preset dev -S .
cmake --build --preset dev-doc
```
Once the documentation is built, the site can be previewed locally in any browser that supports static HTML, i.e.
```
firefox build/dev/doc/build/html/index.html
```

## Code formatting and linting

The C++ source code is formatted using `clang-format`. A `.clang-format` file is provided at the repository root specifying the formatting requirements. When using an IDE providing plugins or extensions to format C++ source code, it is important to
1. Point the plugin/extension to use the `.clang-format` file located at NEML2's repository root.
2. Associate file extensions `.h` and `.cxx` with C++.

The Python scripts shall be formatted using `black`. Formatting requirements are specified under the `[black]` section in `pyproject.toml`.

All pull requests will be run through `clang-format` and `black` to ensure formatting consistency.

For C++ linting, a `.clang-tidy` file is provided at the repository root to specify expected checks. Python linting is not currently enforced.

## Naming conventions

### Reserved axis names

Recall that NEML2 models operates on _labeled tensors_, and that the collection of labels (with their corresponding layout) is called an labeled axis ([LabeledAxis](@ref neml2::LabeledAxis)). NEML2 predefines 6 sub-axes to categorize all the input, output and intermediate variables:
- State \f$\mathcal{S}\f$ (axis name `state`): Variables collectively characterizing the current _state_ of the material subject to given external forces. The state variables are usually the output of a physically meaningful material model.
- Forces \f$\mathcal{F}\f$ (axis name `forces`): Variables defining the _external_ forces that drive the response of the material.
- Old state \f$\mathcal{S}_n\f$ (axis name `old_state`): The state variables _prior to_ the current material update. In the time-discrete setting, these are the state variables from the previous time step.
- Old forces \f$\mathcal{F}_n\f$ (axis name `old_forces`): The external forces _prior to_ the current material update. In the time-discrete setting, these are the forces from the previous time step.
- Residual \f$\mathcal{R}\f$ (axis name `residual`): The residual defines an _implicit_ model/function. An implicit model is updated by solving for the state variables that result in zero residual.
- Parameters \f$\mathcal{P}\f$ (axis name `parameters`): The (nonlinear) parameters.

\note
When authoring C++ source code, it is recommended to avoid hard-coding reserved axis names as pure strings. Instead, inlined `const` string names (defined in `neml2/models/LabeledAxisAccessor.h`) shall be used wherever possible, they are `STATE`, `OLD_STATE`, `FORCES`, `OLD_FORCES`, `RESIDUAL`, and `PARAMETERS` whose names are self-explanatory.

### Variable naming conventions

Variable names are used to _access_ slices of the storage tensor. Variable names have the type neml2::VariableName which is an alias to neml2::LabeledAxisAccessor. The following characters are not allowed in variable names:
- whitespace characters: input file parsing ambiguity
- `,`: input file parsing ambiguity
- `;`: input file parsing ambiguity
- `.`: clash with PyTorch parameter/buffer naming convention
- `/`: separator reserved for nested variable name

In the input file, the separator `/` is used to denote nested variable names. For example, `A/B/foo` specifies a variable named "foo" defined on the sub-axis named "B" which is a nested sub-axis of "A".

### Source code naming conventions

In NEML2 source code, the following naming conventions are recommended:
- User-facing variables and option names should be _as descriptive as possible_. For example, the equivalent plastic strain is named "equivalent_plastic_strain". Note that white spaces, quotes, and left slashes are not allowed in the names. Underscores are recommended as an replacement for white spaces.
- Developer-facing variables and option names should use simple alphanumeric symbols. For example, the equivalent plastic strain is named "ep" in consistency with most of the existing literature.
- Developner-facing member variables and option names should use the same alphanumeric symbols. For example, the member variable for the equivalent plastic strain is named `ep`. However, if the member variable is protected or private, it is recommended to prefix it with an underscore, i.e. `_ep`.
- Struct names and class names should use `PascalCase`.
- Function names should use `snake_case`.
