(build-customization)=
# Building from source

:::{note}
End users should not need this page. The published PyPI wheels are
expected to cover the vast majority of use cases — see [Basic installation](install.md).
Build from source if you are contributing to NEML2 itself, debugging a
build flavor the wheels don't ship, or experimenting with a custom
LibTorch.
:::

## Quick start

```shell
git clone -b main https://github.com/applied-material-modeling/neml2.git
cd neml2
pip install -e ".[dev]" -v
```

This drives [scikit-build-core](https://scikit-build-core.readthedocs.io/)
through the `NEML2_WHEEL` CMake path and lays down an editable Python
install plus everything the dev workflow needs (pytest, pre-commit,
sphinx, …).

## CMake presets

For pure C++ development (the wheel build is invoked separately by
`pip`), `CMakePresets.json` defines two presets, used at different
stages and *not* interchangeable:

`dev` — the day-to-day build preset
: Debug build of the C++ library and pybind extensions. This is the
  only preset with an accompanying *build* preset of the same name, so
  it's the one to use for actually compiling sources:

  ```shell
  cmake --preset dev -S .
  cmake --build --preset dev
  ```

`cc` — a configure-only preset for tooling
: Configures with `CMAKE_EXPORT_COMPILE_COMMANDS=ON` and
  `NEML2_WHEEL=ON` so that the resulting `compile_commands.json`
  covers both `libneml2_aoti` sources and the pybind extension `.cxx`
  files. A `compile_commands.json` symlink is dropped into the repo
  root for clangd / clang-tidy / other static-analysis tools to pick
  up. There is no matching build preset — `cc` is not meant for
  compiling, just for indexing:

  ```shell
  cmake --preset cc -S .
  # No `cmake --build --preset cc` — use `dev` (or a plain
  # `cmake --build build/cc -j$(nproc)`) if you want to actually
  # compile from this build dir.
  ```

## Notable configure options

The presets are tuned for routine work; override them when you need to:

| Option         | Default      | Purpose                                                  |
| :------------- | :----------- | :------------------------------------------------------- |
| `NEML2_TOOLS`  | `ON`         | Build the CLI binaries (`neml2-run`, `neml2-inspect`, …). |
| `NEML2_MPI`    | `OFF`        | Link the dispatcher submodule against MPI. With it OFF the submodule is still built but the MPI-using classes throw at construction. |
| `NEML2_WHEEL`  | `OFF`        | Build the Python wheel. Set automatically by `pip install`. |

Override on the configure line:

```shell
cmake --preset dev -DNEML2_MPI=ON -S .
```

## Custom LibTorch

If you want to build NEML2 against a libtorch other than the one shipped
by the active Python environment's `torch` package, set `torch_ROOT`
before configuring:

```shell
cmake --preset dev -Dtorch_ROOT=/path/to/libtorch -S .
```

`TORCH_ROOT` (all-caps), `torch_ROOT` / `TORCH_ROOT` / `torch_DIR` /
`TORCH_DIR` env vars, and the active Python's `torch` site-packages are
also consulted in that order. See `cmake/Modules/Findtorch.cmake` for
the full discovery procedure.

## Running the test suite

See [](contributing-tests) in the contributing guide.
