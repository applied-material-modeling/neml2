(build-customization)=
# Building from source

:::{note}
End users should not need this page. The published PyPI wheels are
expected to cover the vast majority of use cases — see [Basic installation](../installation/install.md).
Build from source if you are contributing to NEML2 itself, debugging a
build flavor the wheels don't ship, or experimenting with a custom
LibTorch.
:::

## Quick start

```shell
git clone -b main https://github.com/applied-material-modeling/neml2.git
cd neml2

# Build + runtime prerequisites (skip any you already have).
pip install torch nmhit scikit-build-core cmake ninja

# Editable install.
pip install -e ".[dev]" -v --no-build-isolation
```

`--no-build-isolation` builds against the `torch` already in your
environment, so `torch` and `nmhit` must be installed first.

CMake and Ninja are also required. They are available from most system
package managers if you prefer not to use the pip builds above — for
example `apt install cmake ninja-build`, `brew install cmake ninja`, or
`conda install -c conda-forge cmake ninja`.

`pip install -e ".[dev]"` is the only build command you need; there are
no CMake presets to invoke. An editable build:

- compiles at `RelWithDebInfo` (optimized, with debug symbols);
- builds the C++ test executables (run them with `ctest`, below);
- generates `compile_commands.json` and symlinks it to the repo root for
  clangd / clang-tidy;
- drops a stable `build/editable` symlink to the build tree.

## Running the C++ tests

```shell
ctest --test-dir build/editable -L dispatcher    # dispatcher / scheduler tests
ctest --test-dir build/editable -L eager         # embedded-Python eager test
ctest --test-dir build/editable -L benchmark     # benchmark smoke tests
```

The Python suite runs under `pytest` — see [](contributing-tests).

## Build-type and other overrides

Pass `--config-settings` to override a default for one install. A full
Debug build:

```shell
pip install -e ".[dev]" --no-build-isolation \
    --config-settings=cmake.build-type=Debug
```

A Coverage / ThreadSanitizer build (clang), then run the C++ tests:

```shell
CC=clang CXX=clang++ pip install -e ".[dev]" --no-build-isolation \
    --config-settings=cmake.build-type=Coverage
ctest --test-dir build/editable -L "dispatcher|eager"
```

Build against a libtorch other than the active environment's `torch`:

```shell
pip install -e ".[dev]" --no-build-isolation \
    --config-settings=cmake.define.torch_ROOT=/path/to/libtorch
```

## Iterating on the C++ runtime

While editing the C++ sources, rebuild the editable build tree directly
for a fast incremental compile (no `pip` round-trip):

```shell
cmake --build build/editable
```

## Windows (best-effort)

Windows is supported on a **best-effort** basis, from a source build only —
no wheels are published and it is not a primary platform. CI does exercise it
(compile, install, the Python test suite including AOTI, and a
`find_package(neml2)` C++ consumer), but expect rough edges.

You need Visual Studio 2022 (MSVC) and Python. scikit-build-core drives the
Visual Studio generator, which locates MSVC on its own, so no developer shell
is required. From PowerShell:

```powershell
git clone -b main https://github.com/applied-material-modeling/neml2.git
cd neml2

pip install torch nmhit scikit-build-core
pip install ".[dev]" --no-build-isolation -v
```

- Build `Release` or `RelWithDebInfo` only: the pip `torch` wheels ship Release
  binaries on Windows, so a Debug build cannot link against them.
- The embedded-Python eager runtime (`cpp-eager`) is not built on Windows; the
  AOTI routes and the native Python API are unaffected.
- Write filesystem paths inside HIT (`.i`) input files with forward slashes —
  they work on every platform.
