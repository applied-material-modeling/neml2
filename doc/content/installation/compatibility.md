# Torch compatibility {#torch-compat}

[TOC]

## Overview

PyTorch's C++ ABI is not guaranteed to be stable across minor releases, so a
NEML2 wheel from PyPI is only verified to load and run correctly with the
torch versions listed below. Each combination is exercised on every PR by
[`.github/workflows/compat.yml`](https://github.com/applied-material-modeling/neml2/blob/main/.github/workflows/compat.yml),
which installs the wheel alongside the listed `torch == <version>` and runs
the Python test suite end-to-end. A row appearing here means CI is green for
that combination on the current `main`.

The current supported range is

<!-- dependencies: torch.version_min -->
<!-- dependencies: torch.version_max -->
**`torch >= 2.10.0, <= 2.12.0`**

and is enforced as a `pyproject.toml` constraint, so `pip install neml2`
refuses a torch outside this range. Older torch versions either have an
incompatible C++ ABI or have been dropped from the seed because their
wheels are more than a year old; newer torch versions will be added to the
matrix once a maintainer validates them.

Combinations outside this list may still work — for example, a freshly
released torch the maintainers haven't validated yet — but are not
regression-tested. If you need an additional combination, please [open an
issue](https://github.com/applied-material-modeling/neml2/issues) and we will
add it to the matrix.

## Supported combinations

<!-- BEGIN_COMPAT_MATRIX -- regenerate with: python scripts/compat_matrix.py render --in-place FILE -->
| Torch  | Python | OS    |
| ------ | ------ | ----- |
| 2.10.0 | 3.10   | macOS |
| 2.10.0 | 3.10   | linux |
| 2.10.0 | 3.11   | macOS |
| 2.10.0 | 3.11   | linux |
| 2.10.0 | 3.12   | macOS |
| 2.10.0 | 3.12   | linux |
| 2.10.0 | 3.13   | macOS |
| 2.10.0 | 3.13   | linux |
| 2.10.0 | 3.14   | macOS |
| 2.10.0 | 3.14   | linux |
| 2.11.0 | 3.10   | macOS |
| 2.11.0 | 3.10   | linux |
| 2.11.0 | 3.11   | macOS |
| 2.11.0 | 3.11   | linux |
| 2.11.0 | 3.12   | macOS |
| 2.11.0 | 3.12   | linux |
| 2.11.0 | 3.13   | macOS |
| 2.11.0 | 3.13   | linux |
| 2.11.0 | 3.14   | macOS |
| 2.11.0 | 3.14   | linux |
| 2.12.0 | 3.10   | macOS |
| 2.12.0 | 3.10   | linux |
| 2.12.0 | 3.11   | macOS |
| 2.12.0 | 3.11   | linux |
| 2.12.0 | 3.12   | macOS |
| 2.12.0 | 3.12   | linux |
| 2.12.0 | 3.13   | macOS |
| 2.12.0 | 3.13   | linux |
| 2.12.0 | 3.14   | macOS |
| 2.12.0 | 3.14   | linux |
<!-- END_COMPAT_MATRIX -->

The matrix is generated from `compatibility.yaml` at the repository root.

## How the matrix evolves

- When a new torch release is published, a maintainer reruns
  `python scripts/compat_matrix.py seed` to widen the seed, opens a PR, and
  CI reports which new rows pass.
- Rows that fail CI are dropped from `compatibility.yaml` in the same PR with
  a one-line commit message recording the reason (e.g., "torch 2.13.0 drops
  `c10::TypeMeta::print_typename`").
- Wheels older than one year are dropped at seed time — torch makes no ABI
  promises that far back and most pre-built wheels stop installing on
  current Python/manylinux anyway.
- If your environment combines NEML2 with a torch version not in the table,
  testing locally is a one-liner:

  ```shell
  pip install torch==<version>
  pip install neml2
  python -c "import neml2"
  ```

  An `ImportError` mentioning unresolved torch symbols is the canonical
  signature of an ABI mismatch.
