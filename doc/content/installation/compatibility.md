# Torch compatibility {#torch-compat}

[TOC]

## Overview

PyTorch's C++ ABI is not guaranteed to be stable across minor releases, so
a NEML2 wheel from PyPI is only guaranteed to load and run correctly with
the torch versions listed below. Every combination shown here is exercised
by NEML2's automated tests on each change.

## Supported range

The current regression-tested range is

<!-- dependencies: torch.version_min -->
<!-- dependencies: torch.version_max -->
**`torch >= 2.10.0, <= 2.12.0`**

The bounds are asymmetric on purpose:

- The **lower bound** is a hard `pyproject.toml` constraint — older torch
  has a known-incompatible C++ ABI, so `pip install neml2` refuses it
  outright rather than letting the install succeed and `import neml2`
  crash later.
- The **upper bound** is advisory. A newly-released torch may or may not
  work, and rather than block users on it `pip install neml2` accepts any
  newer torch. The upper bound documents which versions have actually been
  verified.

Combinations outside this list may still work — for example, a freshly
released torch that hasn't been verified yet — but are not regression-
tested. If you need an additional combination, please [open an
issue](https://github.com/applied-material-modeling/neml2/issues).

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

## Testing an untested combination

If your environment combines NEML2 with a torch version that is not in
the table above, checking compatibility locally is a one-liner:

```shell
pip install torch==<version>
pip install neml2
python -c "import neml2"
```

An `ImportError` mentioning unresolved torch symbols is the canonical
signature of an ABI mismatch.
