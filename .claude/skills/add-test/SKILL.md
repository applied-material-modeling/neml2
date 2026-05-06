---
name: add-test
description: Scaffold a Catch2 C++ unit test for an existing NEML2 header — utilities under `misc/` or `base/`, tensor classes, solvers, equation systems, drivers, dispatchers, anything that isn't a `Model` subclass. Use this skill whenever the user wants to add, scaffold, or write a unit test for a specific C++ class, header, or free function (e.g., "add a test for `string_utils.h`", "scaffold a unit test for `Newton`", "test the `AxisLayout` class", "I need a test file for the new `R3` constructor"), even when they don't explicitly say "Catch2" or "unit test". For new `Model` subclasses, use `/add-model` instead — those are tested via `.i` `ModelUnitTest` input files, not C++.
---

# add-test

Scaffold a new Catch2 C++ unit test under `tests/unit/`, following the conventions in `tests/unit/README.md`. CMake auto-discovers the new file via `file(GLOB_RECURSE)` so no CMake edits are needed.

## When to use

Use this skill when adding a unit test for any non-Model header, e.g.:
- A new utility under `include/neml2/misc/` or `include/neml2/base/`.
- A new solver / linear algebra / equation-system class.
- A new tensor class or free function under `include/neml2/tensors/`.
- A new driver, dispatcher, or scheduler.

Do **not** use this skill for:
- Constitutive `Model` subclasses → use `/add-model`. Model unit tests are `.i` input files driven by `ModelUnitTest`, not Catch2 C++ files.
- Regression or verification tests (under `tests/regression/` and `tests/verification/`) — those follow different patterns and have their own driver types.

## Required information from the user

Before writing the file, confirm:

1. **Header path** being tested, relative to `include/neml2/`. Example: `misc/string_utils.h`, `base/Registry.h`, `tensors/R2.h`.
2. **What to test.** Either a list of free functions, or a class with the methods/members worth covering. The user usually states this implicitly ("test split, trim, start_with, end_with from string_utils").

If the header doesn't exist yet, stop and surface that — the skill scaffolds tests *for* an existing header, not for a header you're about to write.

## Naming and placement (per `tests/unit/README.md`)

For header `include/neml2/<dir>/<Name>.h`, the test file is `tests/unit/<dir>/test_<Name>.cxx`:

| Header | Test file | TEST_CASE name | Tag |
|---|---|---|---|
| `include/neml2/misc/string_utils.h` | `tests/unit/misc/test_string_utils.cxx` | `"string_utils"` | `"[misc]"` |
| `include/neml2/base/Registry.h` | `tests/unit/base/test_Registry.cxx` | `"Registry"` | `"[base]"` |
| `include/neml2/solvers/Newton.h` | `tests/unit/solvers/test_Newton.cxx` | `"Newton"` | `"[solvers]"` |
| `include/neml2/tensors/R2.h` | `tests/unit/tensors/test_R2.cxx` | `"R2"` | `"[tensors]"` |
| `include/neml2/models/solid_mechanics/elasticity/Elasticity.h` | `tests/unit/models/solid_mechanics/elasticity/test_Elasticity.cxx` | `"Elasticity"` | `"[models/solid_mechanics/elasticity]"` |

The conventions exist so contributors can navigate from any header to its test (and vice versa) without grep, and so test selectors like `unit_tests "[base]"` map cleanly to directories. The rules:

- **One header, one test file.** If you find yourself wanting two TEST_CASEs in the same file, that's a sign two headers should be tested in two files.
- **The TEST_CASE name matches the header's basename** (no `.h`, no `test_` prefix). This is what selectors like `unit_tests "Registry"` match against.
- **The tag matches the file's directory relative to `tests/unit/`**, in `[…]`. For nested dirs, use `/` inside the brackets — e.g., `[models/solid_mechanics/elasticity]`. This lets `unit_tests "[base]"` run an entire directory.
- **Each top-level SECTION covers one function, method, member, or sub-class**, named after it. Sub-SECTIONs are good for overloads or for distinct cases of one function — they nest naturally and produce readable failure paths.

## Two binaries — pick the right one

CMake builds two unit-test executables from `tests/unit/`:

- `tensor_tests` — built from `tests/unit/tensors/*.cxx` (recursive) plus `main.cxx`.
- `unit_tests` — built from everything else under `tests/unit/`.

Both use `file(GLOB_RECURSE)`, so simply placing the new file under the right directory wires it in.

If the header is under `include/neml2/tensors/`, the test file goes under `tests/unit/tensors/` and ends up in `tensor_tests`. Otherwise it ends up in `unit_tests`.

## License header

Every C++ test file must begin with the project's standard MIT copyright header — enforced by `scripts/check_copyright.py`. Read the repo-root `LICENSE` file and reproduce its lines as `// `-prefixed comments at the top of the new file. Use the same year as `LICENSE` (currently `2024`); don't bump it on new files. Mirror the exact format of any existing `tests/unit/**/test_*.cxx` if in doubt.

## Templates

### Template A — generic unit test (goes under `unit_tests`)

For headers anywhere except `include/neml2/tensors/`. File path: `tests/unit/<dir>/test_<Name>.cxx`.

```cpp
// <project MIT header — see "License header" section above>

#include <catch2/catch_test_macros.hpp>
// Add only when needed:
// #include <catch2/matchers/catch_matchers_all.hpp>

#include "neml2/<dir>/<Name>.h"

using namespace neml2;

TEST_CASE("<Name>", "[<dir>]")
{
  SECTION("<function_or_method_1>")
  {
    REQUIRE(/* … */);
  }

  SECTION("<function_or_method_2>")
  {
    SECTION("<overload or case A>")
    {
      REQUIRE(/* … */);
    }

    SECTION("<overload or case B>")
    {
      REQUIRE(/* … */);
    }
  }
}
```

A real example to mirror: `tests/unit/misc/test_string_utils.cxx`.

### Template B — tensor unit test (goes under `tensor_tests`)

For headers under `include/neml2/tensors/`. File path: `tests/unit/tensors/<…>/test_<Name>.cxx`.

```cpp
// <project MIT header — see "License header" section above>

#include <catch2/catch_test_macros.hpp>

#include "neml2/misc/defaults.h"
#include "utils.h"   // tests/unit/tensors/utils.h — provides test::allclose, etc.

#include "neml2/tensors/<Name>.h"

using namespace neml2;

TEST_CASE("<Name>", "[tensors]")
{
  at::manual_seed(42);

  SECTION("<constructor or method>")
  {
    <Name> a(/* … */, default_tensor_options());
    auto b = <Name>::full(/* … */, default_tensor_options());
    REQUIRE_THAT(a, test::allclose(b));
  }
}
```

Notes:
- `default_tensor_options()` from `neml2/misc/defaults.h` ensures the test honors `--devices` (so `--devices cuda` exercises the CUDA path).
- `test::allclose` (from `utils.h`) is the standard tensor comparison matcher — use it with `REQUIRE_THAT(...)` for clearer failure output than `at::allclose`.
- A real example to mirror: `tests/unit/tensors/test_Scalar.cxx`.

## Catch2 conventions used in this codebase

- Prefer `REQUIRE` (fatal) for invariants and `CHECK` (non-fatal) when you want multiple independent assertions to all run on failure.
- For exception checks: `REQUIRE_THROWS_AS(expr, ExceptionType)` or `REQUIRE_THROWS_WITH(expr, "message substring")`.
- For approximate scalar comparison: `REQUIRE(value == Catch::Approx(expected))` (needs `<catch2/matchers/catch_matchers_floating_point.hpp>` or the all-matchers header).
- For tensor comparison: `REQUIRE_THAT(actual, test::allclose(expected))` (tensor tests only — `utils.h` defines `test::allclose`).
- To skip a section conditionally (e.g., debug-only), use `SKIP("reason")` — see `tests/unit/base/test_Registry.cxx` for an example.

## After scaffolding

Print to the user:

```
Scaffolded tests/unit/<dir>/test_<Name>.cxx (TEST_CASE "<Name>", tag [<dir>]).

Build:
  cmake --build --preset dev

Run only this test case:
  ./build/dev/tests/unit/<binary> "<Name>"
  # or, by tag (runs everything in that directory):
  ./build/dev/tests/unit/<binary> "[<dir>]"
  # or, by section pattern:
  ./build/dev/tests/unit/<binary> "<Name>" -c "<section_name>"

CMake auto-discovers the new file via file(GLOB_RECURSE) — no CMakeLists.txt edits needed.
```

Substitute `<binary>` = `tensor_tests` for tensor tests, `unit_tests` otherwise.

## Easy mistakes to verify against

- **License header doesn't match `LICENSE` line-for-line** — `scripts/check_copyright.py` enforces an exact textual match (with `// ` prefix). Paraphrasing fails CI silently locally and loudly in CI.
- **Tensor header tested under `tests/unit/<other dir>/`** — it would compile and link, but into the wrong binary, and `tensor_tests --devices cuda` would skip it. Tensor headers go under `tests/unit/tensors/`.
- **More than one TEST_CASE in the file** — would still build, but breaks the one-header-one-file convention; reviewers will ask to split.

## Reference

- `tests/unit/README.md` — the canonical convention document. If anything in this skill conflicts with it, the README wins.
- `tests/unit/CMakeLists.txt` — the GLOB_RECURSE rules that decide which binary picks up the file.
- Examples to mirror:
  - `tests/unit/misc/test_string_utils.cxx` — minimal generic test.
  - `tests/unit/base/test_Registry.cxx` — generic test with conditional `SKIP`.
  - `tests/unit/tensors/test_Scalar.cxx` — minimal tensor test.
