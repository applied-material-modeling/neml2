Build the NEML2 C++ backend using a CMake preset.

Parse `$ARGUMENTS` for an optional preset name and/or build target.
Default preset: `dev`. Default target: all.

**Valid presets:**
| Preset      | Purpose                                      |
|-------------|----------------------------------------------|
| `dev`       | Debug build, all features on (default)       |
| `release`   | Optimized build                              |
| `cc`        | Export compile commands (PCH off, for IDEs)  |
| `tsan`      | Thread sanitizer                             |
| `asan`      | Address sanitizer                            |
| `coverage`  | Code coverage instrumentation                |
| `profiling` | Profiling build                              |

**Steps:**

1. Configure (only if `build/<preset>/` does not exist or `$ARGUMENTS` includes `--reconfigure`):
   ```bash
   cmake --preset <preset> -GNinja -S .
   ```

2. Build:
   ```bash
   cmake --build --preset <preset> [--target <target>]
   ```

**Named targets in `dev`:**
- `unit_tests` — Catch2 unit test binary
- `regression_tests` — regression tests
- `verification_tests` — verification tests
- `dispatcher_tests` — MPI dispatcher tests

**Examples:**
- `/build` — configure + build dev preset
- `/build release` — optimized build
- `/build dev unit_tests` — build only the unit test binary
- `/build --reconfigure` — force reconfigure then build

**File path resolution:** Header paths given in error messages or arguments may be nested under subdirectories (e.g. `include/neml2/models/solid_mechanics/elasticity/`). If a file is not found at the expected path, search for it with Glob before reporting an error:
```
include/neml2/**/<Filename>.h
```

**On success:** report the binary output path.

**On failure — iterative repair loop (max 3 attempts total):**

1. Extract the **first meaningful compiler error**: file path, line number, and error message.
   ```bash
   cmake --build --preset <preset> 2>&1 | grep -m1 "error:"
   ```

2. Classify the error:
   - **Trivial** (missing `#include`, obvious typo, missing `register_NEML2_object`): apply the minimal fix, limited to **1–2 files**, then retry the build.
   - **Non-trivial** (logic error, wrong API, structural mismatch): **do NOT modify code**. State the exact file and line, describe the fix needed, and stop.

3. Before retrying, check whether the error changed. If the **exact same error repeats**, stop immediately and report it — do not loop further.

4. After 3 build attempts (including the initial), stop and report the outstanding error.
   Suggest running `/fix-build` for deeper repair via the build-engineer agent.
