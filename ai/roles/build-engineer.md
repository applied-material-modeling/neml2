# Skill: BUILD-ENGINEER

Purpose: Diagnose and fix NEML2 build failures with minimal changes.

## Procedure

1. Read the full compiler or linker error first.
2. Read the relevant header and source before editing.
3. If the fix may touch NEML2 C++ code, read [NEML2-GUIDELINES](./neml2-guidelines.md) before editing.
4. Before touching any `CMakeLists.txt`, check whether sources are collected by `GLOB` or `GLOB_RECURSE`.
5. If globbing is used, do not edit `CMakeLists.txt`; ensure the file is placed correctly and that the source ends with `register_NEML2_object(ClassName)`.
6. If sources are explicit, add the file to `target_sources(...)`.
7. Apply the minimal fix.
8. Rebuild.
9. If a new error appears, apply one more targeted fix and rebuild.
10. Stop if the same error repeats or 3 build attempts are exhausted.
11. Never modify `CMakePresets.json` without explicit approval.

## Common Patterns to Check

- missing source registration
- missing `register_NEML2_object`
- PCH/include-order issues
- libtorch ABI mismatches
- missing MPI when dispatcher support is enabled
