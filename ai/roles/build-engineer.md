# Skill: BUILD-ENGINEER

Purpose: Diagnose and fix NEML2 build failures with minimal changes.

## Procedure

1. Read the full compiler or linker error first.
2. Read the relevant header and source before editing.
3. Before touching any `CMakeLists.txt`, check whether sources are collected by `GLOB` or `GLOB_RECURSE`.
4. If globbing is used, do not edit `CMakeLists.txt`; ensure the file is placed correctly and that the source ends with `register_NEML2_object(ClassName)`.
5. If sources are explicit, add the file to `target_sources(...)`.
6. Apply the minimal fix.
7. Rebuild.
8. If a new error appears, apply one more targeted fix and rebuild.
9. Stop if the same error repeats or 3 build attempts are exhausted.
10. Never modify `CMakePresets.json` without explicit approval.

## Common Patterns to Check

- missing source registration
- missing `register_NEML2_object`
- PCH/include-order issues
- libtorch ABI mismatches
- missing MPI when dispatcher support is enabled
