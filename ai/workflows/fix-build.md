# Workflow: FIX-BUILD

Purpose: Diagnose and repair an already failing build.

## Procedure

1. Run:
   - `cmake --build --preset dev`
2. If the build succeeds, report success and stop.
3. If it fails, extract the first meaningful compiler or linker error.
4. Apply the [BUILD-ENGINEER](../roles/build-engineer.md) role guide.
5. Rebuild after each targeted fix, with a hard limit of 3 build attempts total.
6. If resolved, summarize what changed and confirm the build is clean.
7. If unresolved, report the exact outstanding error and why auto-repair stopped.
