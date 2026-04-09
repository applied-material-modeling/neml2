Diagnose and repair a failed NEML2 build.

This command is repair mode for `/build`. Use it when the build is already failing and you
want the build-engineer agent to diagnose and fix it. It does not replace `/build` for
normal build workflows.

---

## Steps

1. Run the build:
   ```bash
   cmake --build --preset dev 2>&1
   ```

2. **If the build succeeds:** report success and exit. No repair needed.

3. **If the build fails:**
   - Extract the first meaningful compiler or linker error (file, line, message):
     ```bash
     cmake --build --preset dev 2>&1 | grep -m1 "error:"
     ```
   - Invoke the **build-engineer agent** with the full error output.
   - The agent will diagnose the root cause, apply a minimal fix, and run up to 3 rebuild
     attempts internally before reporting the outcome.

4. **After the agent completes:**
   - If resolved: summarize what was changed and confirm the build is clean.
   - If unresolved: report the outstanding error (file + line + message) and explain why
     it was not auto-fixed (non-trivial, same error repeated, or 3 attempts exhausted).

---

## When to use

| Command | Use when |
|---------|----------|
| `/build` | Normal build — configure, build, light inline repair |
| `/fix-build` | Build is already failing — invoke build-engineer for deeper repair |

Do NOT use `/fix-build` as a substitute for `/build` in automated pipelines (e.g. `/implement`).
If `/build` exhausts its repair loop, suggest `/fix-build` as a manual follow-up step.
