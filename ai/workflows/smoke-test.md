# Workflow: SMOKE-TEST

Purpose: verify the workflow still works without `.claude/`.

## Procedure

1. **Isolation:** Record changed files before and after any step that edits files.
2. **Reversion:** Immediately revert temporary smoke-test edits after each such step using `git checkout -- <file>` or `git clean -fd` for new files. **Do not leave the workspace in a dirty state.**
3. **Check:**
   - [BUILD](./build.md) variants
   - [TEST](./test.md) variants
   - [Post-Edit](../hooks/post-edit.md) procedure
   - [TEST-WRITER](../roles/test-writer.md) procedure on an existing model
   - [BUILD-ENGINEER](../roles/build-engineer.md) procedure on new-file registration reasoning
   - [DOC-WRITER](../roles/doc-writer.md) procedure on an existing model
   - full [IMPLEMENT](./implement.md) pipeline on a disposable stub model
4. **Report:** `PASS`, `FAIL`, or `PARTIAL` for each check and summarize at the end.
