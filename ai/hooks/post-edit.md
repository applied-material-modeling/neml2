# Hook: Post-Edit

Purpose: Mandatory procedure after any code edit.

## After Any New File Is Created

For any new file with extension `.h`, `.cxx`, `.py`, `.sh`, or `.js`:

1. Run:
   - `python scripts/check_copyright.py`
2. If the check reports missing headers, fix them automatically:
   - `python scripts/check_copyright.py -m`
3. Re-run the check to confirm it passes before committing.

## After Any Code Edit

For edited C++ files (`.h`, `.cxx`, `.cpp`):

1. Resolve the formatter path using:
   - `bash ai/scripts/find-clang-format.sh`
2. If no formatter exists, install a low-risk formatter per the tool policy in AGENTS.md.
3. Run a formatting check:
   - ``$(bash ai/scripts/find-clang-format.sh) --dry-run -Werror <file>``
4. If the check fails, format the file and re-check with the same resolved binary.

For edited Python files (`.py`):

1. Ensure `black` is available, installing it if needed.
2. Run:
   - `black --check --line-length 100 <file>`
3. If the check fails, format the file and re-check.
