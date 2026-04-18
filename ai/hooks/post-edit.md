# Hook: Post-Edit

Purpose: Mandatory procedure after any code edit.

## After Any Code Edit

For edited C++ files (`.h`, `.cxx`, `.cpp`):

1. Locate a usable formatter in this order:
   - `clang-format`
   - `clang-format-20`
   - `clang-format-19`
   - `$(brew --prefix llvm)/bin/clang-format`
   - `/usr/local/opt/llvm/bin/clang-format`
2. If no formatter exists, install a low-risk formatter per the tool policy in AGENTS.md.
3. Run a formatting check:
   - `clang-format --dry-run -Werror <file>`
4. If the check fails, format the file and re-check.

For edited Python files (`.py`):

1. Ensure `black` is available, installing it if needed.
2. Run:
   - `black --check --line-length 100 <file>`
3. If the check fails, format the file and re-check.
