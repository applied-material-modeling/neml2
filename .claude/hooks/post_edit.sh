#!/usr/bin/env bash
# Lightweight post-edit linting for NEML2.
# Receives Claude tool JSON on stdin; exits non-zero to surface style issues.

FILE=$(python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(data.get('file_path', ''))
except Exception:
    print('')
" 2>/dev/null)

[ -z "$FILE" ] && exit 0
[ ! -f "$FILE" ] && exit 0

REPO_ROOT="$(git -C "$(dirname "$FILE")" rev-parse --show-toplevel 2>/dev/null || echo "")"

case "$FILE" in
  *.h|*.cxx|*.cpp)
    CF=""
    for candidate in clang-format clang-format-20 clang-format-19 \
        "$(brew --prefix llvm 2>/dev/null)/bin/clang-format" \
        /usr/local/opt/llvm/bin/clang-format; do
      if command -v "$candidate" &>/dev/null 2>&1; then CF="$candidate"; break; fi
    done
    if [ -z "$CF" ] && command -v brew &>/dev/null; then
      echo "[neml2] clang-format not found — installing via brew install llvm ..."
      brew install llvm 2>&1
      CF="$(brew --prefix llvm 2>/dev/null)/bin/clang-format"
      command -v "$CF" &>/dev/null 2>&1 || CF=""
    fi
    [ -z "$CF" ] && exit 0
    if "$CF" --dry-run -Werror "$FILE" 2>/dev/null; then
      echo "[neml2] clang-format OK: ${FILE#$REPO_ROOT/}"
    else
      echo "[neml2] Style issue — run: $CF -i $FILE"
      exit 1
    fi
    ;;
  *.py)
    if ! command -v black &>/dev/null; then
      echo "[neml2] black not found — installing via pip ..."
      python3 -m pip install black 2>&1
    fi
    if ! command -v black &>/dev/null; then exit 0; fi
    if black --check --line-length 100 --quiet "$FILE" 2>/dev/null; then
      echo "[neml2] black OK: ${FILE#$REPO_ROOT/}"
    else
      echo "[neml2] Style issue — run: black --line-length 100 $FILE"
      exit 1
    fi
    ;;
esac

exit 0
