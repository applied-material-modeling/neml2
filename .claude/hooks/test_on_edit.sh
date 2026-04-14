#!/usr/bin/env bash
# Post-edit hook: run the associated unit test when a *.cxx or *.h file is edited.
#
# Retry logic (mirrors user preference):
#   - Source/header edited → reset attempt counter → run test
#     * PASS: celebrate and exit
#     * FAIL: instruct Claude to fix the TEST FILE (attempt 1/3)
#   - Test file edited → run test again
#     * PASS: reset counter and exit
#     * FAIL attempts 1-2: instruct Claude to fix TEST FILE again
#     * FAIL attempt 3: instruct Claude to fix SOURCE FILE and notify user

set -uo pipefail

# ── Parse file path from hook JSON (same shape as post_edit.sh) ─────────────
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

EXT="${FILE##*.}"
BASENAME=$(basename "$FILE")
BASENAME_NO_EXT="${BASENAME%.*}"

# Only care about C++ source and header files
[[ "$EXT" != "cxx" && "$EXT" != "h" ]] && exit 0

REPO_ROOT="$(git -C "$(dirname "$FILE")" rev-parse --show-toplevel 2>/dev/null || echo "")"
[ -z "$REPO_ROOT" ] && exit 0

TEST_BINARY="$REPO_ROOT/build/dev/tests/unit/unit_tests"

# ── Determine whether the edited file is a test file or a source/header ─────
IS_TEST=0
SOURCE_BASE=""
TEST_FILE=""

if [[ "$BASENAME_NO_EXT" == test_* ]]; then
    IS_TEST=1
    SOURCE_BASE="${BASENAME_NO_EXT#test_}"
    TEST_FILE="$FILE"
else
    SOURCE_BASE="$BASENAME_NO_EXT"
    TEST_FILE=$(find "$REPO_ROOT/tests/unit" -name "test_${SOURCE_BASE}.cxx" 2>/dev/null | head -1)
fi

# Nothing to do if no test file found
if [ -z "$TEST_FILE" ] || [ ! -f "$TEST_FILE" ]; then
    exit 0
fi

# ── Attempt counter (keyed on source basename, lives in /tmp) ────────────────
COUNTER_FILE="/tmp/neml2_test_fix_${SOURCE_BASE}.count"

# Editing a source/header resets the counter
if [ "$IS_TEST" -eq 0 ]; then
    rm -f "$COUNTER_FILE"
fi

# ── Extract test case name from TEST_CASE("...", ...) ───────────────────────
TEST_NAME=$(grep -m1 'TEST_CASE' "$TEST_FILE" 2>/dev/null \
    | sed 's/.*TEST_CASE("\([^"]*\)".*/\1/')

if [ -z "$TEST_NAME" ]; then
    echo "[neml2-test] Could not parse TEST_CASE name from ${TEST_FILE#$REPO_ROOT/}"
    exit 0
fi

# ── Guard: test binary must exist ────────────────────────────────────────────
if [ ! -f "$TEST_BINARY" ]; then
    echo "[neml2-test] Test binary missing — run /build first"
    exit 0
fi

# ── Run the test ─────────────────────────────────────────────────────────────
TEST_OUTPUT=$("$TEST_BINARY" "$TEST_NAME" 2>&1)
TEST_EXIT=$?

REL_TEST="${TEST_FILE#$REPO_ROOT/}"
REL_FILE="${FILE#$REPO_ROOT/}"

if [ "$TEST_EXIT" -eq 0 ]; then
    rm -f "$COUNTER_FILE"
    echo "[neml2-test] PASSED: $TEST_NAME"
    exit 0
fi

# ── Test failed — compute and store attempt count ────────────────────────────
if [ "$IS_TEST" -eq 1 ]; then
    ATTEMPTS=$(cat "$COUNTER_FILE" 2>/dev/null || echo "0")
    ATTEMPTS=$((ATTEMPTS + 1))
    echo "$ATTEMPTS" > "$COUNTER_FILE"
else
    # First failure triggered by source edit — attempt 0 (no fix tried yet)
    ATTEMPTS=0
fi

# Print failing test output (last 40 lines to stay readable)
echo "[neml2-test] FAILED: $TEST_NAME"
echo "─────────────────────────────────────"
echo "$TEST_OUTPUT" | tail -40
echo "─────────────────────────────────────"
echo ""

if [ "$ATTEMPTS" -lt 3 ]; then
    NEXT=$((ATTEMPTS + 1))
    echo "INSTRUCTION (attempt $NEXT/3):"
    echo "  The failure is likely because the algorithm changed slightly."
    echo "  Fix the TEST FILE to match the new behavior — do NOT touch the source yet."
    echo "  Test file: $REL_TEST"
else
    # All 3 test-fix attempts exhausted
    rm -f "$COUNTER_FILE"

    # Locate the source file(s) for reference
    SRC=$(find "$REPO_ROOT/src" "$REPO_ROOT/include" \
        \( -name "${SOURCE_BASE}.cxx" -o -name "${SOURCE_BASE}.h" \) 2>/dev/null \
        | sed "s|$REPO_ROOT/||" | tr '\n' '  ')

    echo "INSTRUCTION (all 3 test-fix attempts exhausted):"
    echo "  Fix the SOURCE FILE instead — the test cannot be made to pass by editing the test alone."
    echo "  Source file(s): ${SRC:-$REL_FILE}"
    echo ""
    echo "  Please NOTIFY THE USER that 3 attempts to fix the test failed and the source"
    echo "  was corrected instead."
fi

exit 0
