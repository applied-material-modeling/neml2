#!/usr/bin/env bash
# Copyright 2024, UChicago Argonne, LLC
# All Rights Reserved
# Software Name: NEML2 -- the New Engineering material Model Library, version 2
# By: Argonne National Laboratory
# OPEN SOURCE LICENSE (MIT)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${1:-$SRC/build/dev}"
OUT_DIR="${2:-$BUILD_DIR/coverage}"
LLVM_PROFDATA="${LLVM_PROFDATA:-llvm-profdata}"
LLVM_COV="${LLVM_COV:-llvm-cov}"

RAW="$OUT_DIR/raw"
rm -rf "$OUT_DIR"
mkdir -p "$RAW"

# Each test process writes its own raw profile (%p = pid). The fixture-compile
# step is a separate Python process and simply writes nothing.
export LLVM_PROFILE_FILE="$RAW/%p.profraw"

ctest --test-dir "$BUILD_DIR" -L dispatcher --output-on-failure

# Instrumented objects: the shared library (carries the neml2/csrc mapping) plus
# every test executable (carries the header-only template instantiations --
# batch_chunk.h, the _guarded / _assert helpers, ...). llvm-cov merges per-source
# coverage across all of them.
mapfile -t OBJECTS < <(
  find "$BUILD_DIR" -maxdepth 1 -name 'libneml2*.so'
  find "$BUILD_DIR/tests/cpp" -maxdepth 1 -type f -perm -u+x -name 'test_*'
)
if [ "${#OBJECTS[@]}" -eq 0 ]; then
  echo "cpp_coverage: no instrumented objects under $BUILD_DIR" >&2
  exit 1
fi
OBJ_ARGS=("${OBJECTS[0]}")
for o in "${OBJECTS[@]:1}"; do OBJ_ARGS+=(-object "$o"); done

"$LLVM_PROFDATA" merge -sparse "$RAW"/*.profraw -o "$OUT_DIR/merged.profdata"

# Report + export, restricted to the runtime sources (the trailing path filters
# out torch headers, the generated aoti_export.h, and the tests themselves).
"$LLVM_COV" report "${OBJ_ARGS[@]}" \
  -instr-profile="$OUT_DIR/merged.profdata" "$SRC/neml2/csrc"
"$LLVM_COV" export "${OBJ_ARGS[@]}" \
  -instr-profile="$OUT_DIR/merged.profdata" -format=lcov "$SRC/neml2/csrc" \
  >"$OUT_DIR/coverage.lcov"

echo "cpp_coverage: lcov written to $OUT_DIR/coverage.lcov"
