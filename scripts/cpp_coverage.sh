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
# Resolve to an absolute path: ctest chdir's into the build tree, and the
# instrumented binaries resolve LLVM_PROFILE_FILE relative to *their* working
# directory -- a relative path would scatter the .profraw files out of reach.
BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
OUT_DIR="${2:-$BUILD_DIR/coverage}"
LLVM_PROFDATA="${LLVM_PROFDATA:-llvm-profdata}"
LLVM_COV="${LLVM_COV:-llvm-cov}"

RAW="$OUT_DIR/raw"
rm -rf "$OUT_DIR"
mkdir -p "$RAW"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
RAW="$OUT_DIR/raw"

# Each test process writes its own raw profile (%p = pid), to the absolute RAW
# dir. The fixture-compile step is a separate Python process and writes nothing.
export LLVM_PROFILE_FILE="$RAW/%p.profraw"

# `dispatcher` (AOTI runtime + schedulers) and `eager` (embedded-Python runtime)
# -- both label groups instrument neml2/csrc sources, so cover them together.
ctest --test-dir "$BUILD_DIR" -L 'dispatcher|eager' --output-on-failure

# --- Python-driven coverage of the AOTI runtime -----------------------------
# The C++ ctests only drive a forward leaf (the dispatch fixture); they never
# reach the implicit / sub-batch / derivative-selection paths in
# csrc/aoti/{ops,jacobian,Model}.cpp. The Python AOTI suite (tests/aoti) DOES --
# it compiles real models and calls forward/jvp/jacobian through the pybind
# binding, which loads libneml2. Point that load at the *instrumented* library so
# those runs contribute to the same coverage profile.
#
# The binding's DT_NEEDED is `libneml2.so`, but a non-Release build names the
# library `libneml2_<config>.so` with a matching SONAME (see CMakeLists OUTPUT_NAME),
# so a plain LD_PRELOAD would not satisfy the NEEDED entry. Stage a copy with the
# SONAME rewritten to `libneml2.so` and preload that; its per-process .profraw land
# in $RAW alongside the ctest ones and merge below. Opt out with NEML2_CPP_COV_PYTEST=0.
if [ "${NEML2_CPP_COV_PYTEST:-1}" = "1" ] && python -c "import neml2.aoti" 2>/dev/null; then
  NEML2_LIB="$(find "$BUILD_DIR" -maxdepth 1 -name 'libneml2*.so' ! -name '*eager*' | head -1)"
  if [ -n "$NEML2_LIB" ] && command -v patchelf >/dev/null 2>&1; then
    PRELOAD_DIR="$OUT_DIR/preload"
    mkdir -p "$PRELOAD_DIR"
    cp "$NEML2_LIB" "$PRELOAD_DIR/libneml2.so"
    patchelf --set-soname libneml2.so "$PRELOAD_DIR/libneml2.so"
    echo "cpp_coverage: driving tests/aoti through the instrumented runtime (LD_PRELOAD $NEML2_LIB)"
    # Keep pyproject's addopts (--import-mode=importlib is load-bearing: it makes
    # `import neml2` resolve to the installed package + its pybind `_aoti.so`).
    # %p in LLVM_PROFILE_FILE gives every xdist worker its own profile.
    LD_PRELOAD="$PRELOAD_DIR/libneml2.so" LLVM_PROFILE_FILE="$RAW/py-%p.profraw" \
      python -m pytest "$SRC/tests/aoti" -n auto -q -p no:cacheprovider \
      || echo "cpp_coverage: WARNING tests/aoti exited non-zero (coverage still merged)" >&2
  else
    echo "cpp_coverage: skipping Python coverage (need patchelf + a built libneml2 under $BUILD_DIR)" >&2
  fi
fi

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
# Rewrite the absolute SF: paths to repo-relative so Codecov can merge uploads
# from different runners (ubuntu vs gpu_runner have different workspace prefixes;
# mismatched paths would otherwise be treated as distinct files and not unioned).
"$LLVM_COV" export "${OBJ_ARGS[@]}" \
  -instr-profile="$OUT_DIR/merged.profdata" -format=lcov "$SRC/neml2/csrc" \
  | sed "s|SF:${SRC}/|SF:|" >"$OUT_DIR/coverage.lcov"

echo "cpp_coverage: lcov written to $OUT_DIR/coverage.lcov"
