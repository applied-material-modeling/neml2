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

# Run EVERY C++ ctest under instrumentation EXCEPT the `benchmark` label (heavy
# per-scenario neml2-compile smoke tests -- minutes each, and their runtime paths
# are already covered by the tests/aoti sweep below). A denylist (`-LE benchmark`),
# not an allowlist, so a NEW test suite is instrumented automatically: an allowlist
# silently drops a new label's coverage (that is exactly how krylov.h coverage was
# lost when `test_krylov` was added under a `krylov` label the filter didn't name).
# Every non-benchmark ctest links + exercises neml2/csrc, so this is pure gain.
ctest --test-dir "$BUILD_DIR" -LE 'benchmark' --output-on-failure

# --- Python-driven coverage of the AOTI runtime -----------------------------
# The C++ ctests only drive a forward leaf (the dispatch fixture); they never
# reach the implicit / sub-batch / derivative-selection paths in
# csrc/aoti/{ops,jacobian,Model}.cpp. The Python AOTI suite (tests/aoti) DOES --
# it compiles real models and calls forward/jvp/jacobian through the pybind
# binding, which loads libneml2. Make the binding load the *instrumented* library
# so those runs contribute to the same coverage profile; the per-process .profraw
# land in $RAW alongside the ctest ones and merge below.
#
# The binding (`neml2.aoti._aoti`) links libneml2 by a config-suffixed SONAME
# (`libneml2_<config>.so`, e.g. `libneml2_Coverage.so` -- CMakeLists OUTPUT_NAME).
# In an editable Coverage build the binding's rpath resolves to the freshly built
# *instrumented* neml2/lib/libneml2_<config>.so, so running tests/aoti drives that
# instrumented lib in-process and its calls land in the same profile -- no library
# swap needed. (We must NOT use LD_PRELOAD: it would be inherited by the g++
# subprocess Inductor spawns to build each .pt2 kernel, which then fails to start
# with "InductorError: InvalidCxxCompiler".) The old plain-`libneml2.so` match
# found nothing, so tests/aoti was silently skipped -- leaving only the thin ctest
# slice in the report and every AOTI/substep .cpp reading as 0%. Opt out with
# NEML2_CPP_COV_PYTEST=0.

# A clang source-based-coverage object carries __llvm_covmap / __llvm_prf_*
# sections. We test for them to (a) confirm the binding's resolved lib is actually
# instrumented before running tests/aoti, and (b) drop stale non-Coverage
# libneml2*.so siblings a local dev may have left in neml2/lib from a prior
# RelWithDebInfo build (a fresh CI runner has only the Coverage libs). Fall open if
# objdump is unavailable.
_instrumented() {
  command -v objdump >/dev/null 2>&1 || return 0
  # Capture then glob-match rather than `objdump | grep -q`: under `set -o
  # pipefail`, grep -q closing the pipe early sends objdump SIGPIPE (exit 141),
  # which would make the pipeline -- and this predicate -- spuriously fail.
  local hdrs
  hdrs="$(objdump -h "$1" 2>/dev/null)" || return 1
  [[ "$hdrs" == *__llvm* ]]
}

if [ "${NEML2_CPP_COV_PYTEST:-1}" = "1" ] && python -c "import neml2.aoti._aoti" 2>/dev/null; then
  AOTI_SO="$(python -c 'import neml2.aoti._aoti as m; print(m.__file__)' 2>/dev/null || true)"
  # The instrumented lib the binding actually loads (config-suffixed name).
  TARGET_LIB="$(ldd "$AOTI_SO" 2>/dev/null \
    | sed -n 's|.*\(libneml2[A-Za-z_]*\.so\) => \([^ ]*\).*|\2|p' | grep -v eager | head -1)"
  if [ -n "$TARGET_LIB" ] && _instrumented "$TARGET_LIB"; then
    echo "cpp_coverage: driving tests/aoti through the instrumented binding ($TARGET_LIB)"
    # Keep pyproject's addopts (--import-mode=importlib is load-bearing: it makes
    # `import neml2` resolve to the installed package + its pybind `_aoti.so`).
    # %p in LLVM_PROFILE_FILE gives every xdist worker its own profile.
    LLVM_PROFILE_FILE="$RAW/py-%p.profraw" \
      python -m pytest "$SRC/tests/aoti" -n auto -q -p no:cacheprovider \
      || echo "cpp_coverage: WARNING tests/aoti exited non-zero (coverage still merged)" >&2
  else
    echo "cpp_coverage: skipping Python coverage (binding does not resolve an instrumented libneml2)" >&2
  fi
fi

# Instrumented objects: the shared libraries (carry the neml2/csrc .cpp mapping)
# plus every test executable (carries the header-only template instantiations --
# batch_chunk.h, the _guarded / _assert helpers, ...). llvm-cov merges per-source
# coverage across all of them. The editable install stages the libneml2*.so into
# neml2/lib/ (not the build root); without them here llvm-cov would report only
# the test executables' header instantiations -- every runtime .cpp would silently
# read as 0%.
mapfile -t OBJECTS < <(
  while IFS= read -r lib; do _instrumented "$lib" && echo "$lib"; done \
    < <(find "$SRC/neml2/lib" "$BUILD_DIR" -maxdepth 1 -name 'libneml2*.so' 2>/dev/null)
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

# Honor the standard lcov exclusion markers in the source (LCOV_EXCL_LINE and
# LCOV_EXCL_START/STOP blocks). llvm-cov does NOT parse these -- unlike geninfo --
# so we apply them here: drop the DA:/BRDA: records for any excluded source line
# before upload. This lets us exclude verbosity-gated diagnostic logging (whose
# many level/collect branch combinations are impractical to fully exercise) from
# the `cpp` coverage flag without weakening coverage of the surrounding logic.
python "$SRC/scripts/lcov_apply_exclusions.py" "$SRC" "$OUT_DIR/coverage.lcov"

echo "cpp_coverage: lcov written to $OUT_DIR/coverage.lcov"
