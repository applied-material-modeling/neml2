#!/usr/bin/env bash

export CC="${CMAKE_C_COMPILER}"
export CXX="${CMAKE_CXX_COMPILER}"
export CFLAGS="${CMAKE_C_FLAGS}"
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI} -I${SOURCE_DIR}"

# Set the SDKROOT if on macOS
if [[ "$(uname)" == "Darwin" ]]; then
  if command -v xcrun >/dev/null 2>&1; then
    export SDKROOT="$(xcrun --show-sdk-path)"
  fi
fi

"${SOURCE_DIR}"/configure \
  --prefix="${INSTALL_PREFIX}" \
  --enable-cpu-profiler \
  --disable-heap-profiler \
  --disable-debugalloc \
  --enable-static=no \
  --enable-shared=yes
make -j "${NEML2_CONTRIB_PARALLEL}"
make install
