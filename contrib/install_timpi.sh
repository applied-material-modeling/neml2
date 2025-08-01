#!/usr/bin/env bash

export CC=${MPI_C_COMPILER}
export CXX=${MPI_CXX_COMPILER}
export CXXFLAGS="-D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}"
export METHODS="${timpi_BUILD_TYPE}"

"${SOURCE_DIR}"/configure --prefix="${INSTALL_PREFIX}" --enable-shared
make -j "${NEML2_CONTRIB_PARALLEL}"
make install
