# Download Doxygen
#
# Right now this only downloads the Doxygen binary for Linux.
#
# Influential variables:
# - DOXYGEN_VERSION
#
# Output variables are those defined by FetchContent such as:
# - doxygen_POPULATED
# - doxygen_SOURCE_DIR

if(UNIX AND NOT APPLE)
  string(REPLACE "." "_" DOXYGEN_RELEASE ${DOXYGEN_VERSION})
  FetchContent_Declare(
    doxygen
    URL https://github.com/doxygen/doxygen/releases/download/Release_${DOXYGEN_RELEASE}/doxygen-${DOXYGEN_VERSION}.linux.bin.tar.gz
  )

  message(STATUS "Downloading/updating Doxygen")
  FetchContent_MakeAvailable(doxygen)
endif()
