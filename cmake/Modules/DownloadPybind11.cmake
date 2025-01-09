# Download pybind11 source code from GitHub
#
# Influential variables:
# - PYBIND11_VERSION
#
# Output variables are those defined by FetchContent such as:
# - pybind11_POPULATED
# - pybind11_SOURCE_DIR
#
# as well as other variables defined by the pybind11 project.

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v${PYBIND11_VERSION}
)

message(STATUS "Downloading/updating Pybind11")
FetchContent_MakeAvailable(pybind11)
