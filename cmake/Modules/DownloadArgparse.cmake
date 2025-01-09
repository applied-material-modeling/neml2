# Download Argparse
#
# Influential variables:
# - ARGPARSE_VERSION
#
# Output variables are those defined by FetchContent such as:
# - argparse_POPULATED
# - argparse_SOURCE_DIR
#
# as well as other variables defined by the argparse project.

include(FetchContent)
FetchContent_Declare(
  argparse
  GIT_REPOSITORY https://github.com/p-ranav/argparse.git
  GIT_TAG v${ARGPARSE_VERSION}
)

message(STATUS "Downloading/updating Argparse")
FetchContent_MakeAvailable(argparse)
