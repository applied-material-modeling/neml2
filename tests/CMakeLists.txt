# ----------------------------------------------------------------------------
# Catch2
# ----------------------------------------------------------------------------
find_package(Catch2)

if(NOT Catch2_FOUND)
  include(DownloadCatch2)
  set(Catch2_ROOT ${CATCH2_INSTALL_DIR} CACHE PATH "Catch2 root directory" FORCE)
  find_package(Catch2 REQUIRED)
endif()

# ----------------------------------------------------------------------------
# Subdirectories
# ----------------------------------------------------------------------------
# Test utilities
add_subdirectory(src)

# Example extension library for testing dynamic registration
add_subdirectory(extension)

# Unit tests
add_subdirectory(unit)

# Regression tests
add_subdirectory(regression)

# Verification tests
add_subdirectory(verification)

# Work dispatcher tests
if(NEML2_WORK_DISPATCHER)
  add_subdirectory(dispatchers)
endif()
