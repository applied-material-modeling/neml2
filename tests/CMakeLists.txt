# ----------------------------------------------------------------------------
# Dependencies and 3rd party packages
# ----------------------------------------------------------------------------
message(STATUS "Configuring Catch2")
FetchContent_MakeAvailable(Catch2)

# ----------------------------------------------------------------------------
# Subdirectories
# ----------------------------------------------------------------------------
# Test utilities
add_subdirectory(src)

# Unit tests
add_subdirectory(unit)

# Regression tests
add_subdirectory(regression)

# Verification tests
add_subdirectory(verification)

# Work dispatcher tests
add_subdirectory(dispatcher)
