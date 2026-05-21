# ----------------------------------------------------------------------------
# Subdirectories
# ----------------------------------------------------------------------------
# catch-all test target
add_custom_target(tests)

# Test utilities
add_subdirectory(src)

# Example extension library for testing dynamic registration
add_subdirectory(extension)

# Unit tests
add_subdirectory(unit)
add_dependencies(unit_tests extension)
add_dependencies(tensor_tests extension)
add_dependencies(tests unit_tests)
add_dependencies(tests tensor_tests)

# Regression tests
add_subdirectory(regression)
add_dependencies(tests regression_tests)

# Verification tests
add_subdirectory(verification)
add_dependencies(tests verification_tests)

# Work dispatcher tests — always built. Tests that exercise SimpleMPIScheduler
# (currently none) will need an `if(NEML2_MPI)` guard or a runtime SKIP.
add_subdirectory(dispatchers)
add_dependencies(tests dispatcher_tests)
