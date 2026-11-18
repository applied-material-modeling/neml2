# -----------------------------------------------------------------------------
# python_package_dir
# -----------------------------------------------------------------------------
# Ask a Python interpreter where `import <package>` resolves, without importing
# it (importlib.util.find_spec only locates the package, so there are no import
# side effects or startup cost). This covers layouts that guessing from the
# site directories misses, e.g. a `--system-site-packages` venv whose package
# lives in the base interpreter's site-packages rather than the venv's own.
#
# Usage:
#   python_package_dir(<python_executable> <package> <result_var>)
#   Sets <result_var> to the package's directory if it resolves to an existing
#   directory, else to the empty string.
#
# Shared by the shipped Find modules (Findtorch, Findnmhit), so it is installed
# alongside them and included via CMAKE_CURRENT_LIST_DIR.

include_guard(GLOBAL)

function(python_package_dir python package result_var)
  execute_process(
    COMMAND ${python} -c
      "import importlib.util as u; s = u.find_spec('${package}'); print(s.submodule_search_locations[0] if s and s.submodule_search_locations else '')"
    OUTPUT_VARIABLE _dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(_dir AND IS_DIRECTORY "${_dir}")
    set(${result_var} "${_dir}" PARENT_SCOPE)
  else()
    set(${result_var} "" PARENT_SCOPE)
  endif()
endfunction()
