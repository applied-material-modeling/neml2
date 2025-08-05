# Download code from url
function(download_from_url NAME URL PREFIX)
  message(STATUS "Downloading/updating ${NAME}")
  include(FetchContent)
  set(SOURCE_DIR ${PREFIX}/${NAME}-src)
  set(SUBBUILD_DIR ${PREFIX}/${NAME}-subbuild)
  set(BINARY_DIR ${PREFIX}/${NAME}-build)
  FetchContent_Populate(
    ${NAME}
    QUIET
    URL ${URL}
    SOURCE_DIR ${SOURCE_DIR}
    BINARY_DIR ${BINARY_DIR}
    SUBBUILD_DIR ${SUBBUILD_DIR}
  )
endfunction()

# Download code from git
function(download_from_git NAME REPO PREFIX TAG)
  message(STATUS "Downloading/updating ${NAME}")
  include(FetchContent)
  set(SOURCE_DIR ${PREFIX}/${NAME}-src)
  set(SUBBUILD_DIR ${PREFIX}/${NAME}-subbuild)
  set(BINARY_DIR ${PREFIX}/${NAME}-build)
  FetchContent_Populate(
    ${NAME}
    QUIET
    GIT_REPOSITORY ${REPO}
    GIT_TAG ${TAG}
    SOURCE_DIR ${SOURCE_DIR}
    BINARY_DIR ${BINARY_DIR}
    SUBBUILD_DIR ${SUBBUILD_DIR}
  )
endfunction()

# Install a dependency by running a custom script
function(custom_install NAME SCRIPT SOURCE_DIR BINARY_DIR INSTALL_PREFIX)
  message(STATUS "Installing ${NAME}")
  configure_file(${SCRIPT} ${BINARY_DIR}/install_${NAME}.sh)
  execute_process(
    COMMAND bash -c ./install_${NAME}.sh
    WORKING_DIRECTORY ${BINARY_DIR}
    OUTPUT_QUIET OUTPUT_FILE configure.log
    ERROR_QUIET ERROR_FILE configure.err
    COMMAND_ERROR_IS_FATAL ANY
  )
endfunction()

# Install a file and all files with the same name but different extensions
function(install_glob path install_dir component)
  get_filename_component(dir ${path} DIRECTORY)
  get_filename_component(filename ${path} NAME_WE)
  file(GLOB files "${dir}/${filename}*")

  if(NOT files)
    message(WARNING "No files found for pattern: ${pattern}")
    return()
  endif()

  install(FILES ${files} DESTINATION "${install_dir}" COMPONENT ${component})
endfunction()
