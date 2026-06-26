# ----------------------------------------------------------------------------
# pkg-config (for non-CMake downstream consumers)
# ----------------------------------------------------------------------------
# Generates these .pc files at <install-prefix>/share/pkgconfig:
#
#   neml2.pc        — meta entry. `pkg-config --cflags --libs neml2` pulls
#                     in neml2-core plus every external dep neml2 needs.
#   neml2-core.pc   — just `-L${libdir} -lneml2` and `-I${includedir}`.
#                     For callers that already manage their own torch /
#                     nlohmann_json discovery.
#   neml2-torch.pc  — torch include + link flags (always external; libtorch
#                     is never bundled into the NEML2 install).
#   neml2-nlohmann-json.pc — header-only nlohmann_json. Bundled (relocatable
#                     `${prefix}/include` path) when we built it ourselves
#                     from the submodule, external (absolute path) otherwise.
#
# The configs are relocatable: `cmake --install --prefix /opt/neml2` plus
# `PKG_CONFIG_PATH=/opt/neml2/share/pkgconfig` gives sane absolute paths
# without re-running configure.
#
# Generation is the canonical two-step: `configure_file(... @ONLY)`
# expands `@VAR@` placeholders into an intermediate under the build tree,
# then `file(GENERATE INPUT ...)` resolves `$<CONFIG>` generator
# expressions at build time. The intermediate `*.pc.in` left in the build
# directory is harmless — it's neither installed nor consumed by anything
# downstream.
if(IS_ABSOLUTE "${INSTALL_LIBDIR}")
      set(NEML2_PKGCONFIG_LIBDIR "${INSTALL_LIBDIR}")
else()
      set(NEML2_PKGCONFIG_LIBDIR "\${exec_prefix}/${INSTALL_LIBDIR}")
endif()
# .pc files live at <prefix>/share/pkgconfig — two levels under prefix.
set(NEML2_PKGCONFIG_PREFIX "\${pcfiledir}/../..")

# neml2-core.pc lib list: every neml2-owned target reachable from aoti
# (today that's just aoti itself; the loop keeps the right shape for when
# we add more libraries later).
set(NEML2_PKGCONFIG_LIB_TARGETS aoti)
get_target_property(_aoti_interface_libs aoti INTERFACE_LINK_LIBRARIES)
foreach(_lib IN LISTS _aoti_interface_libs)
      if(TARGET "${_lib}")
            list(APPEND NEML2_PKGCONFIG_LIB_TARGETS "${_lib}")
      endif()
endforeach()
list(REMOVE_DUPLICATES NEML2_PKGCONFIG_LIB_TARGETS)
# Bake an rpath to our own libdir so a pkg-config consumer resolves libneml2 at
# runtime without LD_LIBRARY_PATH (pkg-config expands ${libdir} to the absolute
# install location).
set(NEML2_PKGCONFIG_LIBS "-L\${libdir} -Wl,-rpath,\${libdir}")
foreach(_lib_target IN LISTS NEML2_PKGCONFIG_LIB_TARGETS)
      # External targets (torch::core, nlohmann_json::nlohmann_json) don't
      # ship a TARGET_FILE_BASE_NAME — skip them; their flags travel through
      # their own .pc files (or the consumer's environment for header-only).
      get_target_property(_t_type ${_lib_target} TYPE)
      if(_t_type STREQUAL "SHARED_LIBRARY" OR _t_type STREQUAL "STATIC_LIBRARY")
            string(APPEND NEML2_PKGCONFIG_LIBS " -l$<TARGET_FILE_BASE_NAME:${_lib_target}>")
      endif()
endforeach()

# Per-dependency .pc generator. Bundled writes a relocatable `${prefix}`
# template (the dep lives under the NEML2 install prefix); external writes
# absolute paths to the dep's discovery location. Dependency .pc files don't
# need per-config suffixes — the dep ABIs don't change with our build type.
function(neml2_pkgconfig_generate_dep_pc)
      cmake_parse_arguments(ARG "" "OUTPUT_NAME;NAME;DESCRIPTION;VERSION;LIBS;CFLAGS;BUNDLED" "" ${ARGN})
      set(NEML2_DEP_PC_NAME        "${ARG_NAME}")
      set(NEML2_DEP_PC_DESCRIPTION "${ARG_DESCRIPTION}")
      set(NEML2_DEP_PC_VERSION     "${ARG_VERSION}")
      set(NEML2_DEP_PC_LIBS        "${ARG_LIBS}")
      set(NEML2_DEP_PC_CFLAGS      "${ARG_CFLAGS}")
      if(ARG_BUNDLED)
            set(_tpl ${NEML2_SOURCE_DIR}/cmake/neml2-dep-bundled.pc.in)
      else()
            set(_tpl ${NEML2_SOURCE_DIR}/cmake/neml2-dep-external.pc.in)
      endif()
      configure_file(${_tpl} ${NEML2_BINARY_DIR}/${ARG_OUTPUT_NAME}.pc @ONLY)
endfunction()

# neml2-torch.pc (torch is always external; we never bundle libtorch). Locate
# torch's lib dir RELATIVE to this .pc for a wheel install: neml2 and torch are
# siblings in site-packages -- the same assumption libneml2's own $ORIGIN rpath
# relies on -- so the shipped .pc resolves to the *user's* torch, not the build
# machine's. An editable .pc lives in the source tree (torch is not a sibling),
# so it bakes the discovered absolute path, which is this machine's torch. An
# rpath is emitted alongside -L so a pkg-config consumer needs no LD_LIBRARY_PATH.
if(NEML2_EDITABLE)
      get_filename_component(_torch_pc_libdir "${torch_LIBRARY}" DIRECTORY)
else()
      # <prefix>/share/pkgconfig/neml2-torch.pc -> three levels up is site-packages,
      # then torch/lib (the sibling torch package).
      set(_torch_pc_libdir "\${pcfiledir}/../../../torch/lib")
endif()
set(_torch_pc_libs "-L${_torch_pc_libdir} -Wl,-rpath,${_torch_pc_libdir} -lc10 -ltorch -ltorch_cpu")
if(TARGET torch::cuda)
      # The pip torch wheel ships the CUDA libs in the same lib dir as c10.
      string(APPEND _torch_pc_libs " -lc10_cuda -ltorch_cuda -ltorch_cuda_linalg")
endif()
neml2_pkgconfig_generate_dep_pc(
      OUTPUT_NAME  neml2-torch
      NAME         "neml2-torch"
      DESCRIPTION  "LibTorch dependency for NEML2"
      VERSION      ""
      LIBS         "${_torch_pc_libs}"
      CFLAGS       "-I${torch_INCLUDE_DIR} -I${torch_csrc_INCLUDE_DIR}"
      BUNDLED      FALSE
)

# neml2-nlohmann-json.pc (header-only; relocatable when bundled)
if(nlohmann_json_CONTRIB OR NEML2_WHEEL)
      set(_json_pc_cflags "-I\${includedir}")
      set(_json_pc_bundled TRUE)
else()
      set(_json_pc_cflags "-I${nlohmann_json_DIR}/include")
      set(_json_pc_bundled FALSE)
endif()
neml2_pkgconfig_generate_dep_pc(
      OUTPUT_NAME  neml2-nlohmann-json
      NAME         "neml2-nlohmann-json"
      DESCRIPTION  "nlohmann/json dependency for NEML2"
      VERSION      ""
      LIBS         ""
      CFLAGS       "${_json_pc_cflags}"
      BUNDLED      ${_json_pc_bundled}
)

set(NEML2_PKGCONFIG_REQUIRES "neml2-torch neml2-nlohmann-json")
set(NEML2_PKGCONFIG_SUFFIX "$<IF:$<CONFIG:Release>,,_$<CONFIG>>")

# neml2.pc + neml2-core.pc — two-step: configure_file for @VAR@, then
# file(GENERATE INPUT) for $<CONFIG>.
foreach(_pc neml2 neml2-core)
      configure_file(
            ${NEML2_SOURCE_DIR}/cmake/${_pc}.pc.in
            ${NEML2_BINARY_DIR}/${_pc}.pc.in
            @ONLY
      )
      file(GENERATE
            OUTPUT ${NEML2_BINARY_DIR}/${_pc}${NEML2_PKGCONFIG_SUFFIX}.pc
            INPUT ${NEML2_BINARY_DIR}/${_pc}.pc.in
      )
endforeach()

# Shipped in both the wheel and an editable install (the latter into the source
# tree via INSTALL_SHAREDIR). The .pc files are relocatable -- prefix is derived
# from ${pcfiledir} at `pkg-config` time -- so they work from wherever they land.
install(
      FILES
      ${NEML2_BINARY_DIR}/neml2${NEML2_PKGCONFIG_SUFFIX}.pc
      ${NEML2_BINARY_DIR}/neml2-core${NEML2_PKGCONFIG_SUFFIX}.pc
      ${NEML2_BINARY_DIR}/neml2-torch.pc
      ${NEML2_BINARY_DIR}/neml2-nlohmann-json.pc
      DESTINATION ${INSTALL_SHAREDIR}/pkgconfig
      COMPONENT libneml2
)
