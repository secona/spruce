find_package(Python3 REQUIRED COMPONENTS Interpreter)

if(NOT LLVM_EXTERNAL_LIT AND NOT EXISTS "${LLVM_DEFAULT_EXTERNAL_LIT}")
  find_program(LLVM_EXTERNAL_LIT NAMES llvm-lit lit
    HINTS "${LLVM_TOOLS_BINARY_DIR}" REQUIRED)
endif()

set(LLVM_LIT_ARGS "-sv" CACHE STRING "Default options for lit")

configure_lit_site_cfg(
  "${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in"
  "${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py"
  MAIN_CONFIG "${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py"
)

add_lit_testsuite(check-spruce "Running the Spruce regression tests"
  "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS spruce-opt
)

get_llvm_lit_path(SPRUCE_LIT_DIR SPRUCE_LIT_FILE ALLOW_EXTERNAL)
separate_arguments(SPRUCE_LIT_ARGS NATIVE_COMMAND "${LLVM_LIT_ARGS}")

file(GLOB_RECURSE SPRUCE_MLIR_TESTS CONFIGURE_DEPENDS
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.mlir"
)
list(FILTER SPRUCE_MLIR_TESTS EXCLUDE REGEX "(^|/)Inputs/")

# Keep future sparsification examples as reference inputs, not active tests.
list(REMOVE_ITEM SPRUCE_MLIR_TESTS
  "Conversions/ConvertLinalgToSpruce/spgemm.mlir"
  "Conversions/ConvertLinalgToSpruce/spmv.mlir"
)

# Build-tree paths select the generated lit site configuration.
foreach(test_file IN LISTS SPRUCE_MLIR_TESTS)
  add_test(NAME "spruce/${test_file}"
    COMMAND "${Python3_EXECUTABLE}" "${SPRUCE_LIT_DIR}/${SPRUCE_LIT_FILE}"
      ${SPRUCE_LIT_ARGS} --param "build_mode=$<CONFIG>"
      "${CMAKE_CURRENT_BINARY_DIR}/${test_file}"
  )
endforeach()
