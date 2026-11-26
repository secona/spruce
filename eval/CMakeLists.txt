add_executable(spruce-spgemm spgemm.cpp)
set_target_properties(spruce-spgemm PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${LLVM_RUNTIME_OUTPUT_INTDIR}")
target_link_libraries(spruce-spgemm PRIVATE Spruce)
