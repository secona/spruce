add_executable(spruce spruce.cpp)
set_target_properties(spruce PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${LLVM_RUNTIME_OUTPUT_INTDIR}")

target_link_libraries(spruce PRIVATE
  LLVMSupport
  MLIRIR
  spruce-adlet
  taco
)
