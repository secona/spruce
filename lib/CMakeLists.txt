add_subdirectory(Dialect)
add_subdirectory(Conversions)
add_subdirectory(Pipelines)

add_library(Spruce Spruce.cpp)
find_library(SPRUCE_CUDA_RUNTIME_LIBRARY NAMES mlir_cuda_runtime
  HINTS ${LLVM_LIBRARY_DIRS} DOC "MLIR CUDA runtime for JIT-compiled kernels")
if(NOT SPRUCE_CUDA_RUNTIME_LIBRARY)
  set(SPRUCE_CUDA_RUNTIME_LIBRARY
    "${CMAKE_SHARED_LIBRARY_PREFIX}mlir_cuda_runtime${CMAKE_SHARED_LIBRARY_SUFFIX}")
endif()
target_compile_definitions(Spruce PRIVATE
  SPRUCE_CUDA_RUNTIME_LIBRARY="${SPRUCE_CUDA_RUNTIME_LIBRARY}")
target_include_directories(Spruce PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(Spruce PRIVATE
  adlet
  SprucePipeline
  MLIRExecutionEngine
  MLIRParser
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRToLLVMIRTranslationRegistration
)
