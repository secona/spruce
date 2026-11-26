add_subdirectory(Dialect)
add_subdirectory(Conversions)
add_subdirectory(Pipelines)

add_library(Spruce Spruce.cpp)
target_include_directories(Spruce PUBLIC "${PROJECT_SOURCE_DIR}/include")
target_link_libraries(Spruce PRIVATE
  adlet
  SprucePipeline
  MLIRParser
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRToLLVMIRTranslationRegistration
)
