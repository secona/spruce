add_subdirectory(Dialect)
add_subdirectory(Conversions)

add_mlir_library(SprucePipeline
  SprucePipeline.cpp

  LINK_LIBS PUBLIC
  SpruceConversions
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRGPUPipelines
  MLIRLinalgTransforms
  MLIRPass
)
