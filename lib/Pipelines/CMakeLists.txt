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
