add_mlir_library(SpruceTransforms
  SpruceGPUSparsification.cpp
  SpruceLowerBlockOps.cpp
  SpruceMapToGPU.cpp
  SpruceMmaToVector.cpp
  SprucePipeline.cpp
  SpruceTileMma.cpp

  DEPENDS
  SprucePassesIncGen

  LINK_LIBS PUBLIC
  SpruceDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSparseTensorDialect
  MLIRTensorDialect
  MLIRTransformUtils
  MLIRTransforms
  MLIRVectorDialect
)
