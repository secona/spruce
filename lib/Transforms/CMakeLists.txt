add_mlir_library(SpruceTransforms
  ConvertLinalgToSpruce.cpp
  ConvertSpruceToGPU.cpp
  ConvertSpruceToVector.cpp
  SpruceLowerBlockOps.cpp
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
