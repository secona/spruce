add_mlir_library(SpruceConversions
  ConvertLinalgToSpruce.cpp
  ConvertSpruceGPUToNVGPU.cpp
  ConvertSpruceToGPU.cpp
  ConvertSpruceToSpruceGPU.cpp
  ConvertSpruceToVector.cpp

  DEPENDS
  SpruceConversionsIncGen

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
