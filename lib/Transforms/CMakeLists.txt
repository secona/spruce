add_mlir_library(SpruceTransforms
  SpruceGPUSparsification.cpp
  SpruceMmaToVector.cpp
  SprucePipeline.cpp
  SpruceSelectTarget.cpp
  SpruceTileMma.cpp

  DEPENDS
  SprucePassesIncGen

  LINK_LIBS PUBLIC
  SpruceDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSparseTensorDialect
  MLIRTensorDialect
  MLIRTransformUtils
  MLIRTransforms
  MLIRVectorDialect
)
