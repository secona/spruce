add_mlir_library(SpruceTransforms
  SpruceLowerBlockOps.cpp
  SpruceTileMma.cpp

  DEPENDS
  SpruceTransformsIncGen

  LINK_LIBS PUBLIC
  SpruceDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRSparseTensorDialect
  MLIRTensorDialect
  MLIRTransforms
)
