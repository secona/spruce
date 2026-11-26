add_mlir_library(SpruceTransforms
  SpruceTileMma.cpp

  DEPENDS
  SpruceTransformsIncGen

  LINK_LIBS PUBLIC
  SpruceDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
)
