add_mlir_library(SpruceTransforms
  SpruceGPUSparsification.cpp
  SpruceTileMma.cpp

  DEPENDS
  SprucePassesIncGen

  LINK_LIBS PUBLIC
  SpruceDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransformUtils
)
