add_mlir_library(SpruceTransforms
  SpruceGPUSparsification.cpp

  DEPENDS
  SprucePassesIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRPass
)
