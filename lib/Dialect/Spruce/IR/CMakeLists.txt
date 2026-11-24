add_mlir_dialect_library(SpruceDialect
  Bufferization.cpp
  SpruceAttributes.cpp
  SpruceDialect.cpp

  DEPENDS
  SpruceDialectIncGen

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRSparseTensorDialect
)
