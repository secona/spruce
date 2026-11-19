add_mlir_dialect_library(SpruceDialect
  SpruceAttributes.cpp
  SpruceDialect.cpp

  DEPENDS
  SpruceDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSparseTensorDialect
)
