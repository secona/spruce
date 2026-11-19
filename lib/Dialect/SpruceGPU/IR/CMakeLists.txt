add_mlir_dialect_library(SpruceGPUDialect
  SpruceGPUDialect.cpp

  DEPENDS
  SpruceGPUDialectIncGen

  LINK_LIBS PUBLIC
  MLIRIR
)
