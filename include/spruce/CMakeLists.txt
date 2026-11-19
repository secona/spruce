add_subdirectory(Dialect)
add_subdirectory(Conversions)

set(LLVM_TARGET_DEFINITIONS SprucePipeline.td)
mlir_tablegen(SprucePipeline.h.inc -gen-pass-decls -name SprucePipelines)
add_public_tablegen_target(SprucePipelinesIncGen)
