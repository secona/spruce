set(LLVM_TARGET_DEFINITIONS SpruceGPUDialect.td)
mlir_tablegen(SpruceGPUDialect.h.inc -gen-dialect-decls -dialect=spruce_gpu)
mlir_tablegen(SpruceGPUDialect.cpp.inc -gen-dialect-defs -dialect=spruce_gpu)

set(LLVM_TARGET_DEFINITIONS SpruceGPUOps.td)
mlir_tablegen(SpruceGPUOps.h.inc -gen-op-decls)
mlir_tablegen(SpruceGPUOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS SpruceGPUAttributes.td)
mlir_tablegen(SpruceGPUAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=spruce_gpu)
mlir_tablegen(SpruceGPUAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=spruce_gpu)

set(LLVM_TARGET_DEFINITIONS SpruceGPUTypes.td)
mlir_tablegen(SpruceGPUTypes.h.inc -gen-typedef-decls -typedefs-dialect=spruce_gpu)
mlir_tablegen(SpruceGPUTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=spruce_gpu)
add_public_tablegen_target(SpruceGPUDialectIncGen)
