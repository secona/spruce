set(LLVM_TARGET_DEFINITIONS SpruceDialect.td)
mlir_tablegen(SpruceDialect.h.inc -gen-dialect-decls -dialect=spruce)
mlir_tablegen(SpruceDialect.cpp.inc -gen-dialect-defs -dialect=spruce)

set(LLVM_TARGET_DEFINITIONS SpruceOps.td)
mlir_tablegen(SpruceOps.h.inc -gen-op-decls)
mlir_tablegen(SpruceOps.cpp.inc -gen-op-defs)

set(LLVM_TARGET_DEFINITIONS SpruceAttributes.td)
mlir_tablegen(SpruceAttributes.h.inc -gen-attrdef-decls -attrdefs-dialect=spruce)
mlir_tablegen(SpruceAttributes.cpp.inc -gen-attrdef-defs -attrdefs-dialect=spruce)
add_public_tablegen_target(SpruceDialectIncGen)
