#ifndef SPRUCE_DIALECT_SPRUCE_TRANSFORMS_PASSES_TD
#define SPRUCE_DIALECT_SPRUCE_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def SpruceTileMma : Pass<"spruce-tile-mma", "::mlir::func::FuncOp"> {
  let summary = "spruce-tile-mma";
  let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::tensor::TensorDialect"];
}

#endif
