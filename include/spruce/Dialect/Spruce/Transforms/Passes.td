#ifndef SPRUCE_DIALECT_SPRUCE_TRANSFORMS_PASSES_TD
#define SPRUCE_DIALECT_SPRUCE_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def SpruceLowerBlockOps : Pass<"spruce-lower-block-ops", "::mlir::func::FuncOp"> {
  let summary = "spruce-lower-block-ops";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::sparse_tensor::SparseTensorDialect"
  ];
}

def SpruceTileMma : Pass<"spruce-tile-mma", "::mlir::func::FuncOp"> {
  let summary = "spruce-tile-mma";
  let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::tensor::TensorDialect"];
}

#endif
