#ifndef SPRUCE_TRANSFORMS_PASSES_TD
#define SPRUCE_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def SpruceGPUSparsification : Pass<"spruce-gpu-sparsification", "::mlir::func::FuncOp"> {
  let summary = "spruce-gpu-sparsification";
  let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::tensor::TensorDialect",
                           "::mlir::spruce::SpruceDialect"];
}

def SpruceTileMma : Pass<"spruce-tile-mma", "::mlir::func::FuncOp"> {
  let summary = "spruce-tile-mma";
  let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::tensor::TensorDialect"];
}

def SpruceMmaToVector : Pass<"spruce-mma-to-vector", "::mlir::func::FuncOp"> {
  let summary = "spruce-mma-to-vector";
  let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::vector::VectorDialect"];
}

#endif
