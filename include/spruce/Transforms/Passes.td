#ifndef SPRUCE_TRANSFORMS_PASSES_TD
#define SPRUCE_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def SprucePipeline : Pass<"spruce-pipeline", "::mlir::ModuleOp"> {
  let summary = "spruce-pipeline";
}

def ConvertLinalgToSpruce : Pass<"convert-linalg-to-spruce", "::mlir::func::FuncOp"> {
  let summary = "convert-linalg-to-spruce";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::spruce::SpruceDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::sparse_tensor::SparseTensorDialect"
  ];
}

def ConvertSpruceToGPU : Pass<"convert-spruce-to-gpu", "::mlir::ModuleOp"> {
  let summary = "convert-spruce-to-gpu";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::spruce::SpruceDialect"
  ];
}

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

def ConvertSpruceToVector : Pass<"convert-spruce-to-vector", "::mlir::func::FuncOp"> {
  let summary = "convert-spruce-to-vector";
  let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::vector::VectorDialect"];
}

#endif
