#ifndef SPRUCE_CONVERSIONS_PASSES_TD
#define SPRUCE_CONVERSIONS_PASSES_TD

include "mlir/Pass/PassBase.td"

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
  let options = [
    Option<"mmaTarget", "target", "std::string", "\"mma\"", "">
  ];
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::func::FuncDialect",
    "::mlir::gpu::GPUDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::spruce::SpruceDialect"
  ];
}

#endif
