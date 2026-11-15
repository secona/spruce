#ifndef SPRUCE_TRANSFORMS_PASSES_TD
#define SPRUCE_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def SprucePipeline : Pass<"spruce-pipeline", "::mlir::ModuleOp"> {
  let summary = "spruce-pipeline";
}

def SpruceSelectTarget : Pass<"spruce-select-target", "::mlir::func::FuncOp"> {
  let summary = "Select a GPU execution tile configuration";
  let description = [{
    Annotates rank-2 tensor contractions expressed as linalg.generic operations
    with a spruce.target attribute containing the selected MMA dimensions.
    This prototype selects M=16, N=8, K=8, matching the existing MMA tiling pass.
    Existing target annotations, sparse storage encodings, and block shapes are
    preserved; subsequent passes can use the annotation to construct execution
    tiles.
  }];
  let dependentDialects = ["::mlir::spruce::SpruceDialect"];
}

def SpruceGPUSparsification : Pass<"spruce-gpu-sparsification", "::mlir::func::FuncOp"> {
  let summary = "spruce-gpu-sparsification";
  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::tensor::TensorDialect",
    "::mlir::spruce::SpruceDialect",
    "::mlir::scf::SCFDialect",
    "::mlir::sparse_tensor::SparseTensorDialect"
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

def SpruceMmaToVector : Pass<"spruce-mma-to-vector", "::mlir::func::FuncOp"> {
  let summary = "spruce-mma-to-vector";
  let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::vector::VectorDialect"];
}

#endif
