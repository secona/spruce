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

def ConvertSpruceGPUToNVGPU : Pass<"convert-spruce-gpu-to-nvgpu"> {
  let summary = "Lower SpruceGPU operations to NVGPU (skeleton)";
  let dependentDialects = ["::mlir::nvgpu::NVGPUDialect"];
}

def ConvertSpruceGPUToNVVM : Pass<"convert-spruce-gpu-to-nvvm"> {
  let summary = "Lower SpruceGPU operations to NVVM (skeleton)";
  let dependentDialects = ["::mlir::NVVM::NVVMDialect"];
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

def ConvertSpruceToSpruceGPU : Pass<"convert-spruce-to-spruce-gpu"> {
  let summary = "convert-spruce-to-spruce-gpu";
  let dependentDialects = [
    "::mlir::spruce_gpu::SpruceGPUDialect",
    "::mlir::spruce::SpruceDialect"
  ];
}

def ConvertSpruceToVector : Pass<"convert-spruce-to-vector", "::mlir::func::FuncOp"> {
  let summary = "convert-spruce-to-vector";
  let dependentDialects = ["::mlir::arith::ArithDialect", "::mlir::vector::VectorDialect"];
}

#endif
