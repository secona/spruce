#ifndef SPRUCE_TRANSFORMS_PASSES_TD
#define SPRUCE_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def SpruceGPUSparsification : Pass<"spruce-gpu-sparsification", "::mlir::func::FuncOp"> {
  let summary = "spruce-gpu-sparsification";
}

#endif
