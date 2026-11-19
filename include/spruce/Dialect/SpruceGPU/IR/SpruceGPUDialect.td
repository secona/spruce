#ifndef SPRUCE_GPU_IR_SPRUCEGPUDIALECT_TD
#define SPRUCE_GPU_IR_SPRUCEGPUDIALECT_TD

include "mlir/IR/AttrTypeBase.td"

def SpruceGPU_Dialect : Dialect {
  let name = "spruce_gpu";
  let cppNamespace = "::mlir::spruce_gpu";
}

#endif
