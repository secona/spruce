#include "spruce/Dialect/SpruceGPU/IR/SpruceGPUDialect.h"

#include "spruce/Dialect/SpruceGPU/IR/SpruceGPUDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "spruce/Dialect/SpruceGPU/IR/SpruceGPUTypes.cpp.inc"

#define GET_OP_CLASSES
#include "spruce/Dialect/SpruceGPU/IR/SpruceGPUOps.cpp.inc"

void mlir::spruce_gpu::SpruceGPUDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "spruce/Dialect/SpruceGPU/IR/SpruceGPUTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "spruce/Dialect/SpruceGPU/IR/SpruceGPUOps.cpp.inc"
      >();
}
