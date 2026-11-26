#ifndef SPRUCE_GPU_IR_SPRUCEDIALECT_H
#define SPRUCE_GPU_IR_SPRUCEDIALECT_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"

#include "spruce/Dialect/SpruceGPU/IR/SpruceGPUDialect.h.inc"
#include "spruce/Dialect/SpruceGPU/IR/SpruceGPUTypes.h"

#define GET_OP_CLASSES
#include "spruce/Dialect/SpruceGPU/IR/SpruceGPUOps.h.inc"

#endif
