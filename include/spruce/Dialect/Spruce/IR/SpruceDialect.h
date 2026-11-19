#ifndef SPRUCE_IR_SPRUCEDIALECT_H
#define SPRUCE_IR_SPRUCEDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "spruce/Dialect/Spruce/IR/SpruceDialect.h.inc"
#include "spruce/Dialect/Spruce/IR/SpruceTypes.h"

#define GET_OP_CLASSES
#include "spruce/Dialect/Spruce/IR/SpruceOps.h.inc"

#endif
