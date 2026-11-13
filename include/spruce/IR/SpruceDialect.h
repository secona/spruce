#ifndef SPRUCE_IR_SPRUCEDIALECT_H
#define SPRUCE_IR_SPRUCEDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "spruce/IR/SpruceDialect.h.inc"

#define GET_OP_CLASSES
#include "spruce/IR/SpruceOps.h.inc"

#endif
