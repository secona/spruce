#ifndef SPRUCE_IR_SPRUCEATTRIBUTES_H
#define SPRUCE_IR_SPRUCEATTRIBUTES_H

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/Bitset.h"
#include "llvm/ADT/STLExtras.h"

#include <array>

#define GET_ATTRDEF_CLASSES
#include "spruce/Dialect/Spruce/IR/SpruceAttributes.h.inc"

#endif
