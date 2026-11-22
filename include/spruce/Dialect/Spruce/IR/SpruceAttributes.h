#ifndef SPRUCE_IR_SPRUCEATTRIBUTES_H
#define SPRUCE_IR_SPRUCEATTRIBUTES_H

#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/TensorEncoding.h"
#include "llvm/ADT/Bitset.h"

#include <array>

#define GET_ATTRDEF_CLASSES
#include "spruce/Dialect/Spruce/IR/SpruceAttributes.h.inc"

#endif
