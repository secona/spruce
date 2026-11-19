#ifndef SPRUCE_DIALECT_SPRUCE_TRANSFORMS_PASSES_H
#define SPRUCE_DIALECT_SPRUCE_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::spruce {
#define GEN_PASS_DECL
#include "spruce/Dialect/Spruce/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "spruce/Dialect/Spruce/Transforms/Passes.h.inc"
} // namespace mlir::spruce

#endif
