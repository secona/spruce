#ifndef SPRUCE_CONVERSIONS_PASSES_H
#define SPRUCE_CONVERSIONS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir::spruce {
#define GEN_PASS_DECL
#include "spruce/Conversions/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "spruce/Conversions/Passes.h.inc"
} // namespace mlir::spruce

#endif
