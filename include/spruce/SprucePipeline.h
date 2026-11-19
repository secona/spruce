#ifndef SPRUCE_SPRUCEPIPELINE_H
#define SPRUCE_SPRUCEPIPELINE_H

#include "mlir/Pass/Pass.h"

namespace mlir::spruce {
#define GEN_PASS_DECL
#include "spruce/SprucePipeline.h.inc"

#define GEN_PASS_REGISTRATION
#include "spruce/SprucePipeline.h.inc"
} // namespace mlir::spruce

#endif
