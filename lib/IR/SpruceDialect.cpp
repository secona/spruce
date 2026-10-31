#include "spruce/IR/SpruceDialect.h"
#include "spruce/IR/SpruceAttributes.h"

#include "spruce/IR/SpruceDialect.cpp.inc"

void mlir::spruce::SpruceDialect::initialize() { registerAttributes(); }
