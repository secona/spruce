#ifndef SPRUCE_SPRUCEPIPELINE_TD
#define SPRUCE_SPRUCEPIPELINE_TD

include "mlir/Pass/PassBase.td"

def SprucePipeline : Pass<"spruce-pipeline", "::mlir::ModuleOp"> {
  let summary = "spruce-pipeline";
}

#endif
