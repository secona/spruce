#ifndef SPRUCE_SPRUCEPIPELINE_H
#define SPRUCE_SPRUCEPIPELINE_H

#include "mlir/Pass/PassOptions.h"

namespace mlir::spruce {

struct SprucePipelineOptions : PassPipelineOptions<SprucePipelineOptions> {
  PassOptions::Option<std::string> chip{
      *this,
      "chip",
      llvm::cl::desc("CUDA compilation target"),
      llvm::cl::init("sm_86"),
  };

  PassOptions::Option<std::string> mmaTarget{
      *this,
      "mma-target",
      llvm::cl::desc("PTX MMA instruction target"),
      llvm::cl::init("wmma"),
  };
};

void registerSprucePipeline();

} // namespace mlir::spruce

#endif
