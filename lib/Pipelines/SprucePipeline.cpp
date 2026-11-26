#include "spruce/Pipelines/SprucePipeline.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "spruce/Conversions/Passes.h"

namespace mlir::spruce {

namespace {

void buildSprucePipeline(OpPassManager &pm,
                         const SprucePipelineOptions &options) {
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToSpruce());

  bufferization::OneShotBufferizePassOptions bufferizeOptions;
  bufferizeOptions.bufferizeFunctionBoundaries = true;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizeOptions));

  ConvertSpruceToGPUOptions gpuOptions;
  gpuOptions.mmaTarget = options.mmaTarget;
  pm.addPass(createConvertSpruceToGPU(gpuOptions));
  pm.addPass(createConvertLinalgToLoopsPass());

  gpu::GPUToNVVMPipelineOptions nvvmOptions;
  nvvmOptions.cubinChip = options.chip;
  nvvmOptions.cubinFormat = "bin";
  gpu::buildLowerToNVVMPassPipeline(pm, nvvmOptions);
}

} // namespace

void registerSprucePipeline() {
  PassPipelineRegistration<SprucePipelineOptions>("spruce-pipeline", "",
                                                  buildSprucePipeline);
}

} // namespace mlir::spruce
