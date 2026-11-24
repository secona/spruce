#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Pipelines/Passes.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"

#include "spruce/Conversions/Passes.h"
#include "spruce/SprucePipeline.h"

namespace mlir::spruce {
#define GEN_PASS_DEF_SPRUCEPIPELINE
#include "spruce/SprucePipeline.h.inc"

namespace {

void buildPipeline(OpPassManager &pipeline) {
  pipeline.addNestedPass<func::FuncOp>(createConvertLinalgToSpruce());

  bufferization::OneShotBufferizePassOptions bufferizeOptions;
  bufferizeOptions.bufferizeFunctionBoundaries = true;
  pipeline.addPass(bufferization::createOneShotBufferizePass(bufferizeOptions));

  ConvertSpruceToGPUOptions gpuOptions;
  gpuOptions.mmaTarget = "wmma";
  pipeline.addPass(createConvertSpruceToGPU(gpuOptions));
  pipeline.addPass(createConvertLinalgToLoopsPass());

  gpu::GPUToNVVMPipelineOptions nvvmOptions;
  nvvmOptions.cubinChip = "sm_80";
  nvvmOptions.cubinFormat = "bin";
  gpu::buildLowerToNVVMPassPipeline(pipeline, nvvmOptions);
}

struct SprucePipeline : impl::SprucePipelineBase<SprucePipeline> {
  using Base::Base;

  void getDependentDialects(DialectRegistry &registry) const override {
    OpPassManager pipeline(ModuleOp::getOperationName());
    buildPipeline(pipeline);
    pipeline.getDependentDialects(registry);
  }

  void runOnOperation() override {
    OpPassManager pipeline(ModuleOp::getOperationName());
    buildPipeline(pipeline);

    if (failed(runPipeline(pipeline, getOperation())))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::spruce
