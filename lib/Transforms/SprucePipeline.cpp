#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "spruce/Transforms/Passes.h"

namespace mlir::spruce {
#define GEN_PASS_DEF_SPRUCEPIPELINE
#include "spruce/Transforms/Passes.h.inc"

namespace {

void buildPipeline(OpPassManager &pipeline) {
  auto &functions = pipeline.nest<func::FuncOp>();
  functions.addPass(createSpruceGPUSparsification());
  functions.addPass(createSpruceTileMma());
  functions.addPass(createSpruceMmaToVector());
  functions.addPass(createSpruceLowerBlockOps());
  functions.addPass(createCanonicalizerPass());
  functions.addPass(createCSEPass());
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
