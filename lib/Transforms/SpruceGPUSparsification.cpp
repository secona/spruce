#include "spruce/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::spruce {
#define GEN_PASS_DEF_SPRUCEGPUSPARSIFICATION
#include "spruce/Transforms/Passes.h.inc"

namespace {

struct SpruceGPUSparsification
    : impl::SpruceGPUSparsificationBase<SpruceGPUSparsification> {
  using Base::Base;

  void runOnOperation() override {
    // TODO: Lower sparse contractions to GPU Tensor Core tile operations.
    markAllAnalysesPreserved();
  }
};

} // namespace
} // namespace mlir::spruce
