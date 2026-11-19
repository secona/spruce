#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "spruce/Dialect/Spruce/IR/SpruceDialect.h"
#include "spruce/Transforms/Passes.h"

namespace mlir::spruce {
#define GEN_PASS_DEF_SPRUCELOWERBLOCKOPS
#include "spruce/Transforms/Passes.h.inc"

namespace {

struct SpruceLowerBlockOps
    : impl::SpruceLowerBlockOpsBase<SpruceLowerBlockOps> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::spruce
