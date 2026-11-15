#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "spruce/IR/SpruceAttributes.h"
#include "spruce/IR/SpruceDialect.h"
#include "spruce/Transforms/Passes.h"

namespace mlir::spruce {
#define GEN_PASS_DEF_SPRUCESELECTTARGET
#include "spruce/Transforms/Passes.h.inc"

namespace {

struct SelectTargetPattern : OpRewritePattern<linalg::GenericOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    if (op->hasAttr("spruce.target") || !linalg::isaContractionOpInterface(op))
      return failure();

    for (Type type : op.getOperandTypes()) {
      auto tensorType = dyn_cast<RankedTensorType>(type);
      if (!tensorType || tensorType.getRank() != 2)
        return failure();
    }

    // Prototype: use the existing MMA shape until target-specific selection
    // considers architecture, operand types, and precision requirements.
    auto target = TargetAttr::get(rewriter.getContext(), 16, 8, 8);
    rewriter.modifyOpInPlace(op, [&] { op->setAttr("spruce.target", target); });
    return success();
  }
};

struct SpruceSelectTarget : impl::SpruceSelectTargetBase<SpruceSelectTarget> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<SelectTargetPattern>(ctx);

    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::spruce
