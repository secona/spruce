#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "spruce/IR/SpruceDialect.h"
#include "spruce/Transforms/Passes.h"

namespace mlir::spruce {
#define GEN_PASS_DEF_SPRUCETILEMMA
#include "spruce/Transforms/Passes.h.inc"

namespace {

struct TileMmaPattern : OpRewritePattern<MmaOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(MmaOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType lhsType = op.getLhs().getType();
    RankedTensorType rhsType = op.getRhs().getType();
    RankedTensorType accType = op.getAcc().getType();
    RankedTensorType resultType = op.getResult().getType();

    if (!lhsType.getElementType().isF32() ||
        !rhsType.getElementType().isF32() ||
        !accType.getElementType().isF32() ||
        !resultType.getElementType().isF32()) {
      return op.emitOpError("spruce-tile-mma currently supports only f32 operands "
                            "and result");
    }

    // TODO: tile with multiple tiles
    if (lhsType.getDimSize(0) != 16 || rhsType.getDimSize(1) != 8 ||
        lhsType.getDimSize(1) != 8) {
      return op.emitOpError("spruce-tile-mma currently supports only M = 16, "
                            "N = 8, K = 8; tiling other sizes is not implemented");
    }

    return rewriter.notifyMatchFailure(op, "already a supported MMA tile");
  }
};

struct SpruceTileMma : impl::SpruceTileMmaBase<SpruceTileMma> {
  using Base::Base;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<TileMmaPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::spruce
