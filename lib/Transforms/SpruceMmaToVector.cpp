#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "spruce/IR/SpruceDialect.h"
#include "spruce/Transforms/MmaUtils.h"
#include "spruce/Transforms/Passes.h"

namespace mlir::spruce {
#define GEN_PASS_DEF_SPRUCEMMATOVECTOR
#include "spruce/Transforms/Passes.h.inc"

namespace {

struct MmaToVectorConversion : OpRewritePattern<MmaOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(MmaOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType lhsType = op.getLhs().getType();
    RankedTensorType rhsType = op.getRhs().getType();
    auto [m, n, k] = getMNK(lhsType, rhsType);

    if (m == 0 || n == 0 || k == 0) {
      rewriter.replaceOp(op, op.getAcc());
      return success();
    }

    Value index = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 0);
    Value zero = arith::ConstantOp::create(rewriter, op.getLoc(),
                                           rewriter.getF32FloatAttr(0.0));
    SmallVector<Value> indices{index, index};
    auto inBounds = rewriter.getBoolArrayAttr({true, true});
    auto read = [&](Value source, VectorType type, AffineMap map) -> Value {
      return vector::TransferReadOp::create(rewriter, op.getLoc(), type, source,
                                            indices, zero,
                                            AffineMapAttr::get(map), inBounds);
    };

    auto identity = rewriter.getMultiDimIdentityMap(2);
    auto transpose = AffineMap::getPermutationMap(ArrayRef<unsigned>{1, 0},
                                                  rewriter.getContext());

    Value lhs = read(op.getLhs(),
                     VectorType::get({m, k}, rewriter.getF32Type()), identity);
    Value rhs = read(op.getRhs(),
                     VectorType::get({n, k}, rewriter.getF32Type()), transpose);
    Value acc = read(op.getAcc(),
                     VectorType::get({m, n}, rewriter.getF32Type()), identity);

    AffineExpr d0, d1, d2;
    bindDims(rewriter.getContext(), d0, d1, d2);

    Value result = vector::ContractionOp::create(
        rewriter, op.getLoc(), lhs, rhs, acc,
        ArrayRef<ArrayRef<AffineExpr>>{{d0, d2}, {d1, d2}, {d0, d1}},
        ArrayRef<vector::IteratorType>{vector::IteratorType::parallel,
                                       vector::IteratorType::parallel,
                                       vector::IteratorType::reduction});

    rewriter.replaceOpWithNewOp<vector::TransferWriteOp>(
        op, result, op.getAcc(), indices, AffineMapAttr::get(identity),
        inBounds);

    return success();
  }
};

struct SpruceMmaToVector : impl::SpruceMmaToVectorBase<SpruceMmaToVector> {
  using Base::Base;

  void runOnOperation() override {
    WalkResult result = getOperation().walk([this](MmaOp op) {
      return failed(verifyF32Mma(op, getArgument())) ? WalkResult::interrupt()
                                                     : WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<MmaToVectorConversion>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::spruce
