#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/MathExtras.h"

#include "spruce/IR/SpruceDialect.h"
#include "spruce/Transforms/Passes.h"

namespace mlir::spruce {
#define GEN_PASS_DEF_SPRUCETILEMMA
#include "spruce/Transforms/Passes.h.inc"

namespace {

constexpr int64_t tileM = 16;
constexpr int64_t tileN = 8;
constexpr int64_t tileK = 8;

LogicalResult verifyTileableMma(MmaOp op) {
  RankedTensorType lhsType = op.getLhs().getType();
  RankedTensorType rhsType = op.getRhs().getType();
  RankedTensorType accType = op.getAcc().getType();
  RankedTensorType resultType = op.getResult().getType();

  if (!lhsType.getElementType().isF32() || !rhsType.getElementType().isF32() ||
      !accType.getElementType().isF32() || !resultType.getElementType().isF32())
    return op.emitOpError(
        "spruce-tile-mma currently supports only f32 operands "
        "and result");

  return success();
}

struct TileMmaPattern : OpRewritePattern<MmaOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(MmaOp op,
                                PatternRewriter &rewriter) const override {
    RankedTensorType lhsType = op.getLhs().getType();
    RankedTensorType rhsType = op.getRhs().getType();
    RankedTensorType resultType = op.getResult().getType();

    int64_t m = lhsType.getDimSize(0);
    int64_t k = lhsType.getDimSize(1);
    int64_t n = rhsType.getDimSize(1);

    // Newly created MMA tiles must not match this pattern again.
    if (m == tileM && n == tileN && k == tileK)
      return rewriter.notifyMatchFailure(op, "already a supported MMA tile");

    if (m == 0 || n == 0 || k == 0) {
      rewriter.replaceOp(op, op.getAcc());
      return success();
    }

    int64_t paddedM = llvm::alignTo(m, tileM);
    int64_t paddedN = llvm::alignTo(n, tileN);
    int64_t paddedK = llvm::alignTo(k, tileK);

    Value zero;
    auto padToShape = [&](Value source, int64_t rows, int64_t cols) -> Value {
      auto sourceType = cast<RankedTensorType>(source.getType());
      if (sourceType.getDimSize(0) == rows && sourceType.getDimSize(1) == cols)
        return source;
      if (!zero)
        zero = arith::ConstantOp::create(rewriter, op.getLoc(),
                                         rewriter.getF32FloatAttr(0.0));
      return tensor::PadOp::create(
          rewriter, op.getLoc(), sourceType.clone({rows, cols}), source,
          {rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)},
          {rewriter.getIndexAttr(rows - sourceType.getDimSize(0)),
           rewriter.getIndexAttr(cols - sourceType.getDimSize(1))},
          zero);
    };

    Value paddedLhs = padToShape(op.getLhs(), paddedM, paddedK);
    Value paddedRhs = padToShape(op.getRhs(), paddedK, paddedN);
    Value paddedAcc = padToShape(op.getAcc(), paddedM, paddedN);

    auto tileType = resultType.clone({tileM, tileN});
    auto extractTile = [&](Value source, int64_t row, int64_t col, int64_t rows,
                           int64_t cols) -> Value {
      return tensor::ExtractSliceOp::create(
          rewriter, op.getLoc(), source,
          {rewriter.getIndexAttr(row), rewriter.getIndexAttr(col)},
          {rewriter.getIndexAttr(rows), rewriter.getIndexAttr(cols)},
          {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)});
    };

    Value result = paddedAcc;
    for (int64_t row = 0; row < paddedM; row += tileM) {
      for (int64_t col = 0; col < paddedN; col += tileN) {
        Value acc = extractTile(paddedAcc, row, col, tileM, tileN);
        for (int64_t reduction = 0; reduction < paddedK; reduction += tileK) {
          Value lhs = extractTile(paddedLhs, row, reduction, tileM, tileK);
          Value rhs = extractTile(paddedRhs, reduction, col, tileK, tileN);
          acc = MmaOp::create(rewriter, op.getLoc(), tileType, lhs, rhs, acc);
        }
        result = tensor::InsertSliceOp::create(
            rewriter, op.getLoc(), acc, result,
            {rewriter.getIndexAttr(row), rewriter.getIndexAttr(col)},
            {rewriter.getIndexAttr(tileM), rewriter.getIndexAttr(tileN)},
            {rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)});
      }
    }

    if (paddedM != m || paddedN != n)
      result = extractTile(result, 0, 0, m, n);

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct SpruceTileMma : impl::SpruceTileMmaBase<SpruceTileMma> {
  using Base::Base;

  void runOnOperation() override {
    // A pattern match failure does not fail the greedy driver, so validate
    // unsupported inputs before rewriting any operations.
    WalkResult result = getOperation().walk([](MmaOp op) {
      return failed(verifyTileableMma(op)) ? WalkResult::interrupt()
                                           : WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<TileMmaPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::spruce
