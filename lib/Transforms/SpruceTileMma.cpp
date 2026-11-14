#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/MathExtras.h"

#include "spruce/IR/SpruceDialect.h"
#include "spruce/Transforms/MmaUtils.h"
#include "spruce/Transforms/Passes.h"

namespace mlir::spruce {
#define GEN_PASS_DEF_SPRUCETILEMMA
#include "spruce/Transforms/Passes.h.inc"

namespace {

constexpr int64_t tileM = 16;
constexpr int64_t tileN = 8;
constexpr int64_t tileK = 8;

struct TileMmaPattern : OpRewritePattern<MmaOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(MmaOp op,
                                PatternRewriter &rewriter) const override {
    if (!isF32Mma(op))
      return rewriter.notifyMatchFailure(
          op, "currently supports only f32 operands and result");

    auto [m, n, k] = op.getMNK();
    RankedTensorType resultType = op.getResult().getType();

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
      return rewriter.createOrFold<tensor::ExtractSliceOp>(
          op.getLoc(), source,
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(row),
                                 rewriter.getIndexAttr(col)},
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(rows),
                                 rewriter.getIndexAttr(cols)},
          ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1),
                                 rewriter.getIndexAttr(1)});
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
        result = rewriter.createOrFold<tensor::InsertSliceOp>(
            op.getLoc(), acc, result,
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(row),
                                   rewriter.getIndexAttr(col)},
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(tileM),
                                   rewriter.getIndexAttr(tileN)},
            ArrayRef<OpFoldResult>{rewriter.getIndexAttr(1),
                                   rewriter.getIndexAttr(1)});
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
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, tensor::TensorDialect>();
    target.addDynamicallyLegalOp<MmaOp>([](MmaOp op) {
      auto [m, n, k] = getMNK(op.getLhs().getType(), op.getRhs().getType());
      return isF32Mma(op) && m == tileM && n == tileN && k == tileK;
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<TileMmaPattern>(&getContext());

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
            .failed())
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::spruce
