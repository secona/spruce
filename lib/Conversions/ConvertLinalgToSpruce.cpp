#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Bitset.h"
#include "llvm/ADT/STLExtras.h"

#include "spruce/Conversions/Passes.h"
#include "spruce/Dialect/Spruce/IR/SpruceAttributes.h"
#include "spruce/Dialect/Spruce/IR/SpruceDialect.h"
#include "spruce/Dialect/Spruce/Transforms/MmaUtils.h"

namespace mlir::spruce {
#define GEN_PASS_DEF_CONVERTLINALGTOSPRUCE
#include "spruce/Conversions/Passes.h.inc"

namespace {

constexpr int64_t tileM = 16;
constexpr int64_t tileN = 8;
constexpr int64_t tileK = 8;

FailureOr<SmallVector<int64_t>>
getActiveTileOffsets(DimMaskAttr dim, int64_t extent, int64_t tileSize) {
  llvm::Bitset<2048> mask = dim.getBitsetMap();

  SmallVector<int64_t> offsets;
  for (int64_t base = 0; base < extent; base += tileSize) {
    if (llvm::any_of(llvm::seq(base, base + tileSize),
                     [&](int64_t i) { return mask.test(i); }))
      offsets.push_back(base);
  }
  return offsets;
}

Value buildActiveTiles(OpBuilder &builder, Location loc, Value lhs, Value rhs,
                       Value init, ArrayRef<int64_t> rows,
                       ArrayRef<int64_t> cols, int64_t k) {
  if (rows.empty() || cols.empty() || k == 0)
    return init;

  MLIRContext *ctx = builder.getContext();

  auto lhsSize = getAsIndexOpFoldResult(ctx, {tileM, tileK});
  auto rhsSize = getAsIndexOpFoldResult(ctx, {tileK, tileN});
  auto accSize = getAsIndexOpFoldResult(ctx, {tileM, tileN});
  auto strides = getAsIndexOpFoldResult(ctx, {1, 1});

  Value zero = arith::ConstantIndexOp::create(builder, loc, 0);
  Value end = arith::ConstantIndexOp::create(builder, loc, k);
  Value step = arith::ConstantIndexOp::create(builder, loc, tileK);

  Value result = init;
  for (int64_t row : rows) {
    for (int64_t col : cols) {
      auto offsets = getAsIndexOpFoldResult(ctx, {row, col});
      Value acc = builder.createOrFold<tensor::ExtractSliceOp>(
          loc, init, offsets, accSize, strides);
      auto loop = scf::ForOp::create(
          builder, loc, zero, end, step, ValueRange{acc},
          [&](OpBuilder &b, Location loc, Value reduction,
              ValueRange iterArgs) {
            Value lhsTile = b.createOrFold<tensor::ExtractSliceOp>(
                loc, lhs, ArrayRef<OpFoldResult>{offsets[0], reduction},
                lhsSize, strides);
            Value rhsTile = b.createOrFold<tensor::ExtractSliceOp>(
                loc, rhs, ArrayRef<OpFoldResult>{reduction, offsets[1]},
                rhsSize, strides);
            auto kernel = KernelOp::create(
                b, loc, TypeRange{acc.getType()},
                ValueRange{lhsTile, rhsTile, iterArgs.front()});
            {
              OpBuilder::InsertionGuard guard(b);
              Block *body = b.createBlock(
                  &kernel.getBody(), {}, kernel->getOperandTypes(),
                  SmallVector<Location>(kernel.getNumOperands(), loc));
              Value sum =
                  MmaOp::create(b, loc, acc.getType(), body->getArgument(0),
                                body->getArgument(1), body->getArgument(2))
                      .getResult();
              YieldOp::create(b, loc, sum);
            }
            scf::YieldOp::create(b, loc, kernel.getResult(0));
          });
      result = builder.createOrFold<tensor::InsertSliceOp>(
          loc, loop.getResult(0), result, offsets, accSize, strides);
    }
  }

  return result;
}

struct LinalgGenericLowering : OpRewritePattern<linalg::GenericOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto sparsityAttr = op->getAttrOfType<SparsityAttr>("spruce.sparsity");
    if (!sparsityAttr)
      return failure();

    if (!op.hasPureTensorSemantics() ||
        !isRowMajorMatmul(op.getIndexingMaps()) ||
        !linalg::isaContractionOpInterface(op) ||
        op.getBody()->getOperations().size() != 3)
      return rewriter.notifyMatchFailure(op,
                                         "expected a row-major tensor matmul");

    Value a = op.getDpsInputOperand(0)->get();
    Value b = op.getDpsInputOperand(1)->get();
    Value c = op.getDpsInitOperand(0)->get();

    auto aT = cast<RankedTensorType>(a.getType());
    auto bT = cast<RankedTensorType>(b.getType());
    auto cT = cast<RankedTensorType>(c.getType());

    if (!aT.hasStaticShape() || !bT.hasStaticShape() || !cT.hasStaticShape() ||
        !isa<FloatType>(cT.getElementType()) ||
        aT.getElementType() != cT.getElementType() ||
        bT.getElementType() != cT.getElementType())
      return rewriter.notifyMatchFailure(
          op, "expected static floating-point matrices");

    auto [m, n, k] = getMNK(aT, bT);

    auto dims = sparsityAttr.getDims();
    auto rows = getActiveTileOffsets(dims[0], m, tileM);
    auto cols = getActiveTileOffsets(dims[1], n, tileN);

    Value result = buildActiveTiles(rewriter, op.getLoc(), a, b, c, *rows,
                                    *cols, k);
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertLinalgToSpruce
    : impl::ConvertLinalgToSpruceBase<ConvertLinalgToSpruce> {
  using Base::Base;

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect,
                           tensor::TensorDialect, SpruceDialect>();
    target.addIllegalOp<linalg::GenericOp>();

    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LinalgGenericLowering>(ctx);

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
            .failed())
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::spruce
