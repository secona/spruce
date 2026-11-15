#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "spruce/IR/SpruceDialect.h"
#include "spruce/Transforms/Passes.h"

namespace mlir::spruce {
#define GEN_PASS_DEF_SPRUCELOWERBLOCKOPS
#include "spruce/Transforms/Passes.h.inc"

namespace {

struct NumBlocksLowering : OpRewritePattern<NumBlocksOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(NumBlocksOp op,
                                PatternRewriter &rewriter) const override {
    Value tensor = op.getTensor();
    auto encoding = sparse_tensor::getSparseTensorEncoding(tensor.getType());

    // TODO: support more than BSR
    if (!encoding || encoding.getDimRank() != 2 || encoding.getLvlRank() != 4 ||
        !encoding.isDenseLvl(0) || !encoding.isCompressedLvl(1) ||
        !encoding.isDenseLvl(2) || !encoding.isDenseLvl(3) ||
        encoding.isSlice() ||
        !sparse_tensor::isBlockSparsity(encoding.getDimToLvl()))
      return rewriter.notifyMatchFailure(op, "expected a BSR tensor");

    Location loc = op.getLoc();

    // TODO: not hard code level = 1
    Value positions = sparse_tensor::ToPositionsOp::create(rewriter, loc,
                                                           tensor, /*level=*/1);
    Value numBlockRows = rewriter.createOrFold<sparse_tensor::LvlOp>(
        loc, tensor, /*index=*/int64_t{0});

    Value count = memref::LoadOp::create(rewriter, loc, positions,
                                         ValueRange{numBlockRows});

    if (!count.getType().isIndex())
      count = arith::IndexCastUIOp::create(rewriter, loc,
                                           rewriter.getIndexType(), count);

    rewriter.replaceOp(op, count);
    return success();
  }
};

struct SpruceLowerBlockOps
    : impl::SpruceLowerBlockOpsBase<SpruceLowerBlockOps> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<NumBlocksLowering>(ctx);

    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed())
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::spruce
