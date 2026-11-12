#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "spruce/Transforms/Passes.h"

namespace mlir::spruce {
#define GEN_PASS_DEF_SPRUCEGPUSPARSIFICATION
#include "spruce/Transforms/Passes.h.inc"

namespace {

struct LinalgGenericLowering : OpRewritePattern<linalg::GenericOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto tensorT = dyn_cast_or_null<RankedTensorType>(op.getType(0));
    if (!tensorT)
      return failure();

    auto encoding = dyn_cast_or_null<sparse_tensor::SparseTensorEncodingAttr>(
        tensorT.getEncoding());
    if (!encoding)
      return failure();

    return failure();
  }
};

struct SpruceGPUSparsification
    : impl::SpruceGPUSparsificationBase<SpruceGPUSparsification> {
  using Base::Base;

  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<LinalgGenericLowering>(ctx);

    if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed())
      return signalPassFailure();
  }
};

} // namespace
} // namespace mlir::spruce
