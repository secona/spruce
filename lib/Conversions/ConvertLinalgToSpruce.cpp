#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

#include "spruce/Dialect/Spruce/IR/SpruceAttributes.h"
#include "spruce/Dialect/Spruce/IR/SpruceDialect.h"
#include "spruce/Dialect/Spruce/Transforms/MmaUtils.h"
#include "spruce/Conversions/Passes.h"

namespace mlir::spruce {
#define GEN_PASS_DEF_CONVERTLINALGTOSPRUCE
#include "spruce/Conversions/Passes.h.inc"

namespace {

using namespace sparse_tensor;

struct LinalgGenericLowering : OpRewritePattern<linalg::GenericOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto sparsityAttr = dyn_cast_or_null<SparsityAttr>(op->getAttr("spruce.sparsity"));
    if (!sparsityAttr)
      return failure();

    auto inputs = op.getDpsInputs();
    auto inits = op.getDpsInits();

    // NOTE: Assume every linalg.generic is proper.
    Value a = inputs[0];
    Value b = inputs[1];
    Value c = inits[0];

    auto aT = dyn_cast_or_null<RankedTensorType>(a.getType());
    auto bT = dyn_cast_or_null<RankedTensorType>(b.getType());
    auto cT = dyn_cast_or_null<RankedTensorType>(c.getType());

    Location loc = op.getLoc();
    Value result;

    {
      OpBuilder::InsertionGuard guard(rewriter);

      // Create spruce.kernel while retaining the target.
      auto kernel = spruce::KernelOp::create(rewriter, loc, TypeRange{cT},
                                             ValueRange{a, b, c});
      kernel->setAttr("spruce.target",
                      TargetAttr::get(rewriter.getContext(), 16, 8, 8));

      // Create the mandatory first block of spruce.kernel.
      Block *body = rewriter.createBlock(
          &kernel.getBody(), {}, kernel->getOperandTypes(),
          SmallVector<Location>(kernel.getNumOperands(), loc));

      // Return the linalg.generic outs.
      spruce::YieldOp::create(rewriter, loc, body->getArgument(2));

      result = kernel.getResult(0);
    }

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
