#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Transforms/DialectConversion.h"
#include "spruce/Conversions/Passes.h"
#include "spruce/Dialect/Spruce/IR/SpruceDialect.h"
#include "spruce/Dialect/SpruceGPU/IR/SpruceGPUDialect.h"
#include "spruce/Dialect/SpruceGPU/IR/SpruceGPUTypes.h"

namespace mlir::spruce {
#define GEN_PASS_DEF_CONVERTSPRUCETOSPRUCEGPU
#include "spruce/Conversions/Passes.h.inc"

namespace {

constexpr int64_t tileM = 16;
constexpr int64_t tileN = 8;
constexpr int64_t tileK = 8;

struct MmaOpConversion : OpRewritePattern<MmaOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(MmaOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto ctx = op.getContext();
    auto [m, n, k] = op.getMNK();

    auto fragT = spruce_gpu::FragmentType::get(ctx);
    auto aFrag = spruce_gpu::CreateFragmentOp::create(rewriter, loc, fragT);
    auto bFrag = spruce_gpu::CreateFragmentOp::create(rewriter, loc, fragT);
    auto cFrag = spruce_gpu::CreateFragmentOp::create(rewriter, loc, fragT);
    auto dFrag = spruce_gpu::CreateFragmentOp::create(rewriter, loc, fragT);

    spruce_gpu::MmaLoadMatrixOp::create(rewriter, loc, op.getLhs(), aFrag);
    spruce_gpu::MmaLoadMatrixOp::create(rewriter, loc, op.getRhs(), bFrag);
    spruce_gpu::MmaLoadMatrixOp::create(rewriter, loc, op.getAcc(), cFrag);
    // TODO: Maybe initialize dFrag to zeros?

    spruce_gpu::MmaOp::create(rewriter, loc, dFrag, aFrag, bFrag, cFrag);

    auto result = spruce_gpu::MmaStoreMatrixOp::create(
        rewriter, loc, op.getAcc().getType(), dFrag, op.getAcc());
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConvertSpruceToSpruceGPU
    : impl::ConvertSpruceToSpruceGPUBase<ConvertSpruceToSpruceGPU> {
  using Base::Base;

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<spruce_gpu::SpruceGPUDialect>();
    target.addIllegalOp<spruce::MmaOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<MmaOpConversion>(&getContext());

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
            .failed())
      signalPassFailure();
  }
};

} // namespace
} // namespace mlir::spruce
