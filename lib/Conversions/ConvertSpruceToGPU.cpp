#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/IR/GPUDialectDecl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#include "mlir/Transforms/DialectConversion.h"
#include "spruce/Dialect/Spruce/IR/SpruceDialect.h"
#include "spruce/Conversions/Passes.h"

#include <string>

namespace mlir::spruce {
#define GEN_PASS_DEF_CONVERTSPRUCETOGPU
#include "spruce/Conversions/Passes.h.inc"

namespace {

void outlineKernel(KernelOp op, gpu::GPUModuleOp gpuModule,
                   StringRef kernelName, IRRewriter &rewriter) {
  gpu::GPUFuncOp function;
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(gpuModule.getBody());
    
    // Create GPU kernel function.
    auto type = rewriter.getFunctionType(op->getOperandTypes(), {});
    function = gpu::GPUFuncOp::create(rewriter, op.getLoc(), kernelName, type);
    function.setKernel(true);

    // Retain the target attribute.
    if (Attribute target = op->getAttr("spruce.target"))
      function->setAttr("spruce.target", target);

    // Existing spruce.kernel body.
    rewriter.eraseBlock(&function.getBody().front());
    rewriter.inlineRegionBefore(op.getBody(), function.getBody(),
                                function.getBody().end());

    // Replace spruce.yield with gpu.return.
    auto yield = cast<YieldOp>(function.getBody().front().getTerminator());
    rewriter.setInsertionPoint(yield);
    rewriter.replaceOpWithNewOp<gpu::ReturnOp>(yield);
  }

  // Create kernel dimensions.
  rewriter.setInsertionPoint(op);
  // TODO: Maybe not 1 for every dimension?
  Value one = arith::ConstantIndexOp::create(rewriter, op.getLoc(), 1);
  gpu::KernelDim3 dimensions{one, one, one};

  // Actually replace with gpu.launch_func op.
  rewriter.replaceOpWithNewOp<gpu::LaunchFuncOp>(
      op, function, dimensions, dimensions, Value(), op.getInputs());
}

struct MmaOpConversion : OpRewritePattern<MmaOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(MmaOp op,
                                PatternRewriter &rewriter) const override {
    return failure();
  }
};

struct ConvertSpruceToGPU : impl::ConvertSpruceToGPUBase<ConvertSpruceToGPU> {
  using Base::Base;

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<gpu::GPUDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<MmaOpConversion>(&getContext());

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
          .failed())
      return signalPassFailure();
  }
};

} // namespace
} // namespace mlir::spruce
