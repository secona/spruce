#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"

#include "spruce/Dialect/Spruce/IR/SpruceDialect.h"
#include "spruce/Transforms/Passes.h"

#include <string>

namespace mlir::spruce {
#define GEN_PASS_DEF_CONVERTSPRUCETOGPU
#include "spruce/Transforms/Passes.h.inc"

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

struct ConvertSpruceToGPU : impl::ConvertSpruceToGPUBase<ConvertSpruceToGPU> {
  using Base::Base;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    // Find every funcs in the current module.
    for (auto host : module.getOps<func::FuncOp>()) {
      // Find all spruce.kernel ops.
      SmallVector<KernelOp> kernels;
      host.walk([&](KernelOp op) { kernels.push_back(op); });
      if (kernels.empty())
        continue;

      // Rejects spruce.kernel operations that return values.
      for (KernelOp op : kernels) {
        if (op.getNumResults() != 0) {
          op.emitOpError("GPU entry functions must return void");
          signalPassFailure();
          return;
        }
      }

      module->setAttr(gpu::GPUDialect::getContainerModuleAttrName(),
                      rewriter.getUnitAttr());

      // Create the gpu.module op.
      std::string hostName = host.getSymName().str();
      rewriter.setInsertionPoint(host);
      auto gpuModule =
          gpu::GPUModuleOp::create(rewriter, host.getLoc(), hostName + "_gpu");

      // Outline the kernels into the newly created gpu.module op.
      unsigned number = 0;
      for (KernelOp op : kernels) {
        std::string kernelName =
            hostName + "_kernel" + std::to_string(number++);
        outlineKernel(op, gpuModule, kernelName, rewriter);
      }
    }
  }
};

} // namespace
} // namespace mlir::spruce
