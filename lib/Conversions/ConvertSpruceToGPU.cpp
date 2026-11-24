#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"
#include "spruce/Conversions/Passes.h"
#include "spruce/Dialect/Spruce/IR/SpruceDialect.h"

namespace mlir::spruce {
#define GEN_PASS_DEF_CONVERTSPRUCETOGPU
#include "spruce/Conversions/Passes.h.inc"

namespace {

struct MmaOpConversion : OpRewritePattern<MmaOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(MmaOp op,
                                PatternRewriter &rewriter) const override {
    auto launch = dyn_cast<gpu::LaunchOp>(op->getParentOp());
    if (!launch || op.getResult() || op.getMNK() != std::make_tuple(16, 8, 8) ||
        !op.getLhs().getType().getElementType().isF16() ||
        !op.getRhs().getType().getElementType().isF16() ||
        !op.getAcc().getType().getElementType().isF16())
      return rewriter.notifyMatchFailure(
          op, "expected a bufferized f16 16x8x8 MMA in gpu.launch");
    auto block = launch.getBlockSizeOperandValues();
    if (getConstantIntValue(block.x) != 32 ||
        getConstantIntValue(block.y) != 1 || getConstantIntValue(block.z) != 1)
      return rewriter.notifyMatchFailure(op, "expected a 32x1x1 thread block");

    Location loc = op.getLoc();
    Value lane = launch.getThreadIds().x;
    Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
    Value width = arith::ConstantIndexOp::create(rewriter, loc, 8);
    Value firstRow = arith::DivUIOp::create(rewriter, loc, lane, width);
    Value col = arith::RemUIOp::create(rewriter, loc, lane, width);

    // Eight lanes copy one row, so the warp covers four rows per step.
    // Scalar copies respect dynamic strides on the incoming subviews.
    auto copyTile = [&](Value source, Value dest) {
      int64_t rows = cast<MemRefType>(source.getType()).getDimSize(0);
      Value row = firstRow;
      for (int64_t i = 0; i < rows; i += 4) {
        if (i) {
          Value offset = arith::ConstantIndexOp::create(rewriter, loc, i);
          row = arith::AddIOp::create(rewriter, loc, firstRow, offset);
        }
        Value value =
            memref::LoadOp::create(rewriter, loc, source, ValueRange{row, col});
        memref::StoreOp::create(rewriter, loc, value, dest,
                                ValueRange{row, col});
      }
    };

    // The subgroup load/store ops need unit inner stride and a static
    // leading dimension. Stage C as well as A/B to support strided C tiles.
    auto stage = [&](Value source) {
      auto type = cast<MemRefType>(source.getType());
      auto space = gpu::AddressSpaceAttr::get(op.getContext(),
                                              gpu::AddressSpace::Workgroup);
      auto sharedType = MemRefType::get(type.getShape(), type.getElementType(),
                                        MemRefLayoutAttrInterface{}, space);
      Value shared;
      rewriter.modifyOpInPlace(launch, [&] {
        shared = launch.addWorkgroupAttribution(sharedType, loc);
      });
      copyTile(source, shared);
      return shared;
    };
    Value sharedA = stage(op.getLhs());
    Value sharedB = stage(op.getRhs());
    Value sharedC = stage(op.getAcc());
    gpu::BarrierOp::create(rewriter, loc);

    auto load = [&](Value tile, StringRef role) {
      auto type = cast<MemRefType>(tile.getType());
      auto fragment =
          gpu::MMAMatrixType::get(type.getShape(), type.getElementType(), role);
      return gpu::SubgroupMmaLoadMatrixOp::create(
                 rewriter, loc, fragment, tile, ValueRange{zero, zero},
                 rewriter.getIndexAttr(type.getDimSize(1)), UnitAttr())
          .getResult();
    };
    Value a = load(sharedA, "AOp");
    Value b = load(sharedB, "BOp");
    Value c = load(sharedC, "COp");
    Value sum = gpu::SubgroupMmaComputeOp::create(rewriter, loc, c.getType(), a,
                                                  b, c, UnitAttr(), UnitAttr());
    gpu::SubgroupMmaStoreMatrixOp::create(rewriter, loc, sum, sharedC,
                                          ValueRange{zero, zero},
                                          rewriter.getIndexAttr(8), UnitAttr());
    gpu::BarrierOp::create(rewriter, loc);
    copyTile(sharedC, op.getAcc());
    rewriter.eraseOp(op);
    return success();
  }
};

struct KernelOpConversion : OpRewritePattern<KernelOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(KernelOp op,
                                PatternRewriter &rewriter) const override {
    Block &body = op.getBody().front();
    auto yield = cast<YieldOp>(body.getTerminator());
    if (llvm::any_of(op.getOperandTypes(),
                     [](Type t) { return isa<TensorType>(t); }))
      return rewriter.notifyMatchFailure(op,
                                         "expected bufferized kernel inputs");
    SmallVector<Value> results;
    for (Value value : yield.getValues()) {
      auto arg = dyn_cast<BlockArgument>(value);
      if (!arg || arg.getOwner() != &body || !isa<MemRefType>(arg.getType()))
        return rewriter.notifyMatchFailure(
            op, "expected a forwarded buffer argument");
      results.push_back(op.getOperand(arg.getArgNumber()));
    }
    // Only the tile MMA has a warp-level execution mapping. Other bodies
    // retain single-thread execution.
    bool hasMma = !body.getOps<MmaOp>().empty();
    if (hasMma && body.getOperations().size() != 2)
      return rewriter.notifyMatchFailure(op,
                                         "expected a single MMA tile kernel");

    Location loc = op.getLoc();
    Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
    Value threads =
        hasMma ? arith::ConstantIndexOp::create(rewriter, loc, 32).getResult()
               : one;
    auto launch =
        gpu::LaunchOp::create(rewriter, loc, one, one, one, threads, one, one);
    if (Attribute target = op->getAttr("spruce.target"))
      launch->setAttr("spruce.target", target);
    IRMapping mapping;
    mapping.map(body.getArguments(), op.getInputs());
    rewriter.setInsertionPointToEnd(&launch.getBody().front());
    for (Operation &nested : body.without_terminator())
      rewriter.clone(nested, mapping);
    gpu::TerminatorOp::create(rewriter, loc);
    rewriter.replaceOp(op, results);
    return success();
  }
};

struct ConvertSpruceToGPU : impl::ConvertSpruceToGPUBase<ConvertSpruceToGPU> {
  using Base::Base;

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, gpu::GPUDialect,
                           memref::MemRefDialect>();
    target.addIllegalOp<KernelOp, MmaOp, YieldOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<KernelOpConversion, MmaOpConversion>(&getContext());

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
            .failed())
      return signalPassFailure();
  }
};

} // namespace
} // namespace mlir::spruce
