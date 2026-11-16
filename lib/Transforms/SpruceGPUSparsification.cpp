#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "spruce/IR/SpruceDialect.h"
#include "spruce/Transforms/MmaUtils.h"
#include "spruce/Transforms/Passes.h"

namespace mlir::spruce {
#define GEN_PASS_DEF_SPRUCEGPUSPARSIFICATION
#include "spruce/Transforms/Passes.h.inc"

namespace {

using namespace sparse_tensor;

bool isMatmul(linalg::GenericOp op) {
  auto maps = op.getIndexingMapsArray();
  auto iters = op.getIteratorTypesArray();

  if (maps.size() != 3 || iters.size() != 3)
    return false;

  if (iters[0] != utils::IteratorType::parallel ||
      iters[1] != utils::IteratorType::parallel ||
      iters[2] != utils::IteratorType::reduction)
    return false;

  if (!maps[0].isProjectedPermutation() || !maps[1].isProjectedPermutation() ||
      !maps[2].isProjectedPermutation())
    return false;

  return true;
}

std::optional<llvm::SmallVector<int64_t>> getBlockSizes(AffineMap dimToLvl) {
  llvm::SmallVector<int64_t> sizes(dimToLvl.getNumDims(), 0);
  llvm::SmallVector<int64_t> modSizes(dimToLvl.getNumDims(), 0);

  for (AffineExpr expr : dimToLvl.getResults()) {
    auto bin = dyn_cast<AffineBinaryOpExpr>(expr);
    if (!bin)
      continue;

    auto dim = dyn_cast<AffineDimExpr>(bin.getLHS());
    auto constant = dyn_cast<AffineConstantExpr>(bin.getRHS());
    if (!dim || !constant)
      continue;

    unsigned d = dim.getPosition();
    if (d >= sizes.size())
      continue;

    switch (bin.getKind()) {
    case AffineExprKind::FloorDiv:
      sizes[d] = constant.getValue();
      break;
    case AffineExprKind::Mod:
      modSizes[d] = constant.getValue();
      break;
    default:
      break;
    }
  }

  for (unsigned d = 0; d < sizes.size(); ++d) {
    if (sizes[d] <= 0 || sizes[d] != modSizes[d])
      return std::nullopt;
  }

  return sizes;
}

struct LinalgGenericLowering : OpRewritePattern<linalg::GenericOp> {
  using Base::Base;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    auto inputs = op.getDpsInputs();

    // NOTE: Assume every linalg.generic is proper.
    Value a = inputs[0];
    Value b = inputs[1];
    Value c = op.getDpsInits()[0];

    auto aT = dyn_cast_or_null<RankedTensorType>(a.getType());
    auto bT = dyn_cast_or_null<RankedTensorType>(b.getType());
    auto cT = dyn_cast_or_null<RankedTensorType>(c.getType());

    if (!aT || !bT || !cT)
      return failure();

    if (bT.getEncoding() || cT.getEncoding())
      return failure();

    auto aEnc = dyn_cast_or_null<SparseTensorEncodingAttr>(aT.getEncoding());
    if (!aEnc)
      return failure();

    auto dim2lvl = aEnc.getDimToLvl();
    if (!sparse_tensor::isBlockSparsity(dim2lvl) || !isMatmul(op))
      return failure();

    // TODO: Validate the BSR layout and scalar multiply-add body.
    // For now, assume dense/compressed/dense/dense levels in block-row order.
    auto blockSizes = getBlockSizes(dim2lvl);
    if (!blockSizes)
      return failure();

    int64_t blockM = (*blockSizes)[0];
    int64_t blockK = (*blockSizes)[1];
    auto [m, n, k] = getMNK(aT, bT);

    if (m % blockM != 0 || k % blockK != 0)
      return failure();

    if (m == 0 || n == 0 || k == 0) {
      rewriter.replaceOp(op, c);
      return success();
    }

    Location loc = op.getLoc();
    Value result;

    {
      OpBuilder::InsertionGuard guard(rewriter);

      Value numBlocks =
          NumBlocksOp::create(rewriter, loc, rewriter.getIndexType(), a);

      auto kernel = KernelOp::create(rewriter, loc, TypeRange{cT},
                                     ValueRange{a, b, c, numBlocks});

      // Retain the target attribute from SpruceSelectTarget.
      if (Attribute target = op->getAttr("spruce.target"))
        kernel->setAttr("spruce.target", target);

      Block *body = rewriter.createBlock(
          &kernel.getBody(), {}, kernel->getOperandTypes(),
          SmallVector<Location>(kernel.getNumOperands(), loc));
      a = body->getArgument(0);
      b = body->getArgument(1);
      c = body->getArgument(2);
      numBlocks = body->getArgument(3);

      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);
      Value blockRows = arith::ConstantIndexOp::create(rewriter, loc, blockM);
      Value blockCols = arith::ConstantIndexOp::create(rewriter, loc, blockK);

      auto aBlockType =
          RankedTensorType::get({blockM, blockK}, aT.getElementType());
      auto accType = RankedTensorType::get({blockM, n}, cT.getElementType());
      SmallVector<OpFoldResult> accSizes{rewriter.getIndexAttr(blockM),
                                         rewriter.getIndexAttr(n)};
      SmallVector<OpFoldResult> rhsSizes{rewriter.getIndexAttr(blockK),
                                         rewriter.getIndexAttr(n)};
      SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1),
                                        rewriter.getIndexAttr(1)};

      // Visit stored blocks by ordinal, keeping their storage layout abstract.
      auto blocks = scf::ForOp::create(rewriter, loc, zero, numBlocks, one,
                                       ValueRange{c});
      rewriter.setInsertionPointToStart(blocks.getBody());
      Value p = blocks.getInductionVar();
      auto coordinates =
          BlockCoordinatesOp::create(rewriter, loc, rewriter.getIndexType(),
                                     rewriter.getIndexType(), a, p);
      Value rowOffset =
          arith::MulIOp::create(rewriter, loc, coordinates.getRow(), blockRows);
      SmallVector<OpFoldResult> accOffsets{rowOffset, rewriter.getIndexAttr(0)};
      Value destination = blocks.getRegionIterArgs()[0];
      Value acc = tensor::ExtractSliceOp::create(rewriter, loc, destination,
                                                 accOffsets, accSizes, strides);

      Value lhs = GetBlockOp::create(rewriter, loc, aBlockType, a, p);
      Value rhsOffset =
          arith::MulIOp::create(rewriter, loc, coordinates.getCol(), blockCols);
      Value rhs = tensor::ExtractSliceOp::create(
          rewriter, loc, b,
          ArrayRef<OpFoldResult>{rhsOffset, rewriter.getIndexAttr(0)}, rhsSizes,
          strides);
      Value updated = MmaOp::create(rewriter, loc, accType, lhs, rhs, acc);

      Value updatedDestination = tensor::InsertSliceOp::create(
          rewriter, loc, updated, destination, accOffsets, accSizes, strides);
      scf::YieldOp::create(rewriter, loc, updatedDestination);

      rewriter.setInsertionPointAfter(blocks);
      spruce::YieldOp::create(rewriter, loc, blocks.getResults());
      result = kernel.getResult(0);
    }

    rewriter.replaceOp(op, result);
    return success();
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
