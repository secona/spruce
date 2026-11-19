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
#include "spruce/Transforms/MmaUtils.h"
#include "spruce/Transforms/Passes.h"

namespace mlir::spruce {
#define GEN_PASS_DEF_CONVERTLINALGTOSPRUCE
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

// Get the block sizes from dim2lvl map of the sparse tensor encoding. It works
// by looping each level and checking for `floordiv` and `mod` kinds.
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
    auto inits = op.getDpsInits();

    // NOTE: Assume every linalg.generic is proper.
    Value a = inputs[0];
    Value b = inputs[1];
    Value c = inits[0];

    auto aT = dyn_cast_or_null<RankedTensorType>(a.getType());
    auto bT = dyn_cast_or_null<RankedTensorType>(b.getType());
    auto cT = dyn_cast_or_null<RankedTensorType>(c.getType());

    // Only ranked tensors.
    if (!aT || !bT || !cT)
      return failure();

    // TODO: sparse B and C matrices.
    if (bT.getEncoding() || cT.getEncoding())
      return failure();

    // Only accept if A is sparse.
    auto aEnc = dyn_cast_or_null<SparseTensorEncodingAttr>(aT.getEncoding());
    if (!aEnc)
      return failure();

    // TODO: Support more than just block matmul.
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

    // TODO: Accept matrices that are not multiples of its blocks.
    if (m % blockM != 0 || k % blockK != 0)
      return failure();

    // Edge case: Rank-0 matrix.
    if (m == 0 || n == 0 || k == 0) {
      rewriter.replaceOp(op, c);
      return success();
    }

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
