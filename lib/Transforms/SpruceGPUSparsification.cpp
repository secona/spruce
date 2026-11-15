#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "spruce/IR/SpruceDialect.h"
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

    auto aT = dyn_cast_or_null<RankedTensorType>(a.getType());
    auto bT = dyn_cast_or_null<RankedTensorType>(b.getType());

    if (!aT || !bT)
      return failure();

    auto aEnc = dyn_cast_or_null<SparseTensorEncodingAttr>(aT.getEncoding());
    if (!aEnc)
      return failure();

    auto dim2lvl = aEnc.getDimToLvl();
    if (sparse_tensor::isBlockSparsity(dim2lvl)) {
      if (!isMatmul(op))
        return failure();

      auto loc = op.getLoc();

      Value pos = sparse_tensor::ToPositionsOp::create(rewriter, loc, a, 1);
      Value crd = sparse_tensor::ToCoordinatesOp::create(rewriter, loc, a, 1);
      Value val = sparse_tensor::ToValuesOp::create(rewriter, loc, a);

      Value zero = arith::ConstantIndexOp::create(rewriter, loc, 0);
      Value one = arith::ConstantIndexOp::create(rewriter, loc, 1);

      // scf::ForOp::create(rewriter, op.getLoc());

      auto blockSizes = *getBlockSizes(dim2lvl);
      for (auto size : blockSizes) {
        arith::ConstantOp::create(rewriter, op.getLoc(),
                                  rewriter.getIndexType(),
                                  rewriter.getIndexAttr(size));
      }

      auto acc = tensor::EmptyOp::create(rewriter, op.getLoc(),
                                         {aT.getDimSize(0), bT.getDimSize(1)},
                                         rewriter.getF32Type());
      Type result = op.getResult(0).getType();
      rewriter.replaceOpWithNewOp<MmaOp>(op, result, a, b, acc.getResult());
      return success();
    }

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
