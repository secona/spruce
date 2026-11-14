#ifndef SPRUCE_TRANSFORMS_MMAUTILS_H
#define SPRUCE_TRANSFORMS_MMAUTILS_H

#include "mlir/IR/Diagnostics.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <tuple>

#include "spruce/IR/SpruceDialect.h"

namespace mlir::spruce {

inline LogicalResult verifyF32Mma(MmaOp op, llvm::StringRef passName) {
  RankedTensorType lhsType = op.getLhs().getType();
  RankedTensorType rhsType = op.getRhs().getType();
  RankedTensorType accType = op.getAcc().getType();
  RankedTensorType resultType = op.getResult().getType();

  if (!lhsType.getElementType().isF32() || !rhsType.getElementType().isF32() ||
      !accType.getElementType().isF32() || !resultType.getElementType().isF32())
    return op.emitOpError()
           << passName << " currently supports only f32 operands and result";

  return success();
}

inline std::tuple<std::int64_t, std::int64_t, std::int64_t>
getMNK(RankedTensorType lhs, RankedTensorType rhs) {
  int64_t m = lhs.getDimSize(0);
  int64_t k = lhs.getDimSize(1);
  int64_t n = rhs.getDimSize(1);

  return {m, n, k};
}

} // namespace mlir::spruce

#endif
