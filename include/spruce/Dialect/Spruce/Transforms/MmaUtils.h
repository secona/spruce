#ifndef SPRUCE_TRANSFORMS_MMAUTILS_H
#define SPRUCE_TRANSFORMS_MMAUTILS_H

#include <cstdint>
#include <tuple>

#include "spruce/Dialect/Spruce/IR/SpruceDialect.h"

namespace mlir::spruce {

inline bool isF32Mma(MmaOp op) {
  ShapedType lhsType = op.getLhs().getType();
  ShapedType rhsType = op.getRhs().getType();
  ShapedType accType = op.getAcc().getType();
  if (!op.getResult())
    return false;
  RankedTensorType resultType = op.getResult().getType();

  return lhsType.getElementType().isF32() && rhsType.getElementType().isF32() &&
         accType.getElementType().isF32() &&
         resultType.getElementType().isF32();
}

inline std::tuple<std::int64_t, std::int64_t, std::int64_t>
getMNK(ShapedType lhs, ShapedType rhs) {
  int64_t m = lhs.getDimSize(0);
  int64_t k = lhs.getDimSize(1);
  int64_t n = rhs.getDimSize(1);

  return {m, n, k};
}

} // namespace mlir::spruce

#endif
