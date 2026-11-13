#include "spruce/IR/SpruceDialect.h"
#include "spruce/IR/SpruceAttributes.h"

#include "spruce/IR/SpruceDialect.cpp.inc"

#define GET_OP_CLASSES
#include "spruce/IR/SpruceOps.cpp.inc"

namespace mlir {
namespace spruce {

// -----------------------------------------------------------------------------
// SpruceDialect
// -----------------------------------------------------------------------------

void SpruceDialect::initialize() {
  registerAttributes();

  addOperations<
#define GET_OP_LIST
#include "spruce/IR/SpruceOps.cpp.inc"
      >();
}

// -----------------------------------------------------------------------------
// MmaOp
// -----------------------------------------------------------------------------

LogicalResult MmaOp::verify() {
  RankedTensorType lhsType = getLhs().getType();
  RankedTensorType rhsType = getRhs().getType();
  RankedTensorType accType = getAcc().getType();
  RankedTensorType resultType = getResult().getType();

  if (lhsType.getRank() != 2)
    return emitOpError("expected lhs to be a rank-2 tensor");
  if (rhsType.getRank() != 2)
    return emitOpError("expected rhs to be a rank-2 tensor");
  if (accType.getRank() != 2)
    return emitOpError("expected acc to be a rank-2 tensor");

  if (resultType.getShape() != accType.getShape())
    return emitOpError("expected result and acc to have the same shape");

  if (!lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      !accType.hasStaticShape())
    return emitOpError("expected statically shaped operands");

  if (lhsType.getDimSize(1) != rhsType.getDimSize(0))
    return emitOpError("expected lhs dimension 1 and rhs dimension 0 to match "
                       "(reduction dimension K)");
  if (lhsType.getDimSize(0) != accType.getDimSize(0))
    return emitOpError("expected lhs dimension 0 and acc dimension 0 to match "
                       "(dimension M)");
  if (rhsType.getDimSize(1) != accType.getDimSize(1))
    return emitOpError("expected rhs dimension 1 and acc dimension 1 to match "
                       "(dimension N)");

  return success();
}

} // namespace spruce
} // namespace mlir
