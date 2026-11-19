#include "spruce/IR/SpruceDialect.h"
#include "spruce/IR/SpruceAttributes.h"

#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/STLExtras.h"

#include "spruce/IR/SpruceDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "spruce/IR/SpruceTypes.cpp.inc"

namespace mlir::spruce {

static ParseResult
parseKernelBody(OpAsmParser &parser,
                SmallVectorImpl<OpAsmParser::UnresolvedOperand> &inputs,
                SmallVectorImpl<Type> &types, Region &body) {
  SmallVector<OpAsmParser::Argument> arguments;
  auto parseInput = [&]() -> ParseResult {
    auto &argument = arguments.emplace_back();
    if (parser.parseArgument(argument) || parser.parseEqual() ||
        parser.parseOperand(inputs.emplace_back()) ||
        parser.parseColonType(types.emplace_back()))
      return failure();
    argument.type = types.back();
    return success();
  };
  if (parser.parseCommaSeparatedList(OpAsmParser::Delimiter::Paren, parseInput))
    return failure();
  return parser.parseRegion(body, arguments, /*enableNameShadowing=*/true);
}

static void printKernelBody(OpAsmPrinter &printer, Operation *,
                            OperandRange inputs, TypeRange types,
                            Region &body) {
  printer << "(";
  llvm::interleaveComma(llvm::zip_equal(body.getArguments(), inputs, types),
                        printer, [&](const auto &input) {
                          auto [argument, value, type] = input;
                          printer << argument << " = " << value << " : "
                                  << type;
                        });
  printer << ") ";
  printer.printRegion(body, /*printEntryBlockArgs=*/false);
}

} // namespace mlir::spruce

#define GET_OP_CLASSES
#include "spruce/IR/SpruceOps.cpp.inc"

namespace mlir {
namespace spruce {

// -----------------------------------------------------------------------------
// SpruceDialect
// -----------------------------------------------------------------------------

void SpruceDialect::initialize() {
  registerAttributes();

  addTypes<
#define GET_TYPEDEF_LIST
#include "spruce/IR/SpruceTypes.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "spruce/IR/SpruceOps.cpp.inc"
      >();
}

// -----------------------------------------------------------------------------
// KernelOp
// -----------------------------------------------------------------------------

LogicalResult KernelOp::verify() {
  Block &body = getBody().front();
  if (body.getNumArguments() != getNumOperands())
    return emitOpError("expected one body argument per operand");

  for (unsigned i = 0, e = getNumOperands(); i < e; ++i) {
    if (body.getArgument(i).getType() != getOperand(i).getType())
      return emitOpError("expected body argument ")
             << i << " to have the same type as operand " << i;
  }

  auto yield = dyn_cast<YieldOp>(&body.back());
  if (!yield)
    return emitOpError("expected body to terminate with spruce.yield");
  if (yield.getNumOperands() != getNumResults())
    return emitOpError("expected one yielded value per result");

  for (unsigned i = 0, e = getNumResults(); i < e; ++i) {
    if (yield.getOperand(i).getType() != getResult(i).getType())
      return emitOpError("expected yield operand ")
             << i << " to have the same type as result " << i;
  }

  return success();
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
