#include "spruce/Dialect/Spruce/IR/SpruceAttributes.h"
#include "spruce/Dialect/Spruce/IR/SpruceDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_ATTRDEF_CLASSES
#include "spruce/Dialect/Spruce/IR/SpruceAttributes.cpp.inc"

void mlir::spruce::SpruceDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "spruce/Dialect/Spruce/IR/SpruceAttributes.cpp.inc"
      >();
}

mlir::LogicalResult mlir::spruce::EncodingAttr::verifyEncoding(
    ArrayRef<int64_t> shape, Type elementType,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  return getSparse().verifyEncoding(shape, elementType, emitError);
}
