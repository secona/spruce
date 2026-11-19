#ifndef SPRUCE_IR_SPRUCEDIALECT_TD
#define SPRUCE_IR_SPRUCEDIALECT_TD

include "mlir/IR/AttrTypeBase.td"

def Spruce_Dialect : Dialect {
  let name = "spruce";
  let cppNamespace = "::mlir::spruce";

  let dependentDialects = ["::mlir::sparse_tensor::SparseTensorDialect"];
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerAttributes();
  }];
}

#endif
