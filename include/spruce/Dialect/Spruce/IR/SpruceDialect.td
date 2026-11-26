#ifndef SPRUCE_IR_SPRUCEDIALECT_TD
#define SPRUCE_IR_SPRUCEDIALECT_TD

include "mlir/IR/AttrTypeBase.td"

def Spruce_Dialect : Dialect {
  let name = "spruce";
  let cppNamespace = "::mlir::spruce";

  let useDefaultAttributePrinterParser = 1;

  let extraClassDeclaration = [{
    void registerAttributes();
    void registerBufferizationInterfaces();
  }];
}

#endif
