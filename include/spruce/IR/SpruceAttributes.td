#ifndef SPRUCE_IR_SPRUCEATTRIBUTES_TD
#define SPRUCE_IR_SPRUCEATTRIBUTES_TD

include "spruce/IR/SpruceDialect.td"
include "mlir/IR/TensorEncoding.td"

class Spruce_Attr<string name, list<Trait> traits = []>
    : AttrDef<Spruce_Dialect, name, traits>;

def Spruce_EncodingAttr : Spruce_Attr<"Encoding", [
  DeclareAttrInterfaceMethods<VerifiableTensorEncoding>
]> {
  let mnemonic = "encoding";
  let genMnemonicAlias = 1;

  let parameters = (ins
    "::mlir::sparse_tensor::SparseTensorEncodingAttr":$sparse);

  let assemblyFormat = [{
    `<` `{` struct(params) `}` `>`
  }];
}

#endif
