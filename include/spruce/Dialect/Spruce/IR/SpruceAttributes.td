#ifndef SPRUCE_IR_SPRUCEATTRIBUTES_TD
#define SPRUCE_IR_SPRUCEATTRIBUTES_TD

include "spruce/Dialect/Spruce/IR/SpruceDialect.td"
include "mlir/IR/TensorEncoding.td"

class Spruce_Attr<string name, list<Trait> traits = []>
    : AttrDef<Spruce_Dialect, name, traits>;

// -----------------------------------------------------------------------------
// TargetAttr
// -----------------------------------------------------------------------------

def Spruce_TargetAttr : Spruce_Attr<"Target"> {
  let mnemonic = "target";
  let genMnemonicAlias = 1;

  let description = [{
    `#spruce.target` attribute defines the target lowering of a specific spruce
    op. As of now, its primary use case is to define the matrix shape for an
    `mma.sync` instruction.
  }];

  let parameters = (ins
    "int64_t":$m,
    "int64_t":$n,
    "int64_t":$k);

  let assemblyFormat = [{
    `<` `{` struct(params) `}` `>`
  }];
}

// -----------------------------------------------------------------------------
// EncodingAttr
// -----------------------------------------------------------------------------

def Spruce_EncodingAttr : Spruce_Attr<"Encoding", [
  DeclareAttrInterfaceMethods<VerifiableTensorEncoding>
]> {
  let mnemonic = "encoding";
  let genMnemonicAlias = 1;

  let description = [{
    `#spruce.encoding` attribute wraps the `#sparse_tensor.encoding` attribute
    with additional data.
  }];

  let parameters = (ins
    "::mlir::sparse_tensor::SparseTensorEncodingAttr":$sparse);

  let assemblyFormat = [{
    `<` `{` struct(params) `}` `>`
  }];
}

// -----------------------------------------------------------------------------
// DimMaskAttr
// -----------------------------------------------------------------------------

def Spruce_DimMaskAttr : Spruce_Attr<"DimMask"> {
  let mnemonic = "dim_mask";
  let genMnemonicAlias = 1;

  let parameters = (ins
    "uint64_t":$size,
    "::mlir::DenseI64ArrayAttr":$words);

  let assemblyFormat = [{
    `<` $size `,` $words `>`
  }];
}

// -----------------------------------------------------------------------------
// SparsityAttr
// -----------------------------------------------------------------------------

def Spruce_SparsityAttr : Spruce_Attr<"Sparsity"> {
  let mnemonic = "sparsity";
  let genMnemonicAlias = 1;

  let parameters = (ins
    ArrayRefParameter<"::mlir::spruce::DimMaskAttr">:$dims);

  let assemblyFormat = [{
    `<` `[` $dims `]` `>`
  }];
}

#endif
