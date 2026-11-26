#ifndef SPRUCE_IR_SPRUCEATTRIBUTES_TD
#define SPRUCE_IR_SPRUCEATTRIBUTES_TD

include "spruce/Dialect/Spruce/IR/SpruceDialect.td"

class Spruce_Attr<string name, list<Trait> traits = []>
    : AttrDef<Spruce_Dialect, name, traits>;

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

  let extraClassDeclaration = [{
    ::llvm::Bitset<2048> getBitsetMap() {
      std::array<uint64_t, llvm::Bitset<2048>::getNumWords64()> words{};
      auto packed = getWords().asArrayRef();
      llvm::copy(packed, words.begin());
      return llvm::Bitset<2048>(words);
    }
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
