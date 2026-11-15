#ifndef SPRUCE_IR_SPRUCEOPS_TD
#define SPRUCE_IR_SPRUCEOPS_TD

include "spruce/IR/SpruceDialect.td"
include "mlir/IR/OpBase.td"

class Spruce_Op<string mnemonic, list<Trait> traits = []>
    : Op<Spruce_Dialect, mnemonic, traits>;

def Spruce_MmaOp : Spruce_Op<"mma"> {
  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
    AnyRankedTensor:$acc);

  let results = (outs
    AnyRankedTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $acc attr-dict
    `:` type($lhs) `,` type($rhs) `,` type($acc)
    `->` type($result)
  }];

  let extraClassDeclaration = [{
    std::int64_t getM() { return getLhs().getType().getDimSize(0); }
    std::int64_t getN() { return getRhs().getType().getDimSize(1); }
    std::int64_t getK() { return getLhs().getType().getDimSize(1); }

    std::tuple<std::int64_t, std::int64_t, std::int64_t> getMNK() {
      return {getM(), getN(), getK()};
    }
  }];

  let hasVerifier = 1;
}

def Spruce_GetBlockOp : Spruce_Op<"get_block"> {
  let arguments = (ins
    AnyRankedTensor:$tensor,
    Index:$blockIndex);

  let results = (outs
    AnyRankedTensor:$result);

  let assemblyFormat = [{
    $tensor `[` $blockIndex `]` attr-dict
    `:` type($tensor) `->` type($result)
  }];
}

#endif
