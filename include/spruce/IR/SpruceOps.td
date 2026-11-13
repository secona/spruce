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

  let hasVerifier = 1;
}

#endif
