#ifndef SPRUCE_IR_SPRUCEOPS_TD
#define SPRUCE_IR_SPRUCEOPS_TD

include "spruce/Dialect/Spruce/IR/SpruceDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// spruce ops should be designed as sparse-aware ops.
class Spruce_Op<string mnemonic, list<Trait> traits = []>
    : Op<Spruce_Dialect, mnemonic, traits>;

// -----------------------------------------------------------------------------
// KernelOp
// -----------------------------------------------------------------------------

def Spruce_KernelOp : Spruce_Op<"kernel", [
  IsolatedFromAbove,
  SingleBlock
]> {
  let description = [{
    `spruce.kernel` represents a GPU kernel function.
  }];

  let arguments = (ins
    Variadic<AnyType>:$inputs);

  let results = (outs
    Variadic<AnyType>:$results);

  let regions = (region
    SizedRegion<1>:$body);

  let assemblyFormat = [{
    `` custom<KernelBody>($inputs, type($inputs), $body)
    (`->` `(` type($results)^ `)`)? attr-dict
  }];

  let hasVerifier = 1;
}

// -----------------------------------------------------------------------------
// YieldOp
// -----------------------------------------------------------------------------

def Spruce_YieldOp : Spruce_Op<"yield", [
  Terminator,
  HasParent<"KernelOp">
]> {
  let description = [{
    `spruce.yield` represents returning a value from a function, like
    `spruce.kernel`.
  }];

  let arguments = (ins
    Variadic<AnyType>:$values);

  let assemblyFormat = [{
    attr-dict ($values^ `:` type($values))?
  }];
}

// -----------------------------------------------------------------------------
// MmaOp
// -----------------------------------------------------------------------------

def Spruce_Matrix : AnyTypeOf<[AnyRankedTensor, AnyMemRef], "", "::mlir::ShapedType">;

def Spruce_MmaOp : Spruce_Op<"mma", [
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let description = [{
    `spruce.mma` represents matrix multiply-add instruction in a GPU, such as
    the `mma.sync` instruction in NVIDIA GPUs. `spruce.mma` is designed to be
    at a higher level without the shape and type restrictions that `mma.sync`
    instructions have. It will be lowered to `mma.sync` by the pipeline via
    tiling and type conversions. Tensor operands produce a tensor result.
    Memref operands update the accumulator in place and produce no results.
  }];

  let arguments = (ins
    Spruce_Matrix:$lhs,
    Spruce_Matrix:$rhs,
    Spruce_Matrix:$acc);

  let results = (outs
    Optional<AnyRankedTensor>:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs `,` $acc attr-dict
    `:` type($lhs) `,` type($rhs) `,` type($acc)
    (`->` type($result)^)?
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

// -----------------------------------------------------------------------------
// GetBlockOp
// -----------------------------------------------------------------------------

def Spruce_GetBlockOp : Spruce_Op<"get_block"> {
  let description = [{
    `spruce.get_block` represents getting a block of a tensor.
  }];

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
