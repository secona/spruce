#ifndef SPRUCE_GPU_IR_SPRUCEGPUOPS_TD
#define SPRUCE_GPU_IR_SPRUCEGPUOPS_TD

include "spruce/Dialect/SpruceGPU/IR/SpruceGPUDialect.td"
include "spruce/Dialect/SpruceGPU/IR/SpruceGPUTypes.td"
include "mlir/IR/OpBase.td"

class SpruceGPU_Op<string mnemonic, list<Trait> traits = []>
    : Op<SpruceGPU_Dialect, mnemonic, traits>;

// -----------------------------------------------------------------------------
// CreateFragmentOp
// -----------------------------------------------------------------------------

def SpruceGPU_CreateFragmentOp : SpruceGPU_Op<"create_fragment"> {
  let results = (outs
    SpruceGPU_FragmentType:$result);

  let assemblyFormat = [{
    attr-dict `:` qualified(type($result))
  }];
}

// -----------------------------------------------------------------------------
// MmaLoadMatrixOp
// -----------------------------------------------------------------------------

def SpruceGPU_MmaLoadMatrixOp : SpruceGPU_Op<"mma.load_matrix", [
  AllTypesMatch<["fragment", "result"]>
]> {
  let arguments = (ins
    AnyRankedTensor:$tensor,
    SpruceGPU_FragmentType:$fragment);

  let results = (outs
    SpruceGPU_FragmentType:$result);

  let assemblyFormat = [{
    $tensor `->` $fragment
    attr-dict
    `:` type($tensor) `->` qualified(type($fragment))
  }];
}

// -----------------------------------------------------------------------------
// MmaStoreMatrixOp
// -----------------------------------------------------------------------------

def SpruceGPU_MmaStoreMatrixOp : SpruceGPU_Op<"mma.store_matrix", [
  AllTypesMatch<["out", "result"]>
]> {
  let arguments = (ins
    SpruceGPU_FragmentType:$fragment,
    AnyRankedTensor:$out);

  let results = (outs
    AnyRankedTensor:$result);

  let assemblyFormat = [{
    $fragment
    `out` `(` $out `)`
    attr-dict
    `:` qualified(type($fragment)) `,` type($result)
  }];
}

// -----------------------------------------------------------------------------
// MmaOp
// -----------------------------------------------------------------------------

def SpruceGPU_MmaOp : SpruceGPU_Op<"mma", [
  AllTypesMatch<["d", "result"]>
]> {
  let arguments = (ins
    SpruceGPU_FragmentType:$d,
    SpruceGPU_FragmentType:$a,
    SpruceGPU_FragmentType:$b,
    SpruceGPU_FragmentType:$c);

  let results = (outs
    SpruceGPU_FragmentType:$result);

  let assemblyFormat = [{
    $a `,` $b `,` $c `->` $d
    attr-dict
    `:` qualified(type($a)) `,` qualified(type($b)) `,` qualified(type($c))
    `->` qualified(type($result))
  }];
}

#endif
