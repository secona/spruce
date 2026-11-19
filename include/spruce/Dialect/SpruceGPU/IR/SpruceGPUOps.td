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
    attr-dict `:` type($result)
  }];
}

// -----------------------------------------------------------------------------
// MmaLoadMatrixOp
// -----------------------------------------------------------------------------

def SpruceGPU_MmaLoadMatrixOp : SpruceGPU_Op<"mma.load_matrix"> {
  let arguments = (ins
    SpruceGPU_FragmentType:$fragment);

  let assemblyFormat = [{
    $fragment attr-dict `:` type($fragment)
  }];
}

// -----------------------------------------------------------------------------
// MmaOp
// -----------------------------------------------------------------------------

def SpruceGPU_MmaOp : SpruceGPU_Op<"mma"> {
  let arguments = (ins
    SpruceGPU_FragmentType:$a,
    SpruceGPU_FragmentType:$b,
    SpruceGPU_FragmentType:$c,
    SpruceGPU_FragmentType:$d);

  let assemblyFormat = [{
    $a `,` $b `,` $c `,` $d
    attr-dict
    `:` type($a) `,` type($b) `,` type($c) `,` type($d)
  }];
}

#endif
