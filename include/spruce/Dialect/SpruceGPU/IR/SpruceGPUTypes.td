#ifndef SPRUCE_GPU_IR_SPRUCEGPUTYPES_TD
#define SPRUCE_GPU_IR_SPRUCEGPUTYPES_TD

include "spruce/Dialect/SpruceGPU/IR/SpruceGPUDialect.td"

class SpruceGPU_Type<string name, list<Trait> traits = []>
    : TypeDef<SpruceGPU_Dialect, name, traits>;

// -----------------------------------------------------------------------------
// FragmentType
// -----------------------------------------------------------------------------

def SpruceGPU_FragmentType : SpruceGPU_Type<"Fragment"> {
  let mnemonic = "fragment";
}

#endif
