#ifndef SPRUCE_IR_SPRUCETYPES_TD
#define SPRUCE_IR_SPRUCETYPES_TD

include "spruce/IR/SpruceDialect.td"

class Spruce_Type<string name, list<Trait> traits = []>
    : TypeDef<Spruce_Dialect, name, traits>;

// -----------------------------------------------------------------------------
// FragmentType
// -----------------------------------------------------------------------------

def Spruce_FragmentType : Spruce_Type<"Fragment"> {
  let mnemonic = "fragment";
}

#endif // SPRUCE_IR_SPRUCETYPES_TD
