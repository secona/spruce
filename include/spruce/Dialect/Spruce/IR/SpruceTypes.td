#ifndef SPRUCE_IR_SPRUCETYPES_TD
#define SPRUCE_IR_SPRUCETYPES_TD

include "spruce/Dialect/Spruce/IR/SpruceDialect.td"

class Spruce_Type<string name, list<Trait> traits = []>
    : TypeDef<Spruce_Dialect, name, traits>;

#endif // SPRUCE_IR_SPRUCETYPES_TD
