// RUN: spruce-opt -verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: func.func @main()
// CHECK:         %{{.*}} = spruce.create_fragment : !spruce.fragment
// CHECK:         return

func.func @main() {
  %0 = spruce.create_fragment : !spruce.fragment
  return
}
