// RUN: spruce-opt -verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: func.func @main()
// CHECK:         %{{.*}} = spruce_gpu.create_fragment : !spruce_gpu.fragment
// CHECK:         return

func.func @main() {
  %0 = spruce_gpu.create_fragment : !spruce_gpu.fragment
  return
}
