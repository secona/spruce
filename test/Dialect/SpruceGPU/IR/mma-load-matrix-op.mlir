// RUN: spruce-opt -verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: func.func @main()
// CHECK:         %[[FRAG:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment
// CHECK:         spruce_gpu.mma.load_matrix %[[FRAG]] : !spruce_gpu.fragment
// CHECK:         return

func.func @main() {
  %frag0 = spruce_gpu.create_fragment : !spruce_gpu.fragment
  spruce_gpu.mma.load_matrix %frag0 : !spruce_gpu.fragment
  return
}
