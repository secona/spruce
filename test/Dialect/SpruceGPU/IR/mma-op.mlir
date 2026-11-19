// RUN: spruce-opt -verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: func.func @main()
// CHECK:         %[[A:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment
// CHECK:         %[[B:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment
// CHECK:         %[[D:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment
// CHECK:         spruce_gpu.mma %[[D]], %[[A]], %[[B]], %[[D]]
// CHECK-SAME:        !spruce_gpu.fragment, !spruce_gpu.fragment, !spruce_gpu.fragment, !spruce_gpu.fragment
// CHECK:         return

func.func @main() {
  %A_frag = spruce_gpu.create_fragment : !spruce_gpu.fragment
  %B_frag = spruce_gpu.create_fragment : !spruce_gpu.fragment
  %D_frag = spruce_gpu.create_fragment : !spruce_gpu.fragment

  spruce_gpu.mma %D_frag, %A_frag, %B_frag, %D_frag : !spruce_gpu.fragment, !spruce_gpu.fragment, !spruce_gpu.fragment, !spruce_gpu.fragment

  return
}
