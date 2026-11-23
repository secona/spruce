// RUN: spruce-opt -verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: func.func @main()
// CHECK:         %[[A:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment<f16>
// CHECK:         %[[B:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment<f16>
// CHECK:         %[[D:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment<f16>
// CHECK:         spruce_gpu.mma %[[A]], %[[B]], %[[D]] -> %[[D]]
// CHECK-SAME:        : !spruce_gpu.fragment<f16>, !spruce_gpu.fragment<f16>, !spruce_gpu.fragment<f16> -> !spruce_gpu.fragment<f16>
// CHECK:         return

func.func @main() {
  %A_frag = spruce_gpu.create_fragment : !spruce_gpu.fragment<f16>
  %B_frag = spruce_gpu.create_fragment : !spruce_gpu.fragment<f16>
  %D_frag = spruce_gpu.create_fragment : !spruce_gpu.fragment<f16>

  spruce_gpu.mma %A_frag, %B_frag, %D_frag -> %D_frag
      : !spruce_gpu.fragment<f16>, !spruce_gpu.fragment<f16>, !spruce_gpu.fragment<f16>
      -> !spruce_gpu.fragment<f16>

  return
}
