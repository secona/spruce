// RUN: spruce-opt -verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK-SAME:      %[[A:.*]]: tensor<32x32xf16>
// CHECK:         %[[FRAG:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment<f16>
// CHECK:         spruce_gpu.mma.load_matrix %[[A]] -> %[[FRAG]] : tensor<32x32xf16> -> !spruce_gpu.fragment<f16>
// CHECK:         return

func.func @main(%A : tensor<32x32xf16>) {
  %frag0 = spruce_gpu.create_fragment : !spruce_gpu.fragment<f16>
  spruce_gpu.mma.load_matrix %A -> %frag0 : tensor<32x32xf16> -> !spruce_gpu.fragment<f16>
  return
}
