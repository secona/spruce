// RUN: spruce-opt -verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK-SAME:      %[[A:.*]]: tensor<32x32xf32>
// CHECK:         %[[FRAG:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment
// CHECK:         spruce_gpu.mma.load_matrix %[[A]] -> %[[FRAG]] : tensor<32x32xf32> -> !spruce_gpu.fragment
// CHECK:         return

func.func @main(%A : tensor<32x32xf32>) {
  %frag0 = spruce_gpu.create_fragment : !spruce_gpu.fragment
  spruce_gpu.mma.load_matrix %A -> %frag0 : tensor<32x32xf32> -> !spruce_gpu.fragment
  return
}
