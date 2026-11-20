// RUN: spruce-opt -verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK-SAME:      %[[A:.*]]: tensor<32x32xf32>) -> tensor<32x32xf32> {
// CHECK:         %[[FRAG:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment
// CHECK:         %[[A_NEW:.*]] = spruce_gpu.mma.store_matrix %[[FRAG]] out(%[[A]]) : !spruce_gpu.fragment, tensor<32x32xf32>
// CHECK:         return %[[A_NEW]] : tensor<32x32xf32>

func.func @main(%A : tensor<32x32xf32>) -> tensor<32x32xf32> {
  %frag0 = spruce_gpu.create_fragment : !spruce_gpu.fragment
  %A_new = spruce_gpu.mma.store_matrix %frag0 out(%A) : !spruce_gpu.fragment, tensor<32x32xf32>
  return %A_new : tensor<32x32xf32>
}
