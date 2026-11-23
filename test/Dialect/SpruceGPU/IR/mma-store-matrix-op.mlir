// RUN: spruce-opt -verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: func.func @main
// CHECK-SAME:      %[[A:.*]]: tensor<32x32xf16>) -> tensor<32x32xf16> {
// CHECK:         %[[FRAG:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment<f16>
// CHECK:         %[[A_NEW:.*]] = spruce_gpu.mma.store_matrix %[[FRAG]] -> %[[A]] : !spruce_gpu.fragment<f16> -> tensor<32x32xf16>
// CHECK:         return %[[A_NEW]] : tensor<32x32xf16>

func.func @main(%A : tensor<32x32xf16>) -> tensor<32x32xf16> {
  %frag0 = spruce_gpu.create_fragment : !spruce_gpu.fragment<f16>
  %A_new = spruce_gpu.mma.store_matrix %frag0 -> %A : !spruce_gpu.fragment<f16> -> tensor<32x32xf16>
  return %A_new : tensor<32x32xf16>
}
