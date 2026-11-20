// RUN: spruce-opt -convert-spruce-to-spruce-gpu %s | FileCheck %s

// CHECK-LABEL: func.func @multiple_tiles
// CHECK-SAME:      %[[A:[^:]+]]: tensor<32x16xf32>
// CHECK-SAME:      %[[B:[^:]+]]: tensor<16x16xf32>
// CHECK-SAME:      %[[C:[^:]+]]: tensor<32x16xf32>) -> tensor<32x16xf32>
// CHECK:         %[[A_FRAG:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment
// CHECK:         %[[B_FRAG:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment
// CHECK:         %[[C_FRAG:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment
// CHECK:         %[[D_FRAG:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment
// CHECK:         spruce_gpu.mma %[[D_FRAG]], %[[A_FRAG]], %[[B_FRAG]], %[[C_FRAG]] : !spruce_gpu.fragment, !spruce_gpu.fragment, !spruce_gpu.fragment, !spruce_gpu.fragment
// CHECK:         %[[RESULT:.*]] = spruce_gpu.mma.store_matrix %[[D_FRAG]] out(%[[C]]) : !spruce_gpu.fragment, tensor<32x16xf32>
// CHECK:         return %[[RESULT]] : tensor<32x16xf32>

func.func @multiple_tiles(
  %lhs: tensor<32x16xf32>,
  %rhs: tensor<16x16xf32>,
  %acc: tensor<32x16xf32>
) -> tensor<32x16xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<32x16xf32>, tensor<16x16xf32>, tensor<32x16xf32> -> tensor<32x16xf32>
  return %result : tensor<32x16xf32>
}
