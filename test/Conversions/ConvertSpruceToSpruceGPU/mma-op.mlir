// RUN: spruce-opt -convert-spruce-to-spruce-gpu %s | FileCheck %s

// CHECK-LABEL: func.func @multiple_tiles
// CHECK-SAME:      %[[A_H:[^:]+]]: tensor<32x16xf16>
// CHECK-SAME:      %[[B_H:[^:]+]]: tensor<16x16xf16>
// CHECK-SAME:      %[[C_H:[^:]+]]: tensor<32x16xf16>) -> tensor<32x16xf16>
// CHECK:         %[[RESULT:.*]] = spruce.kernel
// CHECK-SAME:        %[[A:[^:]+]] = %[[A_H]] : tensor<32x16xf16>
// CHECK-SAME:        %[[B:[^:]+]] = %[[B_H]] : tensor<16x16xf16>
// CHECK-SAME:        %[[C:[^:]+]] = %[[C_H]] : tensor<32x16xf16>
// CHECK:           %[[A_FRAG:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment
// CHECK:           %[[B_FRAG:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment
// CHECK:           %[[C_FRAG:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment
// CHECK:           %[[D_FRAG:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment
// CHECK:           spruce_gpu.mma.load_matrix %[[A]] -> %[[A_FRAG]] : tensor<32x16xf16> -> !spruce_gpu.fragment
// CHECK:           spruce_gpu.mma.load_matrix %[[B]] -> %[[B_FRAG]] : tensor<16x16xf16> -> !spruce_gpu.fragment
// CHECK:           spruce_gpu.mma.load_matrix %[[C]] -> %[[C_FRAG]] : tensor<32x16xf16> -> !spruce_gpu.fragment
// CHECK:           spruce_gpu.mma %[[D_FRAG]], %[[A_FRAG]], %[[B_FRAG]], %[[C_FRAG]] : !spruce_gpu.fragment, !spruce_gpu.fragment, !spruce_gpu.fragment, !spruce_gpu.fragment
// CHECK:           %[[KRESULT:.*]] = spruce_gpu.mma.store_matrix %[[D_FRAG]] out(%[[C]]) : !spruce_gpu.fragment, tensor<32x16xf16>
// CHECK:           spruce.yield %[[KRESULT]] : tensor<32x16xf16>
// CHECK:         return %[[RESULT]] : tensor<32x16xf16>

func.func @multiple_tiles(
  %lhs: tensor<32x16xf16>,
  %rhs: tensor<16x16xf16>,
  %acc: tensor<32x16xf16>
) -> tensor<32x16xf16> {
  %result = spruce.kernel(
    %A_d = %lhs: tensor<32x16xf16>,
    %B_d = %rhs: tensor<16x16xf16>,
    %C_d = %acc: tensor<32x16xf16>
  ) {
    %D_d = spruce.mma %A_d, %B_d, %C_d : tensor<32x16xf16>, tensor<16x16xf16>, tensor<32x16xf16> -> tensor<32x16xf16>
    spruce.yield %D_d : tensor<32x16xf16>
  } -> (tensor<32x16xf16>)
  return %result : tensor<32x16xf16>
}
