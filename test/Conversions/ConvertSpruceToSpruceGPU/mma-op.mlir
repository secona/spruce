// RUN: spruce-opt -convert-spruce-to-spruce-gpu %s | FileCheck %s

// CHECK-LABEL: func.func @multiple_tiles
// CHECK-SAME:      %[[A_H:[^:]+]]: tensor<32x16xf16>
// CHECK-SAME:      %[[B_H:[^:]+]]: tensor<16x16xf16>
// CHECK-SAME:      %[[C_H:[^:]+]]: tensor<32x16xf16>) -> tensor<32x16xf16>
// CHECK:         %[[RESULT:.*]] = spruce.kernel
// CHECK-SAME:        %[[A:[^:]+]] = %[[A_H]] : tensor<32x16xf16>
// CHECK-SAME:        %[[B:[^:]+]] = %[[B_H]] : tensor<16x16xf16>
// CHECK-SAME:        %[[C:[^:]+]] = %[[C_H]] : tensor<32x16xf16>
// CHECK:           %[[AF_INIT:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment<f16>
// CHECK:           %[[BF_INIT:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment<f16>
// CHECK:           %[[CF_INIT:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment<f16>
// CHECK:           %[[DF_INIT:.*]] = spruce_gpu.create_fragment : !spruce_gpu.fragment<f16>
// CHECK:           %[[AF:.*]] = spruce_gpu.mma.load_matrix %[[A]] -> %[[AF_INIT]] : tensor<32x16xf16> -> !spruce_gpu.fragment<f16>
// CHECK:           %[[BF:.*]] = spruce_gpu.mma.load_matrix %[[B]] -> %[[BF_INIT]] : tensor<16x16xf16> -> !spruce_gpu.fragment<f16>
// CHECK:           %[[CF:.*]] = spruce_gpu.mma.load_matrix %[[C]] -> %[[CF_INIT]] : tensor<32x16xf16> -> !spruce_gpu.fragment<f16>
// CHECK:           %[[DF:.*]] = spruce_gpu.mma %[[DF_INIT]], %[[AF]], %[[BF]], %[[CF]] : !spruce_gpu.fragment<f16>, !spruce_gpu.fragment<f16>, !spruce_gpu.fragment<f16>, !spruce_gpu.fragment<f16> -> !spruce_gpu.fragment<f16>
// CHECK:           %[[KRESULT:.*]] = spruce_gpu.mma.store_matrix %[[DF]] out(%[[C]]) : !spruce_gpu.fragment<f16>, tensor<32x16xf16>
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
