// RUN: spruce-opt -spruce-tile-mma -verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: func.func @supported_tile(
// CHECK-SAME: %[[LHS:.*]]: tensor<16x8xf32>, %[[RHS:.*]]: tensor<8x8xf32>, %[[ACC:.*]]: tensor<16x8xf32>
// CHECK: %[[RESULT:.*]] = spruce.mma %[[LHS]], %[[RHS]], %[[ACC]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
// CHECK: return %[[RESULT]] : tensor<16x8xf32>
func.func @supported_tile(%lhs: tensor<16x8xf32>, %rhs: tensor<8x8xf32>, %acc: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
  return %result : tensor<16x8xf32>
}

// CHECK-LABEL: func.func @no_mma(
// CHECK: return
func.func @no_mma() {
  return
}
