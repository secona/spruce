// RUN: spruce-opt -spruce-mma-to-vector -canonicalize -cse -verify-roundtrip %s | FileCheck %s --implicit-check-not=spruce.mma
// RUN: spruce-opt -spruce-mma-to-vector -spruce-mma-to-vector -canonicalize -cse -verify-roundtrip %s | FileCheck %s --implicit-check-not=spruce.mma

// CHECK-DAG: #[[TRANSPOSE:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-DAG: #[[LHS_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-DAG: #[[RHS_MAP:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// CHECK-DAG: #[[ACC_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK-LABEL: func.func @supported_tile(
// CHECK-SAME:      %[[LHS:[^:]+]]: tensor<16x8xf32>,
// CHECK-SAME:      %[[RHS:[^:]+]]: tensor<8x8xf32>,
// CHECK-SAME:      %[[ACC:[^:]+]]: tensor<16x8xf32>)
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32

// CHECK:         %[[LHS_VEC:.*]] = vector.transfer_read %[[LHS]][%[[C0]], %[[C0]]], %[[ZERO]] {in_bounds = [true, true]} : tensor<16x8xf32>, vector<16x8xf32>
// CHECK:         %[[RHS_VEC:.*]] = vector.transfer_read %[[RHS]][%[[C0]], %[[C0]]], %[[ZERO]] {in_bounds = [true, true], permutation_map = #[[TRANSPOSE]]} : tensor<8x8xf32>, vector<8x8xf32>
// CHECK:         %[[ACC_VEC:.*]] = vector.transfer_read %[[ACC]][%[[C0]], %[[C0]]], %[[ZERO]] {in_bounds = [true, true]} : tensor<16x8xf32>, vector<16x8xf32>

// CHECK:         %[[PRODUCT:.*]] = vector.contract {indexing_maps = [#[[LHS_MAP]], #[[RHS_MAP]], #[[ACC_MAP]]], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[LHS_VEC]], %[[RHS_VEC]], %[[ACC_VEC]] : vector<16x8xf32>, vector<8x8xf32> into vector<16x8xf32>
// CHECK:         %[[RESULT:.*]] = vector.transfer_write %[[PRODUCT]], %[[ACC]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} : vector<16x8xf32>, tensor<16x8xf32>
// CHECK:         return %[[RESULT]] : tensor<16x8xf32>

func.func @supported_tile(%lhs: tensor<16x8xf32>, %rhs: tensor<8x8xf32>, %acc: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
  return %result : tensor<16x8xf32>
}

// CHECK-LABEL: func.func @rectangular_mma(
// CHECK-SAME:      %[[LHS:[^:]+]]: tensor<3x5xf32>,
// CHECK-SAME:      %[[RHS:[^:]+]]: tensor<5x7xf32>,
// CHECK-SAME:      %[[ACC:[^:]+]]: tensor<3x7xf32>)
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32

// CHECK:         %[[LHS_VEC:.*]] = vector.transfer_read %[[LHS]][%[[C0]], %[[C0]]], %[[ZERO]] {in_bounds = [true, true]} : tensor<3x5xf32>, vector<3x5xf32>
// CHECK:         %[[RHS_VEC:.*]] = vector.transfer_read %[[RHS]][%[[C0]], %[[C0]]], %[[ZERO]] {in_bounds = [true, true], permutation_map = #[[TRANSPOSE]]} : tensor<5x7xf32>, vector<7x5xf32>
// CHECK:         %[[ACC_VEC:.*]] = vector.transfer_read %[[ACC]][%[[C0]], %[[C0]]], %[[ZERO]] {in_bounds = [true, true]} : tensor<3x7xf32>, vector<3x7xf32>

// CHECK:         %[[PRODUCT:.*]] = vector.contract {indexing_maps = [#[[LHS_MAP]], #[[RHS_MAP]], #[[ACC_MAP]]], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[LHS_VEC]], %[[RHS_VEC]], %[[ACC_VEC]] : vector<3x5xf32>, vector<7x5xf32> into vector<3x7xf32>
// CHECK:         %[[RESULT:.*]] = vector.transfer_write %[[PRODUCT]], %[[ACC]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} : vector<3x7xf32>, tensor<3x7xf32>
// CHECK:         return %[[RESULT]] : tensor<3x7xf32>

func.func @rectangular_mma(%lhs: tensor<3x5xf32>, %rhs: tensor<5x7xf32>, %acc: tensor<3x7xf32>) -> tensor<3x7xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<3x5xf32>, tensor<5x7xf32>, tensor<3x7xf32> -> tensor<3x7xf32>
  return %result : tensor<3x7xf32>
}

// CHECK-LABEL: func.func @chained_mma(
// CHECK-SAME:      %[[LHS0:[^:]+]]: tensor<16x8xf32>,
// CHECK-SAME:      %[[RHS0:[^:]+]]: tensor<8x8xf32>,
// CHECK-SAME:      %[[LHS1:[^:]+]]: tensor<16x8xf32>,
// CHECK-SAME:      %[[RHS1:[^:]+]]: tensor<8x8xf32>,
// CHECK-SAME:      %[[ACC:[^:]+]]: tensor<16x8xf32>)
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32

// CHECK:         %[[LHS_VEC0:.*]] = vector.transfer_read %[[LHS0]][%[[C0]], %[[C0]]], %[[ZERO]]
// CHECK:         %[[RHS_VEC0:.*]] = vector.transfer_read %[[RHS0]][%[[C0]], %[[C0]]], %[[ZERO]]
// CHECK:         %[[ACC_VEC:.*]] = vector.transfer_read %[[ACC]][%[[C0]], %[[C0]]], %[[ZERO]]
// CHECK:         %[[PART:.*]] = vector.contract {{.*}} %[[LHS_VEC0]], %[[RHS_VEC0]], %[[ACC_VEC]] : vector<16x8xf32>, vector<8x8xf32> into vector<16x8xf32>

// CHECK:         %[[LHS_VEC1:.*]] = vector.transfer_read %[[LHS1]][%[[C0]], %[[C0]]], %[[ZERO]]
// CHECK:         %[[RHS_VEC1:.*]] = vector.transfer_read %[[RHS1]][%[[C0]], %[[C0]]], %[[ZERO]]
// CHECK:         %[[PRODUCT:.*]] = vector.contract {{.*}} %[[LHS_VEC1]], %[[RHS_VEC1]], %[[PART]] : vector<16x8xf32>, vector<8x8xf32> into vector<16x8xf32>
// CHECK:         %[[RESULT:.*]] = vector.transfer_write %[[PRODUCT]], %[[ACC]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} : vector<16x8xf32>, tensor<16x8xf32>
// CHECK:         return %[[RESULT]] : tensor<16x8xf32>

func.func @chained_mma(%lhs0: tensor<16x8xf32>, %rhs0: tensor<8x8xf32>, %lhs1: tensor<16x8xf32>, %rhs1: tensor<8x8xf32>, %acc: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %partial = spruce.mma %lhs0, %rhs0, %acc : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
  %result = spruce.mma %lhs1, %rhs1, %partial : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
  return %result : tensor<16x8xf32>
}

// CHECK-LABEL: func.func @empty_reduction(
// CHECK-SAME:      %{{[^:]+}}: tensor<3x0xf32>,
// CHECK-SAME:      %{{[^:]+}}: tensor<0x7xf32>,
// CHECK-SAME:      %[[ACC:[^:]+]]: tensor<3x7xf32>)
// CHECK-NEXT:    return %[[ACC]] : tensor<3x7xf32>

func.func @empty_reduction(%lhs: tensor<3x0xf32>, %rhs: tensor<0x7xf32>, %acc: tensor<3x7xf32>) -> tensor<3x7xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<3x0xf32>, tensor<0x7xf32>, tensor<3x7xf32> -> tensor<3x7xf32>
  return %result : tensor<3x7xf32>
}

// CHECK-LABEL: func.func @empty_rows(
// CHECK-SAME:      %{{[^:]+}}: tensor<0x5xf32>,
// CHECK-SAME:      %{{[^:]+}}: tensor<5x7xf32>,
// CHECK-SAME:      %[[ACC:[^:]+]]: tensor<0x7xf32>)
// CHECK-NEXT:    return %[[ACC]] : tensor<0x7xf32>

func.func @empty_rows(%lhs: tensor<0x5xf32>, %rhs: tensor<5x7xf32>, %acc: tensor<0x7xf32>) -> tensor<0x7xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<0x5xf32>, tensor<5x7xf32>, tensor<0x7xf32> -> tensor<0x7xf32>
  return %result : tensor<0x7xf32>
}

// CHECK-LABEL: func.func @empty_columns(
// CHECK-SAME:      %{{[^:]+}}: tensor<3x5xf32>,
// CHECK-SAME:      %{{[^:]+}}: tensor<5x0xf32>,
// CHECK-SAME:      %[[ACC:[^:]+]]: tensor<3x0xf32>)
// CHECK-NEXT:    return %[[ACC]] : tensor<3x0xf32>

func.func @empty_columns(%lhs: tensor<3x5xf32>, %rhs: tensor<5x0xf32>, %acc: tensor<3x0xf32>) -> tensor<3x0xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<3x5xf32>, tensor<5x0xf32>, tensor<3x0xf32> -> tensor<3x0xf32>
  return %result : tensor<3x0xf32>
}

// CHECK-LABEL: func.func @no_mma(
// CHECK-NEXT:    return

func.func @no_mma() {
  return
}
