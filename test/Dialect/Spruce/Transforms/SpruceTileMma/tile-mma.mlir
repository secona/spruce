// RUN: spruce-opt -spruce-tile-mma -verify-roundtrip %s | FileCheck %s
// RUN: spruce-opt -spruce-tile-mma -spruce-tile-mma -verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: func.func @supported_tile(
// CHECK-SAME: %[[LHS:.*]]: tensor<16x8xf32>, %[[RHS:.*]]: tensor<8x8xf32>, %[[ACC:.*]]: tensor<16x8xf32>
// CHECK: %[[RESULT:.*]] = spruce.mma %[[LHS]], %[[RHS]], %[[ACC]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
// CHECK: return %[[RESULT]] : tensor<16x8xf32>
func.func @supported_tile(%lhs: tensor<16x8xf32>, %rhs: tensor<8x8xf32>, %acc: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
  return %result : tensor<16x8xf32>
}

// CHECK-LABEL: func.func @multiple_tiles(
// CHECK-SAME: %[[LHS:.*]]: tensor<32x16xf32>, %[[RHS:.*]]: tensor<16x16xf32>, %[[ACC:.*]]: tensor<32x16xf32>
// CHECK: %[[ACC00:.*]] = tensor.extract_slice %[[ACC]][0, 0] [16, 8] [1, 1]
// CHECK: %[[LHS00:.*]] = tensor.extract_slice %[[LHS]][0, 0] [16, 8] [1, 1]
// CHECK: %[[RHS00:.*]] = tensor.extract_slice %[[RHS]][0, 0] [8, 8] [1, 1]
// CHECK: %[[PART00:.*]] = spruce.mma %[[LHS00]], %[[RHS00]], %[[ACC00]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
// CHECK: %[[LHS01:.*]] = tensor.extract_slice %[[LHS]][0, 8] [16, 8] [1, 1]
// CHECK: %[[RHS10:.*]] = tensor.extract_slice %[[RHS]][8, 0] [8, 8] [1, 1]
// CHECK: %[[TILE00:.*]] = spruce.mma %[[LHS01]], %[[RHS10]], %[[PART00]]
// CHECK: %[[RESULT00:.*]] = tensor.insert_slice %[[TILE00]] into %[[ACC]][0, 0] [16, 8] [1, 1]
// CHECK: %[[ACC01:.*]] = tensor.extract_slice %[[ACC]][0, 8] [16, 8] [1, 1]
// CHECK: %[[RHS01:.*]] = tensor.extract_slice %[[RHS]][0, 8] [8, 8] [1, 1]
// CHECK: %[[PART01:.*]] = spruce.mma %{{.*}}, %[[RHS01]], %[[ACC01]]
// CHECK: %[[RHS11:.*]] = tensor.extract_slice %[[RHS]][8, 8] [8, 8] [1, 1]
// CHECK: %[[TILE01:.*]] = spruce.mma %{{.*}}, %[[RHS11]], %[[PART01]]
// CHECK: %[[RESULT01:.*]] = tensor.insert_slice %[[TILE01]] into %[[RESULT00]][0, 8] [16, 8] [1, 1]
// CHECK: %[[ACC10:.*]] = tensor.extract_slice %[[ACC]][16, 0] [16, 8] [1, 1]
// CHECK: %[[LHS10:.*]] = tensor.extract_slice %[[LHS]][16, 0] [16, 8] [1, 1]
// CHECK: %[[PART10:.*]] = spruce.mma %[[LHS10]], %{{.*}}, %[[ACC10]]
// CHECK: %[[LHS11:.*]] = tensor.extract_slice %[[LHS]][16, 8] [16, 8] [1, 1]
// CHECK: %[[TILE10:.*]] = spruce.mma %[[LHS11]], %{{.*}}, %[[PART10]]
// CHECK: %[[RESULT10:.*]] = tensor.insert_slice %[[TILE10]] into %[[RESULT01]][16, 0] [16, 8] [1, 1]
// CHECK: %[[ACC11:.*]] = tensor.extract_slice %[[ACC]][16, 8] [16, 8] [1, 1]
// CHECK: %[[PART11:.*]] = spruce.mma %{{.*}}, %{{.*}}, %[[ACC11]]
// CHECK: %[[TILE11:.*]] = spruce.mma %{{.*}}, %{{.*}}, %[[PART11]]
// CHECK: %[[RESULT11:.*]] = tensor.insert_slice %[[TILE11]] into %[[RESULT10]][16, 8] [16, 8] [1, 1]
// CHECK: return %[[RESULT11]] : tensor<32x16xf32>
func.func @multiple_tiles(%lhs: tensor<32x16xf32>, %rhs: tensor<16x16xf32>, %acc: tensor<32x16xf32>) -> tensor<32x16xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<32x16xf32>, tensor<16x16xf32>, tensor<32x16xf32> -> tensor<32x16xf32>
  return %result : tensor<32x16xf32>
}

// CHECK-LABEL: func.func @empty_reduction(
// CHECK-SAME: %{{.*}}: tensor<3x0xf32>, %{{.*}}: tensor<0x7xf32>, %[[ACC:.*]]: tensor<3x7xf32>
// CHECK-NEXT: return %[[ACC]] : tensor<3x7xf32>
func.func @empty_reduction(%lhs: tensor<3x0xf32>, %rhs: tensor<0x7xf32>, %acc: tensor<3x7xf32>) -> tensor<3x7xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<3x0xf32>, tensor<0x7xf32>, tensor<3x7xf32> -> tensor<3x7xf32>
  return %result : tensor<3x7xf32>
}

// CHECK-LABEL: func.func @empty_rows(
// CHECK-SAME: %{{.*}}: tensor<0x5xf32>, %{{.*}}: tensor<5x7xf32>, %[[ACC:.*]]: tensor<0x7xf32>
// CHECK-NEXT: return %[[ACC]] : tensor<0x7xf32>
func.func @empty_rows(%lhs: tensor<0x5xf32>, %rhs: tensor<5x7xf32>, %acc: tensor<0x7xf32>) -> tensor<0x7xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<0x5xf32>, tensor<5x7xf32>, tensor<0x7xf32> -> tensor<0x7xf32>
  return %result : tensor<0x7xf32>
}

// CHECK-LABEL: func.func @empty_columns(
// CHECK-SAME: %{{.*}}: tensor<3x5xf32>, %{{.*}}: tensor<5x0xf32>, %[[ACC:.*]]: tensor<3x0xf32>
// CHECK-NEXT: return %[[ACC]] : tensor<3x0xf32>
func.func @empty_columns(%lhs: tensor<3x5xf32>, %rhs: tensor<5x0xf32>, %acc: tensor<3x0xf32>) -> tensor<3x0xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<3x5xf32>, tensor<5x0xf32>, tensor<3x0xf32> -> tensor<3x0xf32>
  return %result : tensor<3x0xf32>
}

// CHECK-LABEL: func.func @no_mma(
// CHECK: return
func.func @no_mma() {
  return
}
