// RUN: spruce-opt -spruce-tile-mma -verify-roundtrip %s | FileCheck %s
// RUN: spruce-opt -spruce-tile-mma -spruce-tile-mma -verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: func.func @m_remainder(
// CHECK-SAME:      %[[LHS:.*]]: tensor<17x8xf32>,
// CHECK-SAME:      %[[RHS:.*]]: tensor<8x8xf32>,
// CHECK-SAME:      %[[ACC:.*]]: tensor<17x8xf32>)
// CHECK:         %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32

// CHECK:         %[[PAD_LHS:.*]] = tensor.pad %[[LHS]] low[0, 0] high[15, 0]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<17x8xf32> to tensor<32x8xf32>
// CHECK:         %[[PAD_ACC:.*]] = tensor.pad %[[ACC]] low[0, 0] high[15, 0]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<17x8xf32> to tensor<32x8xf32>

// CHECK:         %[[ACC0:.*]] = tensor.extract_slice %[[PAD_ACC]][0, 0] [16, 8] [1, 1]
// CHECK:         %[[LHS0:.*]] = tensor.extract_slice %[[PAD_LHS]][0, 0] [16, 8] [1, 1]
// CHECK:         %[[TILE0:.*]] = spruce.mma %[[LHS0]], %[[RHS]], %[[ACC0]]
// CHECK:         %[[OUT0:.*]] = tensor.insert_slice %[[TILE0]] into %[[PAD_ACC]][0, 0] [16, 8] [1, 1]

// CHECK:         %[[ACC1:.*]] = tensor.extract_slice %[[PAD_ACC]][16, 0] [16, 8] [1, 1]
// CHECK:         %[[LHS1:.*]] = tensor.extract_slice %[[PAD_LHS]][16, 0] [16, 8] [1, 1]
// CHECK:         %[[TILE1:.*]] = spruce.mma %[[LHS1]], %[[RHS]], %[[ACC1]]
// CHECK:         %[[OUT1:.*]] = tensor.insert_slice %[[TILE1]] into %[[OUT0]][16, 0] [16, 8] [1, 1]

// CHECK:         %[[RESULT:.*]] = tensor.extract_slice %[[OUT1]][0, 0] [17, 8] [1, 1] : tensor<32x8xf32> to tensor<17x8xf32>
// CHECK:         return %[[RESULT]] : tensor<17x8xf32>

func.func @m_remainder(%lhs: tensor<17x8xf32>, %rhs: tensor<8x8xf32>, %acc: tensor<17x8xf32>) -> tensor<17x8xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<17x8xf32>, tensor<8x8xf32>, tensor<17x8xf32> -> tensor<17x8xf32>
  return %result : tensor<17x8xf32>
}

// CHECK-LABEL: func.func @n_remainder(
// CHECK-SAME:      %[[LHS:.*]]: tensor<16x8xf32>,
// CHECK-SAME:      %[[RHS:.*]]: tensor<8x9xf32>,
// CHECK-SAME:      %[[ACC:.*]]: tensor<16x9xf32>)
// CHECK:         %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32

// CHECK:         %[[PAD_RHS:.*]] = tensor.pad %[[RHS]] low[0, 0] high[0, 7]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<8x9xf32> to tensor<8x16xf32>
// CHECK:         %[[PAD_ACC:.*]] = tensor.pad %[[ACC]] low[0, 0] high[0, 7]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<16x9xf32> to tensor<16x16xf32>

// CHECK:         %[[ACC0:.*]] = tensor.extract_slice %[[PAD_ACC]][0, 0] [16, 8] [1, 1]
// CHECK:         %[[RHS0:.*]] = tensor.extract_slice %[[PAD_RHS]][0, 0] [8, 8] [1, 1]
// CHECK:         %[[TILE0:.*]] = spruce.mma %[[LHS]], %[[RHS0]], %[[ACC0]]
// CHECK:         %[[OUT0:.*]] = tensor.insert_slice %[[TILE0]] into %[[PAD_ACC]][0, 0] [16, 8] [1, 1]

// CHECK:         %[[ACC1:.*]] = tensor.extract_slice %[[PAD_ACC]][0, 8] [16, 8] [1, 1]
// CHECK:         %[[RHS1:.*]] = tensor.extract_slice %[[PAD_RHS]][0, 8] [8, 8] [1, 1]
// CHECK:         %[[TILE1:.*]] = spruce.mma %[[LHS]], %[[RHS1]], %[[ACC1]]
// CHECK:         %[[OUT1:.*]] = tensor.insert_slice %[[TILE1]] into %[[OUT0]][0, 8] [16, 8] [1, 1]

// CHECK:         %[[RESULT:.*]] = tensor.extract_slice %[[OUT1]][0, 0] [16, 9] [1, 1] : tensor<16x16xf32> to tensor<16x9xf32>
// CHECK:         return %[[RESULT]] : tensor<16x9xf32>

func.func @n_remainder(%lhs: tensor<16x8xf32>, %rhs: tensor<8x9xf32>, %acc: tensor<16x9xf32>) -> tensor<16x9xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<16x8xf32>, tensor<8x9xf32>, tensor<16x9xf32> -> tensor<16x9xf32>
  return %result : tensor<16x9xf32>
}

// CHECK-LABEL: func.func @k_remainder(
// CHECK-SAME:      %[[LHS:.*]]: tensor<16x9xf32>,
// CHECK-SAME:      %[[RHS:.*]]: tensor<9x8xf32>,
// CHECK-SAME:      %[[ACC:.*]]: tensor<16x8xf32>)
// CHECK:         %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32

// CHECK:         %[[PAD_LHS:.*]] = tensor.pad %[[LHS]] low[0, 0] high[0, 7]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<16x9xf32> to tensor<16x16xf32>
// CHECK:         %[[PAD_RHS:.*]] = tensor.pad %[[RHS]] low[0, 0] high[7, 0]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<9x8xf32> to tensor<16x8xf32>

// CHECK:         %[[LHS0:.*]] = tensor.extract_slice %[[PAD_LHS]][0, 0] [16, 8] [1, 1]
// CHECK:         %[[RHS0:.*]] = tensor.extract_slice %[[PAD_RHS]][0, 0] [8, 8] [1, 1]
// CHECK:         %[[PART:.*]] = spruce.mma %[[LHS0]], %[[RHS0]], %[[ACC]]

// CHECK:         %[[LHS1:.*]] = tensor.extract_slice %[[PAD_LHS]][0, 8] [16, 8] [1, 1]
// CHECK:         %[[RHS1:.*]] = tensor.extract_slice %[[PAD_RHS]][8, 0] [8, 8] [1, 1]
// CHECK:         %[[RESULT:.*]] = spruce.mma %[[LHS1]], %[[RHS1]], %[[PART]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
// CHECK-NEXT:    return %[[RESULT]] : tensor<16x8xf32>

func.func @k_remainder(%lhs: tensor<16x9xf32>, %rhs: tensor<9x8xf32>, %acc: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<16x9xf32>, tensor<9x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
  return %result : tensor<16x8xf32>
}

// CHECK-LABEL: func.func @m_smaller_than_tile(
// CHECK-SAME:      %[[LHS:.*]]: tensor<15x8xf32>,
// CHECK-SAME:      %[[RHS:.*]]: tensor<8x8xf32>,
// CHECK-SAME:      %[[ACC:.*]]: tensor<15x8xf32>)
// CHECK:         %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32

// CHECK:         %[[PAD_LHS:.*]] = tensor.pad %[[LHS]] low[0, 0] high[1, 0]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<15x8xf32> to tensor<16x8xf32>
// CHECK:         %[[PAD_ACC:.*]] = tensor.pad %[[ACC]] low[0, 0] high[1, 0]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<15x8xf32> to tensor<16x8xf32>

// CHECK:         %[[TILE:.*]] = spruce.mma %[[PAD_LHS]], %[[RHS]], %[[PAD_ACC]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>

// CHECK:         %[[RESULT:.*]] = tensor.extract_slice %[[TILE]][0, 0] [15, 8] [1, 1] : tensor<16x8xf32> to tensor<15x8xf32>
// CHECK:         return %[[RESULT]] : tensor<15x8xf32>

func.func @m_smaller_than_tile(%lhs: tensor<15x8xf32>, %rhs: tensor<8x8xf32>, %acc: tensor<15x8xf32>) -> tensor<15x8xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<15x8xf32>, tensor<8x8xf32>, tensor<15x8xf32> -> tensor<15x8xf32>
  return %result : tensor<15x8xf32>
}

// CHECK-LABEL: func.func @n_smaller_than_tile(
// CHECK-SAME:      %[[LHS:.*]]: tensor<16x8xf32>,
// CHECK-SAME:      %[[RHS:.*]]: tensor<8x7xf32>,
// CHECK-SAME:      %[[ACC:.*]]: tensor<16x7xf32>)
// CHECK:         %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32

// CHECK:         %[[PAD_RHS:.*]] = tensor.pad %[[RHS]] low[0, 0] high[0, 1]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<8x7xf32> to tensor<8x8xf32>
// CHECK:         %[[PAD_ACC:.*]] = tensor.pad %[[ACC]] low[0, 0] high[0, 1]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<16x7xf32> to tensor<16x8xf32>

// CHECK:         %[[TILE:.*]] = spruce.mma %[[LHS]], %[[PAD_RHS]], %[[PAD_ACC]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>

// CHECK:         %[[RESULT:.*]] = tensor.extract_slice %[[TILE]][0, 0] [16, 7] [1, 1] : tensor<16x8xf32> to tensor<16x7xf32>
// CHECK:         return %[[RESULT]] : tensor<16x7xf32>

func.func @n_smaller_than_tile(%lhs: tensor<16x8xf32>, %rhs: tensor<8x7xf32>, %acc: tensor<16x7xf32>) -> tensor<16x7xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<16x8xf32>, tensor<8x7xf32>, tensor<16x7xf32> -> tensor<16x7xf32>
  return %result : tensor<16x7xf32>
}

// CHECK-LABEL: func.func @k_smaller_than_tile(
// CHECK-SAME:      %[[LHS:.*]]: tensor<16x7xf32>,
// CHECK-SAME:      %[[RHS:.*]]: tensor<7x8xf32>,
// CHECK-SAME:      %[[ACC:.*]]: tensor<16x8xf32>)
// CHECK:         %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32

// CHECK:         %[[PAD_LHS:.*]] = tensor.pad %[[LHS]] low[0, 0] high[0, 1]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<16x7xf32> to tensor<16x8xf32>
// CHECK:         %[[PAD_RHS:.*]] = tensor.pad %[[RHS]] low[0, 0] high[1, 0]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<7x8xf32> to tensor<8x8xf32>

// CHECK:         %[[RESULT:.*]] = spruce.mma %[[PAD_LHS]], %[[PAD_RHS]], %[[ACC]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
// CHECK-NEXT:    return %[[RESULT]] : tensor<16x8xf32>

func.func @k_smaller_than_tile(%lhs: tensor<16x7xf32>, %rhs: tensor<7x8xf32>, %acc: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<16x7xf32>, tensor<7x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
  return %result : tensor<16x8xf32>
}

// CHECK-LABEL: func.func @all_smaller_than_tile(
// CHECK-SAME:      %[[LHS:.*]]: tensor<3x5xf32>,
// CHECK-SAME:      %[[RHS:.*]]: tensor<5x7xf32>,
// CHECK-SAME:      %[[ACC:.*]]: tensor<3x7xf32>)
// CHECK:         %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32

// CHECK:         %[[PAD_LHS:.*]] = tensor.pad %[[LHS]] low[0, 0] high[13, 3]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<3x5xf32> to tensor<16x8xf32>
// CHECK:         %[[PAD_RHS:.*]] = tensor.pad %[[RHS]] low[0, 0] high[3, 1]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<5x7xf32> to tensor<8x8xf32>
// CHECK:         %[[PAD_ACC:.*]] = tensor.pad %[[ACC]] low[0, 0] high[13, 1]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<3x7xf32> to tensor<16x8xf32>

// CHECK:         %[[TILE:.*]] = spruce.mma %[[PAD_LHS]], %[[PAD_RHS]], %[[PAD_ACC]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>

// CHECK:         %[[RESULT:.*]] = tensor.extract_slice %[[TILE]][0, 0] [3, 7] [1, 1] : tensor<16x8xf32> to tensor<3x7xf32>
// CHECK:         return %[[RESULT]] : tensor<3x7xf32>

func.func @all_smaller_than_tile(%lhs: tensor<3x5xf32>, %rhs: tensor<5x7xf32>, %acc: tensor<3x7xf32>) -> tensor<3x7xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<3x5xf32>, tensor<5x7xf32>, tensor<3x7xf32> -> tensor<3x7xf32>
  return %result : tensor<3x7xf32>
}

// CHECK-LABEL: func.func @scalar(
// CHECK-SAME:      %[[LHS:[^:]+]]: tensor<1x1xf32>,
// CHECK-SAME:      %[[RHS:[^:]+]]: tensor<1x1xf32>,
// CHECK-SAME:      %[[ACC:[^:]+]]: tensor<1x1xf32>)
// CHECK:         %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32

// CHECK:         %[[PAD_LHS:.*]] = tensor.pad %[[LHS]] low[0, 0] high[15, 7]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<1x1xf32> to tensor<16x8xf32>
// CHECK:         %[[PAD_RHS:.*]] = tensor.pad %[[RHS]] low[0, 0] high[7, 7]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<1x1xf32> to tensor<8x8xf32>
// CHECK:         %[[PAD_ACC:.*]] = tensor.pad %[[ACC]] low[0, 0] high[15, 7]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<1x1xf32> to tensor<16x8xf32>

// CHECK:         %[[TILE:.*]] = spruce.mma %[[PAD_LHS]], %[[PAD_RHS]], %[[PAD_ACC]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>

// CHECK:         %[[RESULT:.*]] = tensor.extract_slice %[[TILE]][0, 0] [1, 1] [1, 1] : tensor<16x8xf32> to tensor<1x1xf32>
// CHECK:         return %[[RESULT]] : tensor<1x1xf32>

func.func @scalar(%lhs: tensor<1x1xf32>, %rhs: tensor<1x1xf32>, %acc: tensor<1x1xf32>) -> tensor<1x1xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<1x1xf32>, tensor<1x1xf32>, tensor<1x1xf32> -> tensor<1x1xf32>
  return %result : tensor<1x1xf32>
}

// CHECK-LABEL: func.func @multiple_tiles_with_remainders(
// CHECK-SAME:      %[[LHS:.*]]: tensor<33x17xf32>,
// CHECK-SAME:      %[[RHS:.*]]: tensor<17x17xf32>,
// CHECK-SAME:      %[[ACC:.*]]: tensor<33x17xf32>)
// CHECK:         %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32

// CHECK:         %[[PAD_LHS:.*]] = tensor.pad %[[LHS]] low[0, 0] high[15, 7]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<33x17xf32> to tensor<48x24xf32>
// CHECK:         %[[PAD_RHS:.*]] = tensor.pad %[[RHS]] low[0, 0] high[7, 7]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<17x17xf32> to tensor<24x24xf32>
// CHECK:         %[[PAD_ACC:.*]] = tensor.pad %[[ACC]] low[0, 0] high[15, 7]
// CHECK:           tensor.yield %[[ZERO]] : f32
// CHECK:         } : tensor<33x17xf32> to tensor<48x24xf32>

// CHECK:         %[[ACC0:.*]] = tensor.extract_slice %[[PAD_ACC]][0, 0] [16, 8] [1, 1]
// CHECK:         %[[LHS0_0:.*]] = tensor.extract_slice %[[PAD_LHS]][0, 0] [16, 8] [1, 1]
// CHECK:         %[[RHS0_0:.*]] = tensor.extract_slice %[[PAD_RHS]][0, 0] [8, 8] [1, 1]
// CHECK:         %[[PART0_0:.*]] = spruce.mma %[[LHS0_0]], %[[RHS0_0]], %[[ACC0]]

// CHECK:         %[[LHS0_1:.*]] = tensor.extract_slice %[[PAD_LHS]][0, 8] [16, 8] [1, 1]
// CHECK:         %[[RHS0_1:.*]] = tensor.extract_slice %[[PAD_RHS]][8, 0] [8, 8] [1, 1]
// CHECK:         %[[PART0_1:.*]] = spruce.mma %[[LHS0_1]], %[[RHS0_1]], %[[PART0_0]]

// CHECK:         %[[LHS0_2:.*]] = tensor.extract_slice %[[PAD_LHS]][0, 16] [16, 8] [1, 1]
// CHECK:         %[[RHS0_2:.*]] = tensor.extract_slice %[[PAD_RHS]][16, 0] [8, 8] [1, 1]
// CHECK:         %[[TILE0:.*]] = spruce.mma %[[LHS0_2]], %[[RHS0_2]], %[[PART0_1]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
// CHECK:         %[[OUT0:.*]] = tensor.insert_slice %[[TILE0]] into %[[PAD_ACC]][0, 0] [16, 8] [1, 1]

// CHECK:         %[[ACC1:.*]] = tensor.extract_slice %[[PAD_ACC]][0, 8] [16, 8] [1, 1]
// CHECK:         %[[LHS1_0:.*]] = tensor.extract_slice %[[PAD_LHS]][0, 0] [16, 8] [1, 1]
// CHECK:         %[[RHS1_0:.*]] = tensor.extract_slice %[[PAD_RHS]][0, 8] [8, 8] [1, 1]
// CHECK:         %[[PART1_0:.*]] = spruce.mma %[[LHS1_0]], %[[RHS1_0]], %[[ACC1]]

// CHECK:         %[[LHS1_1:.*]] = tensor.extract_slice %[[PAD_LHS]][0, 8] [16, 8] [1, 1]
// CHECK:         %[[RHS1_1:.*]] = tensor.extract_slice %[[PAD_RHS]][8, 8] [8, 8] [1, 1]
// CHECK:         %[[PART1_1:.*]] = spruce.mma %[[LHS1_1]], %[[RHS1_1]], %[[PART1_0]]

// CHECK:         %[[LHS1_2:.*]] = tensor.extract_slice %[[PAD_LHS]][0, 16] [16, 8] [1, 1]
// CHECK:         %[[RHS1_2:.*]] = tensor.extract_slice %[[PAD_RHS]][16, 8] [8, 8] [1, 1]
// CHECK:         %[[TILE1:.*]] = spruce.mma %[[LHS1_2]], %[[RHS1_2]], %[[PART1_1]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
// CHECK:         %[[OUT1:.*]] = tensor.insert_slice %[[TILE1]] into %[[OUT0]][0, 8] [16, 8] [1, 1]

// CHECK:         %[[ACC2:.*]] = tensor.extract_slice %[[PAD_ACC]][0, 16] [16, 8] [1, 1]
// CHECK:         %[[LHS2_0:.*]] = tensor.extract_slice %[[PAD_LHS]][0, 0] [16, 8] [1, 1]
// CHECK:         %[[RHS2_0:.*]] = tensor.extract_slice %[[PAD_RHS]][0, 16] [8, 8] [1, 1]
// CHECK:         %[[PART2_0:.*]] = spruce.mma %[[LHS2_0]], %[[RHS2_0]], %[[ACC2]]

// CHECK:         %[[LHS2_1:.*]] = tensor.extract_slice %[[PAD_LHS]][0, 8] [16, 8] [1, 1]
// CHECK:         %[[RHS2_1:.*]] = tensor.extract_slice %[[PAD_RHS]][8, 16] [8, 8] [1, 1]
// CHECK:         %[[PART2_1:.*]] = spruce.mma %[[LHS2_1]], %[[RHS2_1]], %[[PART2_0]]

// CHECK:         %[[LHS2_2:.*]] = tensor.extract_slice %[[PAD_LHS]][0, 16] [16, 8] [1, 1]
// CHECK:         %[[RHS2_2:.*]] = tensor.extract_slice %[[PAD_RHS]][16, 16] [8, 8] [1, 1]
// CHECK:         %[[TILE2:.*]] = spruce.mma %[[LHS2_2]], %[[RHS2_2]], %[[PART2_1]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
// CHECK:         %[[OUT2:.*]] = tensor.insert_slice %[[TILE2]] into %[[OUT1]][0, 16] [16, 8] [1, 1]

// CHECK:         %[[ACC3:.*]] = tensor.extract_slice %[[PAD_ACC]][16, 0] [16, 8] [1, 1]
// CHECK:         %[[LHS3_0:.*]] = tensor.extract_slice %[[PAD_LHS]][16, 0] [16, 8] [1, 1]
// CHECK:         %[[RHS3_0:.*]] = tensor.extract_slice %[[PAD_RHS]][0, 0] [8, 8] [1, 1]
// CHECK:         %[[PART3_0:.*]] = spruce.mma %[[LHS3_0]], %[[RHS3_0]], %[[ACC3]]

// CHECK:         %[[LHS3_1:.*]] = tensor.extract_slice %[[PAD_LHS]][16, 8] [16, 8] [1, 1]
// CHECK:         %[[RHS3_1:.*]] = tensor.extract_slice %[[PAD_RHS]][8, 0] [8, 8] [1, 1]
// CHECK:         %[[PART3_1:.*]] = spruce.mma %[[LHS3_1]], %[[RHS3_1]], %[[PART3_0]]

// CHECK:         %[[LHS3_2:.*]] = tensor.extract_slice %[[PAD_LHS]][16, 16] [16, 8] [1, 1]
// CHECK:         %[[RHS3_2:.*]] = tensor.extract_slice %[[PAD_RHS]][16, 0] [8, 8] [1, 1]
// CHECK:         %[[TILE3:.*]] = spruce.mma %[[LHS3_2]], %[[RHS3_2]], %[[PART3_1]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
// CHECK:         %[[OUT3:.*]] = tensor.insert_slice %[[TILE3]] into %[[OUT2]][16, 0] [16, 8] [1, 1]

// CHECK:         %[[ACC4:.*]] = tensor.extract_slice %[[PAD_ACC]][16, 8] [16, 8] [1, 1]
// CHECK:         %[[LHS4_0:.*]] = tensor.extract_slice %[[PAD_LHS]][16, 0] [16, 8] [1, 1]
// CHECK:         %[[RHS4_0:.*]] = tensor.extract_slice %[[PAD_RHS]][0, 8] [8, 8] [1, 1]
// CHECK:         %[[PART4_0:.*]] = spruce.mma %[[LHS4_0]], %[[RHS4_0]], %[[ACC4]]

// CHECK:         %[[LHS4_1:.*]] = tensor.extract_slice %[[PAD_LHS]][16, 8] [16, 8] [1, 1]
// CHECK:         %[[RHS4_1:.*]] = tensor.extract_slice %[[PAD_RHS]][8, 8] [8, 8] [1, 1]
// CHECK:         %[[PART4_1:.*]] = spruce.mma %[[LHS4_1]], %[[RHS4_1]], %[[PART4_0]]

// CHECK:         %[[LHS4_2:.*]] = tensor.extract_slice %[[PAD_LHS]][16, 16] [16, 8] [1, 1]
// CHECK:         %[[RHS4_2:.*]] = tensor.extract_slice %[[PAD_RHS]][16, 8] [8, 8] [1, 1]
// CHECK:         %[[TILE4:.*]] = spruce.mma %[[LHS4_2]], %[[RHS4_2]], %[[PART4_1]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
// CHECK:         %[[OUT4:.*]] = tensor.insert_slice %[[TILE4]] into %[[OUT3]][16, 8] [16, 8] [1, 1]

// CHECK:         %[[ACC5:.*]] = tensor.extract_slice %[[PAD_ACC]][16, 16] [16, 8] [1, 1]
// CHECK:         %[[LHS5_0:.*]] = tensor.extract_slice %[[PAD_LHS]][16, 0] [16, 8] [1, 1]
// CHECK:         %[[RHS5_0:.*]] = tensor.extract_slice %[[PAD_RHS]][0, 16] [8, 8] [1, 1]
// CHECK:         %[[PART5_0:.*]] = spruce.mma %[[LHS5_0]], %[[RHS5_0]], %[[ACC5]]

// CHECK:         %[[LHS5_1:.*]] = tensor.extract_slice %[[PAD_LHS]][16, 8] [16, 8] [1, 1]
// CHECK:         %[[RHS5_1:.*]] = tensor.extract_slice %[[PAD_RHS]][8, 16] [8, 8] [1, 1]
// CHECK:         %[[PART5_1:.*]] = spruce.mma %[[LHS5_1]], %[[RHS5_1]], %[[PART5_0]]

// CHECK:         %[[LHS5_2:.*]] = tensor.extract_slice %[[PAD_LHS]][16, 16] [16, 8] [1, 1]
// CHECK:         %[[RHS5_2:.*]] = tensor.extract_slice %[[PAD_RHS]][16, 16] [8, 8] [1, 1]
// CHECK:         %[[TILE5:.*]] = spruce.mma %[[LHS5_2]], %[[RHS5_2]], %[[PART5_1]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
// CHECK:         %[[OUT5:.*]] = tensor.insert_slice %[[TILE5]] into %[[OUT4]][16, 16] [16, 8] [1, 1]

// CHECK:         %[[ACC6:.*]] = tensor.extract_slice %[[PAD_ACC]][32, 0] [16, 8] [1, 1]
// CHECK:         %[[LHS6_0:.*]] = tensor.extract_slice %[[PAD_LHS]][32, 0] [16, 8] [1, 1]
// CHECK:         %[[RHS6_0:.*]] = tensor.extract_slice %[[PAD_RHS]][0, 0] [8, 8] [1, 1]
// CHECK:         %[[PART6_0:.*]] = spruce.mma %[[LHS6_0]], %[[RHS6_0]], %[[ACC6]]

// CHECK:         %[[LHS6_1:.*]] = tensor.extract_slice %[[PAD_LHS]][32, 8] [16, 8] [1, 1]
// CHECK:         %[[RHS6_1:.*]] = tensor.extract_slice %[[PAD_RHS]][8, 0] [8, 8] [1, 1]
// CHECK:         %[[PART6_1:.*]] = spruce.mma %[[LHS6_1]], %[[RHS6_1]], %[[PART6_0]]

// CHECK:         %[[LHS6_2:.*]] = tensor.extract_slice %[[PAD_LHS]][32, 16] [16, 8] [1, 1]
// CHECK:         %[[RHS6_2:.*]] = tensor.extract_slice %[[PAD_RHS]][16, 0] [8, 8] [1, 1]
// CHECK:         %[[TILE6:.*]] = spruce.mma %[[LHS6_2]], %[[RHS6_2]], %[[PART6_1]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
// CHECK:         %[[OUT6:.*]] = tensor.insert_slice %[[TILE6]] into %[[OUT5]][32, 0] [16, 8] [1, 1]

// CHECK:         %[[ACC7:.*]] = tensor.extract_slice %[[PAD_ACC]][32, 8] [16, 8] [1, 1]
// CHECK:         %[[LHS7_0:.*]] = tensor.extract_slice %[[PAD_LHS]][32, 0] [16, 8] [1, 1]
// CHECK:         %[[RHS7_0:.*]] = tensor.extract_slice %[[PAD_RHS]][0, 8] [8, 8] [1, 1]
// CHECK:         %[[PART7_0:.*]] = spruce.mma %[[LHS7_0]], %[[RHS7_0]], %[[ACC7]]

// CHECK:         %[[LHS7_1:.*]] = tensor.extract_slice %[[PAD_LHS]][32, 8] [16, 8] [1, 1]
// CHECK:         %[[RHS7_1:.*]] = tensor.extract_slice %[[PAD_RHS]][8, 8] [8, 8] [1, 1]
// CHECK:         %[[PART7_1:.*]] = spruce.mma %[[LHS7_1]], %[[RHS7_1]], %[[PART7_0]]

// CHECK:         %[[LHS7_2:.*]] = tensor.extract_slice %[[PAD_LHS]][32, 16] [16, 8] [1, 1]
// CHECK:         %[[RHS7_2:.*]] = tensor.extract_slice %[[PAD_RHS]][16, 8] [8, 8] [1, 1]
// CHECK:         %[[TILE7:.*]] = spruce.mma %[[LHS7_2]], %[[RHS7_2]], %[[PART7_1]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
// CHECK:         %[[OUT7:.*]] = tensor.insert_slice %[[TILE7]] into %[[OUT6]][32, 8] [16, 8] [1, 1]

// CHECK:         %[[ACC8:.*]] = tensor.extract_slice %[[PAD_ACC]][32, 16] [16, 8] [1, 1]
// CHECK:         %[[LHS8_0:.*]] = tensor.extract_slice %[[PAD_LHS]][32, 0] [16, 8] [1, 1]
// CHECK:         %[[RHS8_0:.*]] = tensor.extract_slice %[[PAD_RHS]][0, 16] [8, 8] [1, 1]
// CHECK:         %[[PART8_0:.*]] = spruce.mma %[[LHS8_0]], %[[RHS8_0]], %[[ACC8]]

// CHECK:         %[[LHS8_1:.*]] = tensor.extract_slice %[[PAD_LHS]][32, 8] [16, 8] [1, 1]
// CHECK:         %[[RHS8_1:.*]] = tensor.extract_slice %[[PAD_RHS]][8, 16] [8, 8] [1, 1]
// CHECK:         %[[PART8_1:.*]] = spruce.mma %[[LHS8_1]], %[[RHS8_1]], %[[PART8_0]]

// CHECK:         %[[LHS8_2:.*]] = tensor.extract_slice %[[PAD_LHS]][32, 16] [16, 8] [1, 1]
// CHECK:         %[[RHS8_2:.*]] = tensor.extract_slice %[[PAD_RHS]][16, 16] [8, 8] [1, 1]
// CHECK:         %[[TILE8:.*]] = spruce.mma %[[LHS8_2]], %[[RHS8_2]], %[[PART8_1]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
// CHECK:         %[[OUT8:.*]] = tensor.insert_slice %[[TILE8]] into %[[OUT7]][32, 16] [16, 8] [1, 1]

// CHECK:         %[[RESULT:.*]] = tensor.extract_slice %[[OUT8]][0, 0] [33, 17] [1, 1] : tensor<48x24xf32> to tensor<33x17xf32>
// CHECK:         return %[[RESULT]] : tensor<33x17xf32>

func.func @multiple_tiles_with_remainders(%lhs: tensor<33x17xf32>, %rhs: tensor<17x17xf32>, %acc: tensor<33x17xf32>) -> tensor<33x17xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<33x17xf32>, tensor<17x17xf32>, tensor<33x17xf32> -> tensor<33x17xf32>
  return %result : tensor<33x17xf32>
}
