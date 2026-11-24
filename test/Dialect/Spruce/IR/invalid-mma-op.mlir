// RUN: spruce-opt -split-input-file -verify-diagnostics %s

func.func @lhs_rank(%lhs: tensor<f16>, %rhs: tensor<5x7xf16>, %acc: tensor<3x7xf32>) -> tensor<3x7xf32> {
  // expected-error@+1 {{expected lhs to be a rank-2 tensor}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<f16>, tensor<5x7xf16>, tensor<3x7xf32> -> tensor<3x7xf32>
  return %result : tensor<3x7xf32>
}

// -----

func.func @rhs_rank(%lhs: tensor<3x5xf16>, %rhs: tensor<1x5x7xf16>, %acc: tensor<3x7xf32>) -> tensor<3x7xf32> {
  // expected-error@+1 {{expected rhs to be a rank-2 tensor}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<3x5xf16>, tensor<1x5x7xf16>, tensor<3x7xf32> -> tensor<3x7xf32>
  return %result : tensor<3x7xf32>
}

// -----

func.func @acc_rank(%lhs: tensor<3x5xf16>, %rhs: tensor<5x7xf16>, %acc: tensor<21xf32>) -> tensor<21xf32> {
  // expected-error@+1 {{expected acc to be a rank-2 tensor}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<3x5xf16>, tensor<5x7xf16>, tensor<21xf32> -> tensor<21xf32>
  return %result : tensor<21xf32>
}

// -----

func.func @result_shape(%lhs: tensor<3x5xf16>, %rhs: tensor<5x7xf16>, %acc: tensor<3x7xf32>) -> tensor<7x3xf32> {
  // expected-error@+1 {{expected result and acc to have the same shape}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<3x5xf16>, tensor<5x7xf16>, tensor<3x7xf32> -> tensor<7x3xf32>
  return %result : tensor<7x3xf32>
}

// -----

func.func @reduction_dimension(%lhs: tensor<3x5xf16>, %rhs: tensor<6x7xf16>, %acc: tensor<3x7xf32>) -> tensor<3x7xf32> {
  // expected-error@+1 {{expected lhs dimension 1 and rhs dimension 0 to match (reduction dimension K)}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<3x5xf16>, tensor<6x7xf16>, tensor<3x7xf32> -> tensor<3x7xf32>
  return %result : tensor<3x7xf32>
}

// -----

func.func @row_dimension(%lhs: tensor<3x5xf16>, %rhs: tensor<5x7xf16>, %acc: tensor<4x7xf32>) -> tensor<4x7xf32> {
  // expected-error@+1 {{expected lhs dimension 0 and acc dimension 0 to match (dimension M)}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<3x5xf16>, tensor<5x7xf16>, tensor<4x7xf32> -> tensor<4x7xf32>
  return %result : tensor<4x7xf32>
}

// -----

func.func @column_dimension(%lhs: tensor<3x5xf16>, %rhs: tensor<5x7xf16>, %acc: tensor<3x8xf32>) -> tensor<3x8xf32> {
  // expected-error@+1 {{expected rhs dimension 1 and acc dimension 1 to match (dimension N)}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<3x5xf16>, tensor<5x7xf16>, tensor<3x8xf32> -> tensor<3x8xf32>
  return %result : tensor<3x8xf32>
}

// -----

func.func @mixed_operands(%a: tensor<8x8xf32>, %b: memref<8x8xf32>) {
  // expected-error@+1 {{expected memref operands when there is no result}}
  spruce.mma %a, %b, %b : tensor<8x8xf32>, memref<8x8xf32>, memref<8x8xf32>
  return
}

// -----

func.func @buffer_result(%a: memref<8x8xf32>) {
  // expected-error@+1 {{expected tensor operands with a tensor result}}
  %r = spruce.mma %a, %a, %a : memref<8x8xf32>, memref<8x8xf32>, memref<8x8xf32> -> tensor<8x8xf32>
  return
}
