// RUN: spruce-opt -spruce-tile-mma -split-input-file -verify-diagnostics %s

func.func @unsupported_type(%lhs: tensor<16x8xf16>, %rhs: tensor<8x8xf16>, %acc: tensor<16x8xf32>) -> tensor<16x8xf32> {
  // expected-error@+1 {{failed to legalize operation 'spruce.mma' that was explicitly marked illegal}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<16x8xf16>, tensor<8x8xf16>, tensor<16x8xf32> -> tensor<16x8xf32>
  return %result : tensor<16x8xf32>
}

// -----

func.func @unsupported_remainder(%lhs: tensor<17x9xf16>, %rhs: tensor<9x9xf16>, %acc: tensor<17x9xf32>) -> tensor<17x9xf32> {
  // expected-error@+1 {{failed to legalize operation 'spruce.mma' that was explicitly marked illegal}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<17x9xf16>, tensor<9x9xf16>, tensor<17x9xf32> -> tensor<17x9xf32>
  return %result : tensor<17x9xf32>
}

// -----

func.func @unsupported_rhs(%lhs: tensor<16x8xf32>, %rhs: tensor<8x8xf16>, %acc: tensor<16x8xf32>) -> tensor<16x8xf32> {
  // expected-error@+1 {{failed to legalize operation 'spruce.mma' that was explicitly marked illegal}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<16x8xf32>, tensor<8x8xf16>, tensor<16x8xf32> -> tensor<16x8xf32>
  return %result : tensor<16x8xf32>
}

// -----

func.func @unsupported_accumulator(%lhs: tensor<16x8xf32>, %rhs: tensor<8x8xf32>, %acc: tensor<16x8xf16>) -> tensor<16x8xf16> {
  // expected-error@+1 {{failed to legalize operation 'spruce.mma' that was explicitly marked illegal}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf16> -> tensor<16x8xf16>
  return %result : tensor<16x8xf16>
}

// -----

func.func @unsupported_result(%lhs: tensor<16x8xf32>, %rhs: tensor<8x8xf32>, %acc: tensor<16x8xf32>) -> tensor<16x8xf16> {
  // expected-error@+1 {{failed to legalize operation 'spruce.mma' that was explicitly marked illegal}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf16>
  return %result : tensor<16x8xf16>
}

// -----

func.func @unsupported_empty_reduction(%lhs: tensor<3x0xf16>, %rhs: tensor<0x7xf16>, %acc: tensor<3x7xf32>) -> tensor<3x7xf32> {
  // expected-error@+1 {{failed to legalize operation 'spruce.mma' that was explicitly marked illegal}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<3x0xf16>, tensor<0x7xf16>, tensor<3x7xf32> -> tensor<3x7xf32>
  return %result : tensor<3x7xf32>
}
