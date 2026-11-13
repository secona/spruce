// RUN: spruce-opt -spruce-tile-mma -split-input-file -verify-diagnostics %s

func.func @unsupported_type(%lhs: tensor<16x8xf16>, %rhs: tensor<8x8xf16>, %acc: tensor<16x8xf32>) -> tensor<16x8xf32> {
  // expected-error@+1 {{spruce-tile-mma currently supports only f32 operands and result}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<16x8xf16>, tensor<8x8xf16>, tensor<16x8xf32> -> tensor<16x8xf32>
  return %result : tensor<16x8xf32>
}

// -----

func.func @unsupported_m(%lhs: tensor<17x8xf32>, %rhs: tensor<8x8xf32>, %acc: tensor<17x8xf32>) -> tensor<17x8xf32> {
  // expected-error@+1 {{spruce-tile-mma requires M to be a multiple of 16 and N and K to be multiples of 8}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<17x8xf32>, tensor<8x8xf32>, tensor<17x8xf32> -> tensor<17x8xf32>
  return %result : tensor<17x8xf32>
}

// -----

func.func @unsupported_n(%lhs: tensor<16x8xf32>, %rhs: tensor<8x9xf32>, %acc: tensor<16x9xf32>) -> tensor<16x9xf32> {
  // expected-error@+1 {{spruce-tile-mma requires M to be a multiple of 16 and N and K to be multiples of 8}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<16x8xf32>, tensor<8x9xf32>, tensor<16x9xf32> -> tensor<16x9xf32>
  return %result : tensor<16x9xf32>
}

// -----

func.func @unsupported_k(%lhs: tensor<16x9xf32>, %rhs: tensor<9x8xf32>, %acc: tensor<16x8xf32>) -> tensor<16x8xf32> {
  // expected-error@+1 {{spruce-tile-mma requires M to be a multiple of 16 and N and K to be multiples of 8}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<16x9xf32>, tensor<9x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
  return %result : tensor<16x8xf32>
}

// -----

func.func @m_smaller_than_tile(%lhs: tensor<15x8xf32>, %rhs: tensor<8x8xf32>, %acc: tensor<15x8xf32>) -> tensor<15x8xf32> {
  // expected-error@+1 {{spruce-tile-mma requires M to be a multiple of 16 and N and K to be multiples of 8}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<15x8xf32>, tensor<8x8xf32>, tensor<15x8xf32> -> tensor<15x8xf32>
  return %result : tensor<15x8xf32>
}

// -----

func.func @n_smaller_than_tile(%lhs: tensor<16x8xf32>, %rhs: tensor<8x7xf32>, %acc: tensor<16x7xf32>) -> tensor<16x7xf32> {
  // expected-error@+1 {{spruce-tile-mma requires M to be a multiple of 16 and N and K to be multiples of 8}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<16x8xf32>, tensor<8x7xf32>, tensor<16x7xf32> -> tensor<16x7xf32>
  return %result : tensor<16x7xf32>
}

// -----

func.func @k_smaller_than_tile(%lhs: tensor<16x7xf32>, %rhs: tensor<7x8xf32>, %acc: tensor<16x8xf32>) -> tensor<16x8xf32> {
  // expected-error@+1 {{spruce-tile-mma requires M to be a multiple of 16 and N and K to be multiples of 8}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<16x7xf32>, tensor<7x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
  return %result : tensor<16x8xf32>
}

// -----

func.func @multiple_tiles_with_remainders(%lhs: tensor<33x17xf32>, %rhs: tensor<17x17xf32>, %acc: tensor<33x17xf32>) -> tensor<33x17xf32> {
  // expected-error@+1 {{spruce-tile-mma requires M to be a multiple of 16 and N and K to be multiples of 8}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<33x17xf32>, tensor<17x17xf32>, tensor<33x17xf32> -> tensor<33x17xf32>
  return %result : tensor<33x17xf32>
}
