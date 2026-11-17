// RUN: spruce-opt -convert-spruce-to-vector -split-input-file -verify-diagnostics %s

func.func @unsupported_inputs(%lhs: tensor<16x8xf16>, %rhs: tensor<8x8xf16>, %acc: tensor<16x8xf32>) -> tensor<16x8xf32> {
  // expected-error@+1 {{failed to legalize operation 'spruce.mma' that was explicitly marked illegal}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<16x8xf16>, tensor<8x8xf16>, tensor<16x8xf32> -> tensor<16x8xf32>
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
