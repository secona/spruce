// RUN: spruce-opt -spruce-tile-mma -split-input-file -verify-diagnostics %s

func.func @unsupported_type(%lhs: tensor<16x8xf16>, %rhs: tensor<8x8xf16>, %acc: tensor<16x8xf32>) -> tensor<16x8xf32> {
  // expected-error@+1 {{spruce-tile-mma currently supports only f32 operands and result}}
  %result = spruce.mma %lhs, %rhs, %acc : tensor<16x8xf16>, tensor<8x8xf16>, tensor<16x8xf32> -> tensor<16x8xf32>
  return %result : tensor<16x8xf32>
}
