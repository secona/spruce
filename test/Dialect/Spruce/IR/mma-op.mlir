// RUN: spruce-opt -verify-roundtrip -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @mma
// CHECK: %[[RESULT:.*]] = spruce.mma %{{.*}}, %{{.*}}, %{{.*}} : tensor<3x5xf16>, tensor<5x7xf16>, tensor<3x7xf32> -> tensor<3x7xf32>

func.func @mma(
  %lhs: tensor<3x5xf16>,
  %rhs: tensor<5x7xf16>,
  %acc: tensor<3x7xf32>
) -> tensor<3x7xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<3x5xf16>, tensor<5x7xf16>, tensor<3x7xf32> -> tensor<3x7xf32>
  return %result : tensor<3x7xf32>
}

// -----

// CHECK: #[[CSR:.*]] = #sparse_tensor.encoding
#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: func.func @mma_sp
// CHECK: %[[RESULT:.*]] = spruce.mma %{{.*}}, %{{.*}}, %{{.*}} : tensor<3x5xf16, #[[CSR]]>, tensor<5x7xf16>, tensor<3x7xf32> -> tensor<3x7xf32>

func.func @mma_sp(
  %lhs: tensor<3x5xf16, #CSR>,
  %rhs: tensor<5x7xf16>,
  %acc: tensor<3x7xf32>
) -> tensor<3x7xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<3x5xf16, #CSR>, tensor<5x7xf16>, tensor<3x7xf32> -> tensor<3x7xf32>
  return %result : tensor<3x7xf32>
}
