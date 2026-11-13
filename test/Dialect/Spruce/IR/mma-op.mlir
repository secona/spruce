// RUN: spruce-opt -verify-roundtrip %s | FileCheck %s

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
