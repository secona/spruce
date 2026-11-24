// RUN: spruce-opt -one-shot-bufferize="bufferize-function-boundaries" -verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: func.func @live_acc(
// CHECK-SAME:    %[[LHS:[^:]+]]: memref<16x8xf16, {{.*}}>, %[[RHS:[^:]+]]: memref<8x8xf16, {{.*}}>, %[[ACC:[^:]+]]: memref
// CHECK:         %[[COPY:.*]] = memref.alloc()
// CHECK:         memref.copy %[[ACC]], %[[COPY]]
// CHECK:         spruce.mma %[[LHS]], %[[RHS]], %[[COPY]] : memref
// CHECK:         return %[[COPY]], %[[ACC]]
func.func @live_acc(%lhs: tensor<16x8xf16>, %rhs: tensor<8x8xf16>, %acc: tensor<16x8xf16>) -> (tensor<16x8xf16>, tensor<16x8xf16>) {
  %sum = spruce.mma %lhs, %rhs, %acc : tensor<16x8xf16>, tensor<8x8xf16>, tensor<16x8xf16> -> tensor<16x8xf16>
  return %sum, %acc : tensor<16x8xf16>, tensor<16x8xf16>
}
