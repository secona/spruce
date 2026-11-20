// RUN: spruce-opt -convert-linalg-to-spruce -verify-roundtrip %s | FileCheck %s

// CHECK-DAG: #[[TARGET:.*]] = #spruce.target<{m = 16, n = 8, k = 8}>
// CHECK-DAG: #[[BSR:.*]] = #sparse_tensor.encoding
#BSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (
    d0 floordiv 16 : dense,
    d1 floordiv 8 : compressed,
    d0 mod 16 : dense,
    d1 mod 8 : dense
  )
}>

// CHECK-LABEL: func.func @spgemm
// CHECK-SAME:    %[[A:[^:]+]]: tensor<32x16xf32, #[[BSR]]>,
// CHECK-SAME:    %[[B:[^:]+]]: tensor<16x8xf32>
// CHECK-SAME:  ) -> tensor<32x8xf32> {
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<32x8xf32>
// CHECK:         %[[C:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<32x8xf32>) -> tensor<32x8xf32>
// CHECK:         %[[RESULT:.*]] = spruce.kernel(
// CHECK-SAME:      %[[A_D:.*]] = %[[A]] : tensor<32x16xf32, #[[BSR]]>,
// CHECK-SAME:      %[[B_D:.*]] = %[[B]] : tensor<16x8xf32>,
// CHECK-SAME:      %[[C_D:.*]] = %[[C]] : tensor<32x8xf32>
// CHECK:             spruce.yield %[[C_D]] : tensor<32x8xf32>
// CHECK:         } -> (tensor<32x8xf32>) {spruce.target = #[[TARGET]]}
// CHECK:         return %[[RESULT]] : tensor<32x8xf32>
// CHECK:      }

func.func @spgemm(%A: tensor<32x16xf32, #BSR>,
                  %B: tensor<16x8xf32>) -> tensor<32x8xf32> {
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x8xf32>
  %C = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x8xf32>) -> tensor<32x8xf32>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i,k,j) -> (i,j)>,
      affine_map<(i,k,j) -> (j,k)>,
      affine_map<(i,k,j) -> (i,k)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B: tensor<32x16xf32, #BSR>, tensor<16x8xf32>)
    outs(%C: tensor<32x8xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %product = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %product : f32
      linalg.yield %sum : f32
  } -> tensor<32x8xf32>
  return %result : tensor<32x8xf32>
}
