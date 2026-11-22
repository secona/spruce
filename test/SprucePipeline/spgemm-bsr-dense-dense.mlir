// REQUIRES: enable-spgemm-bsr-dense-dense
// RUN: spruce-opt -spruce-pipeline -verify-roundtrip %s | FileCheck %s

// CHECK-DAG: #[[BSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 floordiv 16 : dense, d1 floordiv 8 : compressed, d0 mod 16 : dense, d1 mod 8 : dense) }>
// CHECK-DAG: #[[TARGET:.*]] = #spruce.target<{m = 16, n = 8, k = 8}>
#BSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (
    d0 floordiv 16 : dense,
    d1 floordiv 8 : compressed,
    d0 mod 16 : dense,
    d1 mod 8 : dense
  )
}>

// CHECK:       module {
// CHECK-LABEL:   func.func @spgemm(
// CHECK-SAME:        %[[A:[^:]+]]: tensor<32x16xf16, #[[BSR]]>,
// CHECK-SAME:        %[[B:[^:]+]]: tensor<16x8xf16>) -> tensor<32x8xf16> {
// CHECK-NEXT:      %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK-NEXT:      %[[EMPTY:.*]] = tensor.empty() : tensor<32x8xf16>
// CHECK-NEXT:      %[[C:.*]] = linalg.fill ins(%[[ZERO]] : f16) outs(%[[EMPTY]] : tensor<32x8xf16>) -> tensor<32x8xf16>
// CHECK-NEXT:      %[[RESULT:.*]] = spruce.kernel(%[[K_A:[^ ]+]] = %[[A]] : tensor<32x16xf16, #[[BSR]]>,
// CHECK-SAME:          %[[K_B:[^ ]+]] = %[[B]] : tensor<16x8xf16>,
// CHECK-SAME:          %[[K_C:[^ ]+]] = %[[C]] : tensor<32x8xf16>) {
// CHECK-NEXT:        spruce.yield %[[K_C]] : tensor<32x8xf16>
// CHECK-NEXT:      } -> (tensor<32x8xf16>) {spruce.target = #[[TARGET]]}
// CHECK-NEXT:      return %[[RESULT]] : tensor<32x8xf16>
// CHECK-NEXT:    }
// CHECK-NEXT:  }

func.func @spgemm(%A: tensor<32x16xf16, #BSR>,
                  %B: tensor<16x8xf16>) -> tensor<32x8xf16> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<32x8xf16>
  %C = linalg.fill ins(%zero : f16) outs(%empty : tensor<32x8xf16>) -> tensor<32x8xf16>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i,k,j) -> (i,j)>,
      affine_map<(i,k,j) -> (j,k)>,
      affine_map<(i,k,j) -> (i,k)>
    ],
    iterator_types = [
      "parallel",
      "parallel",
      "reduction"
    ]
  } ins(%A, %B: tensor<32x16xf16, #BSR>, tensor<16x8xf16>)
    outs(%C: tensor<32x8xf16>) {
    ^bb0(%a: f16, %b: f16, %c: f16):
      %product = arith.mulf %a, %b : f16
      %sum = arith.addf %c, %product : f16
      linalg.yield %sum : f16
  } -> tensor<32x8xf16>
  return %result : tensor<32x8xf16>
}
