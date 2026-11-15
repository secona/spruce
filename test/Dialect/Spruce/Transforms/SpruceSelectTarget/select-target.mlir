// RUN: spruce-opt -spruce-select-target -verify-roundtrip %s | FileCheck %s
// RUN: spruce-opt -spruce-select-target -spruce-select-target -verify-roundtrip %s | FileCheck %s

// CHECK-DAG: #[[TARGET:.*]] = #spruce.target<{m = 16, n = 8, k = 8}>
// CHECK-DAG: #[[BSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 floordiv 2 : dense, d1 floordiv 3 : compressed, d0 mod 2 : dense, d1 mod 3 : dense) }>
#BSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (
    d0 floordiv 2 : dense,
    d1 floordiv 3 : compressed,
    d0 mod 2 : dense,
    d1 mod 3 : dense
  )
}>

// CHECK-LABEL: func.func @spgemm(
// CHECK-SAME:      %[[A:[^:]+]]: tensor<12x12xf32, #[[BSR]]>,
// CHECK-SAME:      %[[B:[^:]+]]: tensor<12x12xf32>,
// CHECK-SAME:      %[[C:[^:]+]]: tensor<12x12xf32>) -> tensor<12x12xf32> {
// CHECK:         %[[RESULT:.*]] = linalg.generic
// CHECK-SAME:        ins(%[[A]], %[[B]] : tensor<12x12xf32, #[[BSR]]>, tensor<12x12xf32>)
// CHECK-SAME:        outs(%[[C]] : tensor<12x12xf32>)
// CHECK-SAME:        attrs = {spruce.target = #[[TARGET]]}
// CHECK:         return %[[RESULT]] : tensor<12x12xf32>

func.func @spgemm(%A: tensor<12x12xf32, #BSR>,
                  %B: tensor<12x12xf32>,
                  %C: tensor<12x12xf32>) -> tensor<12x12xf32> {
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(m, n, k) -> (m, k)>,
      affine_map<(m, n, k) -> (k, n)>,
      affine_map<(m, n, k) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B : tensor<12x12xf32, #BSR>, tensor<12x12xf32>)
    outs(%C : tensor<12x12xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %product = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %product : f32
      linalg.yield %sum : f32
  } -> tensor<12x12xf32>
  return %result : tensor<12x12xf32>
}

// CHECK-LABEL: func.func @elementwise_add(
// CHECK-NOT: spruce.target
// CHECK:         return

func.func @elementwise_add(%A: tensor<12x12xf32>,
                           %B: tensor<12x12xf32>,
                           %C: tensor<12x12xf32>) -> tensor<12x12xf32> {
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(m, n) -> (m, n)>,
      affine_map<(m, n) -> (m, n)>,
      affine_map<(m, n) -> (m, n)>
    ],
    iterator_types = ["parallel", "parallel"]
  } ins(%A, %B : tensor<12x12xf32>, tensor<12x12xf32>)
    outs(%C : tensor<12x12xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %sum = arith.addf %a, %b : f32
      linalg.yield %sum : f32
  } -> tensor<12x12xf32>
  return %result : tensor<12x12xf32>
}
