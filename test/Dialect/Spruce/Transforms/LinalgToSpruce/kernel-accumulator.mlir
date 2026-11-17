// RUN: spruce-opt -convert-linalg-to-spruce -verify-roundtrip %s | FileCheck %s --implicit-check-not=linalg.generic --implicit-check-not=linalg.fill --implicit-check-not=tensor.empty

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

// CHECK-LABEL: func.func @preserve_accumulator(
// CHECK-SAME:      %[[A:[^:]+]]: tensor<32x16xf32, #[[BSR]]>,
// CHECK-SAME:      %[[B:[^:]+]]: tensor<16x8xf32>,
// CHECK-SAME:      %[[C:[^:]+]]: tensor<32x8xf32>) -> tensor<32x8xf32> {
// CHECK:         %[[N_BLOCKS:.*]] = spruce.num_blocks %[[A]] : tensor<32x16xf32, #[[BSR]]>
// CHECK:         %[[RESULT:.*]] = spruce.kernel(%[[A]], %[[B]], %[[C]], %[[N_BLOCKS]])
// CHECK-SAME:        : (tensor<32x16xf32, #[[BSR]]>, tensor<16x8xf32>, tensor<32x8xf32>, index) {
// CHECK:         ^bb0(%[[K_A:[^:]+]]: tensor<32x16xf32, #[[BSR]]>,
// CHECK-SAME:          %[[K_B:[^:]+]]: tensor<16x8xf32>,
// CHECK-SAME:          %[[K_C:[^:]+]]: tensor<32x8xf32>,
// CHECK-SAME:          %[[K_N_BLOCKS:[^:]+]]: index):
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:           %[[LOOP_RESULT:.*]] = scf.for %[[P:.*]] = %[[C0]] to %[[K_N_BLOCKS]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[K_C]]) -> (tensor<32x8xf32>) {
// CHECK:             %[[ROW:.*]], %[[COL:.*]] = spruce.block_coordinates %[[K_A]][%[[P]]] : tensor<32x16xf32, #[[BSR]]>
// CHECK:             %[[ACC_TILE:.*]] = tensor.extract_slice %[[ACC]]
// CHECK:             %[[A_BLOCK:.*]] = spruce.get_block %[[K_A]][%[[P]]] : tensor<32x16xf32, #[[BSR]]> -> tensor<16x8xf32>
// CHECK:             %[[B_TILE:.*]] = tensor.extract_slice %[[K_B]]
// CHECK:             %[[PRODUCT:.*]] = spruce.mma %[[A_BLOCK]], %[[B_TILE]], %[[ACC_TILE]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
// CHECK:             %[[UPDATED:.*]] = tensor.insert_slice %[[PRODUCT]] into %[[ACC]]
// CHECK:             scf.yield %[[UPDATED]] : tensor<32x8xf32>
// CHECK:           }
// CHECK:           spruce.yield %[[LOOP_RESULT]] : tensor<32x8xf32>
// CHECK:         } -> (tensor<32x8xf32>) {spruce.target = #[[TARGET]]}
// CHECK:         return %[[RESULT]] : tensor<32x8xf32>

func.func @preserve_accumulator(%A: tensor<32x16xf32, #BSR>,
                               %B: tensor<16x8xf32>,
                               %C: tensor<32x8xf32>) -> tensor<32x8xf32> {
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
