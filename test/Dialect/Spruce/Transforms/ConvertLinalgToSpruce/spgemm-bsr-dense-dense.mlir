// RUN: spruce-opt -split-input-file -convert-linalg-to-spruce -spruce-tile-mma -canonicalize -cse -verify-roundtrip %s | FileCheck %s --implicit-check-not=tensor.pad --implicit-check-not=linalg.generic

// M=32, N=8, K=16 and 16x8 storage blocks produce exact 16x8x8 MMA tiles.

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

// CHECK-LABEL: func.func @spgemm(
// CHECK-SAME:      %[[A:[^:]+]]: tensor<32x16xf32, #[[BSR]]>,
// CHECK-SAME:      %[[B:[^:]+]]: tensor<16x8xf32>) -> tensor<32x8xf32>
// CHECK-DAG:     %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32

// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<32x8xf32>
// CHECK:         %[[C:.*]] = linalg.fill ins(%[[ZERO]] : f32) outs(%[[EMPTY]] : tensor<32x8xf32>) -> tensor<32x8xf32>
// CHECK:         %[[N_BLOCKS:.*]] = spruce.num_blocks %[[A]] : tensor<32x16xf32, #[[BSR]]>
// CHECK:         %[[RESULT:.*]] = spruce.kernel(%[[K_A:[^ ]+]] = %[[A]] : tensor<32x16xf32, #[[BSR]]>,
// CHECK-SAME:        %[[K_B:[^ ]+]] = %[[B]] : tensor<16x8xf32>,
// CHECK-SAME:        %[[K_C:[^ ]+]] = %[[C]] : tensor<32x8xf32>,
// CHECK-SAME:        %[[K_N_BLOCKS:[^ ]+]] = %[[N_BLOCKS]] : index) {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[C8:.*]] = arith.constant 8 : index
// CHECK-DAG:       %[[C16:.*]] = arith.constant 16 : index
// CHECK:           %[[LOOP_RESULT:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[K_N_BLOCKS]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[K_C]]) -> (tensor<32x8xf32>) {
// CHECK:             %[[ROW_IDX:.*]], %[[COL_IDX:.*]] = spruce.block_coordinates %[[K_A]][%[[I]]] : tensor<32x16xf32, #[[BSR]]>
// CHECK:             %[[ROW_OFFSET:.*]] = arith.muli %[[ROW_IDX]], %[[C16]] : index
// CHECK:             %[[ACC_SLICE:.*]] = tensor.extract_slice %[[ACC]][%[[ROW_OFFSET]], 0] [16, 8] [1, 1] : tensor<32x8xf32> to tensor<16x8xf32>
// CHECK:             %[[A_BLOCK:.*]] = spruce.get_block %[[K_A]][%[[I]]] : tensor<32x16xf32, #[[BSR]]> -> tensor<16x8xf32>
// CHECK:             %[[COL_OFFSET:.*]] = arith.muli %[[COL_IDX]], %[[C8]] : index
// CHECK:             %[[B_SLICE:.*]] = tensor.extract_slice %[[K_B]][%[[COL_OFFSET]], 0] [8, 8] [1, 1] : tensor<16x8xf32> to tensor<8x8xf32>
// CHECK:             %[[MMA_RESULT:.*]] = spruce.mma %[[A_BLOCK]], %[[B_SLICE]], %[[ACC_SLICE]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
// CHECK:             %[[UPDATED:.*]] = tensor.insert_slice %[[MMA_RESULT]] into %[[ACC]][%[[ROW_OFFSET]], 0] [16, 8] [1, 1] : tensor<16x8xf32> into tensor<32x8xf32>
// CHECK:             scf.yield %[[UPDATED]] : tensor<32x8xf32>
// CHECK:           }
// CHECK:           spruce.yield %[[LOOP_RESULT]] : tensor<32x8xf32>
// CHECK:         } -> (tensor<32x8xf32>) {spruce.target = #[[TARGET]]}
// CHECK:         return %[[RESULT]] : tensor<32x8xf32>

func.func @spgemm(%A: tensor<32x16xf32, #BSR>,
                  %B: tensor<16x8xf32>) -> tensor<32x8xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %zero = arith.constant 0.0 : f32
  %empty = tensor.empty() : tensor<32x8xf32>
  %C = linalg.fill ins(%zero : f32) outs(%empty : tensor<32x8xf32>) -> tensor<32x8xf32>
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
  } ins(%A, %B: tensor<32x16xf32, #BSR>, tensor<16x8xf32>)
    outs(%C: tensor<32x8xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %product = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %product : f32
      linalg.yield %sum : f32
  } -> tensor<32x8xf32>
  return %result : tensor<32x8xf32>
}
