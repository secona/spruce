// RUN: spruce-opt -split-input-file -spruce-gpu-sparsification -canonicalize -cse %s | FileCheck %s

// CHECK: #[[BSR:.*]] = #sparse_tensor.encoding
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
// CHECK-SAME:      %[[B:[^:]+]]: tensor<12x12xf32>) -> tensor<12x12xf32>
// CHECK-DAG:     %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK-DAG:     %[[C3:.*]] = arith.constant 3 : index

// CHECK:         %[[C:.*]] = tensor.empty() : tensor<12x12xf32>
// CHECK:         %[[N_BLOCKS:.*]] = spruce.num_blocks %[[A]] : tensor<12x12xf32, #sparse>
// CHECK:         %[[RESULT:.*]] = scf.for %[[I:.*]] = %[[C0]] to %[[N_BLOCKS]] step %[[C1]] iter_args(%[[ACC:.*]] = %[[C]]) -> (tensor<12x12xf32>) {
// CHECK:           %[[ROW_IDX:.*]], %[[COL_IDX:.*]] = spruce.block_coordinates %[[A]][%[[I]]] : tensor<12x12xf32, #sparse>
// CHECK:           %[[ROW_IDX_MUL_3:.*]] = arith.muli %[[ROW_IDX]], %[[C2]] : index
// CHECK:           %[[ACC_SLICE:.*]] = tensor.extract_slice %[[ACC]][%[[ROW_IDX_MUL_3]], 0] [2, 12] [1, 1] : tensor<12x12xf32> to tensor<2x12xf32>
// CHECK:           %[[A_BLOCK:.*]] = spruce.get_block %[[A]][%[[I]]] : tensor<12x12xf32, #sparse> -> tensor<2x3xf32>
// CHECK:           %[[COL_IDX_MUL_3:.*]] = arith.muli %[[COL_IDX:.*]], %[[C3]] : index
// CHECK:           %[[B_SLICE:.*]] = tensor.extract_slice %[[B]][%[[COL_IDX_MUL_3]], 0] [3, 12] [1, 1] : tensor<12x12xf32> to tensor<3x12xf32>
// CHECK:           %[[MMA_RESULT:.*]] = spruce.mma %[[A_BLOCK]], %[[B_SLICE]], %[[ACC_SLICE]] : tensor<2x3xf32>, tensor<3x12xf32>, tensor<2x12xf32> -> tensor<2x12xf32>
// CHECK:           %[[UPDATED:.*]] = tensor.insert_slice %[[MMA_RESULT]] into %[[ACC]][%[[ROW_IDX_MUL_3]], 0] [2, 12] [1, 1] : tensor<2x12xf32> into tensor<12x12xf32>
// CHECK:           scf.yield %[[UPDATED]] : tensor<12x12xf32>
// CHECK:         }
// CHECK:         return %[[RESULT]] : tensor<12x12xf32>

func.func @spgemm(%A: tensor<12x12xf32, #BSR>,
                  %B: tensor<12x12xf32>) -> tensor<12x12xf32> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %C = tensor.empty() : tensor<12x12xf32>
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
  } ins(%A, %B: tensor<12x12xf32, #BSR>, tensor<12x12xf32>)
    outs(%C: tensor<12x12xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %product = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %product : f32
      linalg.yield %sum : f32
  } -> tensor<12x12xf32>
  return %result : tensor<12x12xf32>
}
