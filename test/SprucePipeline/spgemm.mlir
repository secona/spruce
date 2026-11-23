// RUN: spruce-opt -spruce-pipeline -verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: func.func @spgemm(
// CHECK-SAME:    %[[A:[^:]+]]: tensor<64x64xf16>, %[[B:[^:]+]]: tensor<64x64xf16>) -> tensor<64x64xf16>
// CHECK:         %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<64x64xf16>
// CHECK:         %[[C:.*]] = linalg.fill ins(%[[ZERO]] : f16) outs(%[[EMPTY]] : tensor<64x64xf16>)
// CHECK:         %[[RESULT:.*]] = spruce.kernel(%[[A_D:[^ ]+]] = %[[A]] : tensor<64x64xf16>, %[[B_D:[^ ]+]] = %[[B]] : tensor<64x64xf16>, %[[C_D:[^ ]+]] = %[[C]] : tensor<64x64xf16>) {
// CHECK-DAG:       %[[START:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[END:.*]] = arith.constant 64 : index
// CHECK-DAG:       %[[STEP:.*]] = arith.constant 8 : index
// CHECK:           %[[INIT:.*]] = tensor.extract_slice %[[C_D]][0, 0] [16, 8] [1, 1]
// CHECK:           %[[TILE:.*]] = scf.for %[[K:[^ ]+]] = %[[START]] to %[[END]] step %[[STEP]] iter_args(%[[ACC:[^ ]+]] = %[[INIT]]) -> (tensor<16x8xf16>) {
// CHECK:             %[[LHS:.*]] = tensor.extract_slice %[[A_D]][0, %[[K]]] [16, 8] [1, 1]
// CHECK:             %[[RHS:.*]] = tensor.extract_slice %[[B_D]][%[[K]], 0] [8, 8] [1, 1]
// CHECK:             %[[AF:.*]] = spruce_gpu.mma.load_matrix %[[LHS]] -> %{{.*}}
// CHECK:             %[[BF:.*]] = spruce_gpu.mma.load_matrix %[[RHS]] -> %{{.*}}
// CHECK:             %[[CF:.*]] = spruce_gpu.mma.load_matrix %[[ACC]] -> %{{.*}}
// CHECK:             %[[DF:.*]] = spruce_gpu.mma %[[AF]], %[[BF]], %[[CF]] -> %{{[^,]+}}
// CHECK:             %[[SUM:.*]] = spruce_gpu.mma.store_matrix %[[DF]] -> %[[ACC]]
// CHECK:             scf.yield %[[SUM]] : tensor<16x8xf16>
// CHECK:           %[[OUT:.*]] = tensor.insert_slice %[[TILE]] into %[[C_D]][0, 0] [16, 8] [1, 1]
// CHECK:           spruce.yield %[[OUT]] : tensor<64x64xf16>
// CHECK:         } -> (tensor<64x64xf16>) {spruce.target = #target}
// CHECK:         return %[[RESULT]] : tensor<64x64xf16>

func.func @spgemm(
  %A: tensor<64x64xf16>,
  %B: tensor<64x64xf16>
) -> tensor<64x64xf16> {
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<64x64xf16>
  %C = linalg.fill ins(%zero : f16) outs(%empty : tensor<64x64xf16>) -> tensor<64x64xf16>
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
    ],
    spruce.sparsity = #spruce.sparsity<[
      #spruce.dim_mask<64, [65535]>,
      #spruce.dim_mask<64, [255]>
    ]>
  } ins(%A, %B: tensor<64x64xf16>, tensor<64x64xf16>)
    outs(%C: tensor<64x64xf16>) {
    ^bb0(%a: f16, %b: f16, %c: f16):
      %product = arith.mulf %a, %b : f16
      %sum = arith.addf %c, %product : f16
      linalg.yield %sum : f16
  } -> tensor<64x64xf16>
  return %result : tensor<64x64xf16>
}
