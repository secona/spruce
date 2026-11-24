// RUN: spruce-opt -spruce-pipeline -verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: func.func @spgemm(
// CHECK-SAME:    %[[A:[^:]+]]: tensor<64x64xf16>, %[[B:[^:]+]]: tensor<64x64xf16>) -> tensor<64x64xf16>
// CHECK-DAG:     %[[START:.*]] = arith.constant 0 : index
// CHECK-DAG:     %[[END:.*]] = arith.constant 64 : index
// CHECK-DAG:     %[[STEP:.*]] = arith.constant 8 : index
// CHECK:         %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<64x64xf16>
// CHECK:         %[[C:.*]] = linalg.fill ins(%[[ZERO]] : f16) outs(%[[EMPTY]] : tensor<64x64xf16>)
// CHECK:         %[[INIT:.*]] = tensor.extract_slice %[[C]][0, 0] [16, 8] [1, 1]
// CHECK:         %[[TILE:.*]] = scf.for %[[K:[^ ]+]] = %[[START]] to %[[END]] step %[[STEP]] iter_args(%[[ACC:[^ ]+]] = %[[INIT]]) -> (tensor<16x8xf16>) {
// CHECK-NEXT:      %[[LHS:.*]] = tensor.extract_slice %[[A]][0, %[[K]]] [16, 8] [1, 1]
// CHECK-NEXT:      %[[RHS:.*]] = tensor.extract_slice %[[B]][%[[K]], 0] [8, 8] [1, 1]
// CHECK-NEXT:      %[[SUM:.*]] = spruce.kernel(%[[LHS_D:[^ ]+]] = %[[LHS]] : tensor<16x8xf16>, %[[RHS_D:[^ ]+]] = %[[RHS]] : tensor<8x8xf16>, %[[ACC_D:[^ ]+]] = %[[ACC]] : tensor<16x8xf16>) {
// CHECK-NEXT:        %[[A_FRAG:.*]] = spruce_gpu.create_fragment
// CHECK-NEXT:        %[[B_FRAG:.*]] = spruce_gpu.create_fragment
// CHECK-NEXT:        %[[C_FRAG:.*]] = spruce_gpu.create_fragment
// CHECK-NEXT:        %[[D_FRAG:.*]] = spruce_gpu.create_fragment
// CHECK-NEXT:        %[[AF:.*]] = spruce_gpu.mma.load_matrix %[[LHS_D]] -> %[[A_FRAG]]
// CHECK-NEXT:        %[[BF:.*]] = spruce_gpu.mma.load_matrix %[[RHS_D]] -> %[[B_FRAG]]
// CHECK-NEXT:        %[[CF:.*]] = spruce_gpu.mma.load_matrix %[[ACC_D]] -> %[[C_FRAG]]
// CHECK-NEXT:        %[[DF:.*]] = spruce_gpu.mma %[[AF]], %[[BF]], %[[CF]] -> %[[D_FRAG]]
// CHECK-NEXT:        %[[SUM_D:.*]] = spruce_gpu.mma.store_matrix %[[DF]] -> %[[ACC_D]]
// CHECK-NEXT:        spruce.yield %[[SUM_D]] : tensor<16x8xf16>
// CHECK-NEXT:      } -> (tensor<16x8xf16>) {spruce.target = #target}
// CHECK-NEXT:      scf.yield %[[SUM]] : tensor<16x8xf16>
// CHECK-NEXT:    }
// CHECK-NEXT:    %[[OUT:.*]] = tensor.insert_slice %[[TILE]] into %[[C]][0, 0] [16, 8] [1, 1]
// CHECK-NEXT:    return %[[OUT]] : tensor<64x64xf16>

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
