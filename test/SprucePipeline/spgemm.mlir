// RUN: spruce-opt -spruce-pipeline -verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: func.func @spgemm(
// CHECK-SAME:    %arg0: tensor<64x64xf16>
// CHECK-SAME:    %arg1: tensor<64x64xf16>
// CHECK-SAME:  ) -> tensor<64x64xf16> {
// CHECK:         %cst = arith.constant 0.000000e+00 : f16
// CHECK:         %0 = tensor.empty() : tensor<64x64xf16>
// CHECK:         %1 = linalg.fill ins(%cst : f16) outs(%0 : tensor<64x64xf16>) -> tensor<64x64xf16>
// CHECK:         %2 = spruce.kernel(%arg2 = %arg0 : tensor<64x64xf16>, %arg3 = %arg1 : tensor<64x64xf16>, %arg4 = %1 : tensor<64x64xf16>) {
// CHECK:           spruce.yield %arg4 : tensor<64x64xf16>
// CHECK:         } -> (tensor<64x64xf16>) {spruce.target = #target}
// CHECK:         return %2 : tensor<64x64xf16>
// CHECK:       }

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
