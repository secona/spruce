// RUN: spruce-opt -spruce-pipeline -verify-roundtrip %s | FileCheck %s

// CHECK-LABEL: llvm.func @spgemm

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
