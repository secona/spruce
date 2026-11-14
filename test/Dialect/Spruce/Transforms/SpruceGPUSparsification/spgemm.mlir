// RUN: spruce-opt -split-input-file -spruce-gpu-sparsification -canonicalize -cse %s | FileCheck %s

// CHECK: #[[CSR:.*]] = #sparse_tensor.encoding
#CSR = #sparse_tensor.encoding<{
  map = (i, j) -> (i : dense, j : compressed)
}>

func.func @spgemm(%A: tensor<?x?xf32, #CSR>,
                  %B: tensor<?x?xf32, #CSR>) -> tensor<?x?xf32, #CSR> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %A, %c0 : tensor<?x?xf32, #CSR>
  %n = tensor.dim %B, %c1 : tensor<?x?xf32, #CSR>
  %C = tensor.empty(%m, %n) : tensor<?x?xf32, #CSR>
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
  } ins(%A, %B: tensor<?x?xf32, #CSR>, tensor<?x?xf32, #CSR>)
    outs(%C: tensor<?x?xf32, #CSR>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %product = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %product : f32
      linalg.yield %sum : f32
  } -> tensor<?x?xf32, #CSR>
  return %result : tensor<?x?xf32, #CSR>
}

// -----

// CHECK: #[[BSR:.*]] = #sparse_tensor.encoding
#BSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (
    d0 floordiv 2 : dense,
    d1 floordiv 3 : compressed,
    d0 mod 2 : dense,
    d1 mod 3 : dense
  )
}>

func.func @spgemm(%A: tensor<?x?xf32, #BSR>,
                  %B: tensor<?x?xf32, #BSR>) -> tensor<?x?xf32, #BSR> {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %m = tensor.dim %A, %c0 : tensor<?x?xf32, #BSR>
  %n = tensor.dim %B, %c1 : tensor<?x?xf32, #BSR>
  %C = tensor.empty(%m, %n) : tensor<?x?xf32, #BSR>
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
  } ins(%A, %B: tensor<?x?xf32, #BSR>, tensor<?x?xf32, #BSR>)
    outs(%C: tensor<?x?xf32, #BSR>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %product = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %product : f32
      linalg.yield %sum : f32
  } -> tensor<?x?xf32, #BSR>
  return %result : tensor<?x?xf32, #BSR>
}
