// RUN: spruce-opt -convert-linalg-to-spruce -verify-diagnostics %s

func.func @unsupported_dense_matmul(%lhs: tensor<16x8xf32>,
                                   %rhs: tensor<8x8xf32>,
                                   %acc: tensor<16x8xf32>) -> tensor<16x8xf32> {
  // expected-error@+1 {{failed to legalize operation 'linalg.generic' that was explicitly marked illegal}}
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i,j,k) -> (i,k)>,
      affine_map<(i,j,k) -> (k,j)>,
      affine_map<(i,j,k) -> (i,j)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%lhs, %rhs: tensor<16x8xf32>, tensor<8x8xf32>)
    outs(%acc: tensor<16x8xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %product = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %product : f32
      linalg.yield %sum : f32
  } -> tensor<16x8xf32>
  return %result : tensor<16x8xf32>
}
