#include "spruce/Spruce.h"
#include <iostream>

constexpr char mlirSrc[] = R"mlir(
func.func @spgemm(
  %a: tensor<16x8xf16>,
  %b: tensor<8x8xf16>
) -> tensor<16x8xf16> {
  %zero = arith.constant 0.0 : f16
  %empty = tensor.empty() : tensor<16x8xf16>
  %init = linalg.fill
    ins(%zero : f16)
    outs(%empty : tensor<16x8xf16>) -> tensor<16x8xf16>
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(m,n,k) -> (m,k)>,
      affine_map<(m,n,k) -> (k,n)>,
      affine_map<(m,n,k) -> (m,n)>
    ],
    iterator_types = [
      "parallel",
      "parallel",
      "reduction"
    ],
    spruce.sparsity = #spruce.sparsity<[
      #spruce.dim_mask<16, [65535]>,
      #spruce.dim_mask<8, [255]>
    ]>
  } ins(%a, %b : tensor<16x8xf16>, tensor<8x8xf16>)
    outs(%init : tensor<16x8xf16>) {
    ^bb0(%lhs: f16, %rhs: f16, %acc: f16):
      %product = arith.mulf %lhs, %rhs : f16
      %sum = arith.addf %acc, %product : f16
      linalg.yield %sum : f16
  } -> tensor<16x8xf16>
  return %result : tensor<16x8xf16>
}
)mlir";

int main() {
  spruce::InputTensor A("A", "data/a_top_ones.mtx");
  A.read();

  spruce::InputTensor B("B", "data/b_ones.mtx");
  B.read();

  spruce::Kernel kernel("ij,jk->ik", mlirSrc, {A, B});

  spruce::Timing t = kernel.compile();
  t.print();

  spruce::Timing::Aggregator aggregator;
  aggregator.iterate(50, [&kernel] { return kernel.run(); });
  aggregator.print();

  auto c = kernel.getResult();

  for (int64_t row = 0; row < c.sizes[0]; ++row) {
    for (int64_t col = 0; col < c.sizes[1]; ++col) {
      auto index = c.offset + row * c.strides[0] + col * c.strides[1];
      std::cout << (col ? " " : "") << static_cast<float>(c.data[index]);
    }
    std::cout << '\n';
  }
}
