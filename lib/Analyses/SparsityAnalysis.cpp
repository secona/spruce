#include "spruce/Analyses/SparsityAnalysis.h"

#include <bitset>
#include <cstddef>
#include <iostream>
#include <memory>

namespace spruce {

namespace {

std::bitset<256> chunk(std::bitset<2048> original, int rows, int cols) {
  std::bitset<256> result;

  for (int i = 0; i < (rows * cols + 7) / 8; ++i) {
    const std::size_t offset = i * 8;
    bool set = false;
    for (std::size_t j = 0; j < 8; ++j) {
      if (original[offset + j]) {
        set = true;
        break;
      }
    }
    result.set(i, set);
  }

  return result;
}

} // namespace

void runSparsityAnalysis(std::vector<adlet::TensorPtr> ins) {
  // int rows = 2;
  // int cols = 2;

  // std::vector<adlet::TensorPtr> ins;

  int rows = ins[0]->sizes[0];
  int cols = ins[1]->sizes[1];

  auto out =
      std::make_shared<adlet::Tensor>(std::vector<int>{rows, cols}, "out");

  auto einsum = std::make_shared<adlet::Einsum>(ins, out, "ij,jk->ik");

  auto graph = adlet::Graph::build_graph(ins, out, {einsum});
  graph.run_propagation();

  for (int i = 0; i < rows; ++i)
    std::cout << out->sparsities[0][i];
  std::cout << '\n';
  for (int i = 0; i < cols; ++i)
    std::cout << out->sparsities[1][i];
  std::cout << '\n';

  std::vector<std::bitset<256>> chunked{
    chunk(out->sparsities[0], rows, cols),
    chunk(out->sparsities[1], rows, cols),
  };

  for (int i = 0; i < (rows + 7) / 8; ++i)
    std::cout << chunked[0][i];
  std::cout << '\n';
  for (int i = 0; i < (cols + 7) / 8; ++i)
    std::cout << chunked[1][i];
  std::cout << '\n';
}

} // namespace spruce
