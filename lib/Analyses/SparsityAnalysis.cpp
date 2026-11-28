#include "spruce/Analyses/SparsityAnalysis.h"

#include <memory>

namespace spruce {

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
}

} // namespace spruce
