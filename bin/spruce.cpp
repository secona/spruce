#include "adlet/graph.hpp"
#include "taco/tensor.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

constexpr std::int64_t chunkSize = 64;

namespace {

using namespace llvm;

struct InputTensor {
  std::string name;
  std::string path;
};

std::shared_ptr<Tensor> loadTensor(const InputTensor &t) {
  taco::Tensor<double> matrix(taco::read(t.path, taco::Sparse));
  const std::vector<int> &shape = matrix.getDimensions();

  std::vector<SparsityVector> sparsities(2);
  for (const auto &[crd, v] : matrix) {
    if (v == 0.0)
      continue;
    sparsities[0].set(crd[0]);
    sparsities[1].set(crd[1]);
  }

  return std::make_shared<Tensor>(shape, sparsities, t.name);
}

json::Array chunkBitsToWords(const SparsityVector &mask, int extent) {
  json::Array out;

  const size_t numChunks = (extent + chunkSize - 1) / chunkSize;
  for (size_t c = 0; c < numChunks; c++) {
    const size_t offset = c * chunkSize;
    const size_t numBits = std::min<size_t>(chunkSize, extent - offset);

    uint64_t bits = 0;
    for (size_t i = 0; i < numBits; i++) {
      if (mask.test(offset + i))
        bits |= uint64_t{1} << i;
    }

    out.push_back(bits);
  }

  return out;
}

} // namespace

namespace opt {

cl::opt<std::string> einsum(cl::Positional, cl::Required);
cl::list<std::string> inputs(cl::Positional, cl::OneOrMore);
cl::opt<bool> sparsityInformation("sparsity-information");

} // namespace opt

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv);

  std::vector<InputTensor> bindings;
  for (StringRef input : opt::inputs) {
    auto [name, path] = input.split('=');
    if (name.empty() || path.empty())
      throw std::invalid_argument("input tensor");
    bindings.push_back({name.str(), path.str()});
  }

  std::vector<std::shared_ptr<Tensor>> inputs(bindings.size());
  std::transform(bindings.begin(), bindings.end(), inputs.begin(), loadTensor);

  // TODO: Implement more than just this matrix size.
  auto rows = inputs[0]->sizes[0];
  auto cols = inputs[1]->sizes[1];
  std::shared_ptr<Tensor> out =
      std::make_shared<Tensor>(std::vector<int>{rows, cols}, "out");

  auto einsum = std::make_shared<Einsum>(inputs, out, opt::einsum.getValue());

  auto graph = Graph::build_graph(inputs, out, {einsum});
  graph.run_propagation();

  if (opt::sparsityInformation) {
    json::Value result(json::Array{
        json::Object{{"size", rows},
                     {"words", chunkBitsToWords(out->sparsities[0], rows)}},
        json::Object{{"size", cols},
                     {"words", chunkBitsToWords(out->sparsities[1], cols)}},
    });
    llvm::outs() << result << '\n';
  }

  return 0;
}
