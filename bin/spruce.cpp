#include "adlet/graph.hpp"
#include "muopt/muopt.hpp"
#include "taco/tensor.h"
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

  static InputTensor fromString(const std::string &s) {
    auto pos = s.find("=");
    if (pos == std::string::npos)
      throw std::invalid_argument(s);
    return {s.substr(0, pos), s.substr(pos + 1)};
  }
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

struct SpruceOptions {
  std::string einsum;
  std::vector<InputTensor> inputs;
  bool sparsityInformation;

  static SpruceOptions parse(int argc, char **argv) {
    if (argc <= 2)
      throw std::invalid_argument("not enough argument");

    std::string einsum;
    std::vector<InputTensor> inputs;
    bool sparsityInformation = false;

    muopt::Parser parser(argc, argv);
    while (muopt::Arg arg = parser.next()) {
      if (arg.is_plain()) {
        if (einsum.empty()) {
          einsum = arg.as_str();
          continue;
        }

        std::string input(arg.as_str());
        inputs.push_back(InputTensor::fromString(input));

        continue;
      }

      if (arg.is_long("sparsity-information")) {
        sparsityInformation = true;
        continue;
      }

      throw std::invalid_argument(std::string(arg.as_str()));
    }

    return {einsum, inputs, sparsityInformation};
  }
};

} // namespace

int main(int argc, char **argv) {
  auto opt = SpruceOptions::parse(argc, argv);

  std::vector<std::shared_ptr<Tensor>> inputs(opt.inputs.size());
  std::transform(opt.inputs.begin(), opt.inputs.end(), inputs.begin(),
                 loadTensor);

  // TODO: Implement more than just this matrix size.
  auto rows = inputs[0]->sizes[0];
  auto cols = inputs[1]->sizes[1];
  std::shared_ptr<Tensor> out =
      std::make_shared<Tensor>(std::vector<int>{rows, cols}, "out");

  auto einsum = std::make_shared<Einsum>(inputs, out, opt.einsum);

  auto graph = Graph::build_graph(inputs, out, {einsum});
  graph.run_propagation();

  if (opt.sparsityInformation) {
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
