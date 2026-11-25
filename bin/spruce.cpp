#include "muopt/muopt.hpp"
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct SpruceOptions {
  struct InputTensor {
    std::string name;
    std::string path;
  };

  std::string einsum;
  std::vector<InputTensor> inputs;

  static SpruceOptions parse(int argc, char **argv) {
    if (argc <= 2)
      throw std::invalid_argument("not enough argument");

    std::string einsum;
    std::vector<InputTensor> inputs;

    muopt::Parser parser(argc, argv);
    while (muopt::Arg arg = parser.next()) {
      if (einsum.empty()) {
        einsum = arg.as_str();
        continue;
      }

      std::string input(arg.as_str());
      auto pos = input.find("=");
      if (pos == std::string::npos)
        throw std::invalid_argument("invalid input tensor format");
      inputs.push_back({input.substr(0, pos), input.substr(pos + 1)});
    }

    return {einsum, inputs};
  }
};

} // namespace

int main(int argc, char **argv) {
  auto opt = SpruceOptions::parse(argc, argv);

  return 0;
}
