#ifndef SPRUCE_SPRUCE_H
#define SPRUCE_SPRUCE_H

#include "mlir/IR/DialectRegistry.h"
#include <chrono>
#include <optional>
#include <string>
#include <vector>

namespace spruce {

struct Tensor {
  std::string name;
  std::string path;

  Tensor(std::string name, std::string path) : name(name), path(path) {}
};

struct CompileTimes {
  std::optional<std::chrono::nanoseconds> spaAnalysis;
  std::optional<std::chrono::nanoseconds> mlirLowering;
  std::optional<std::chrono::nanoseconds> llvmCompilation;
  std::chrono::nanoseconds total{};
};

struct RunTimes {
  std::chrono::nanoseconds total{};
};

class Kernel {
  std::string einsum;
  std::vector<Tensor> ins;
  mlir::DialectRegistry registry;

public:
  Kernel(std::string einsum, std::vector<Tensor> ins);

  void compile(CompileTimes &times);

  void run(RunTimes &times);
};

} // namespace spruce

#endif // SPRUCE_SPRUCE_H
