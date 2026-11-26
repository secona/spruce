#ifndef SPRUCE_SPRUCE_H
#define SPRUCE_SPRUCE_H

#include "mlir/IR/DialectRegistry.h"
#include <chrono>
#include <memory>
#include <optional>
#include <string>
#include <vector>

class Tensor; // Adlet tensor

namespace spruce {

struct Tensor {
  std::string name;
  std::string path;
  std::shared_ptr<::Tensor> adletTensor;

  Tensor(std::string name, std::string path) : name(name), path(path) {}

  void read();
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
