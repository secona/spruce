#ifndef SPRUCE_SPRUCE_H
#define SPRUCE_SPRUCE_H

#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringMap.h"
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <vector>

class Tensor; // Adlet tensor

namespace mlir {
class ExecutionEngine;
} // namespace mlir

namespace spruce {

struct InputTensor {
  std::string name;
  std::string path;
  std::shared_ptr<::Tensor> adletTensor;

  InputTensor(std::string name, std::string path) : name(name), path(path) {}

  void read();
};

struct InputMLIR {
  std::string path;
  std::string src;

  InputMLIR(std::string path) : path(path) {}

  void read();
};

class Timing {
  using Clock = std::chrono::steady_clock;

  struct Entry {
    std::string label;
    double time;
  };

  Clock::time_point start;
  std::vector<Entry> entries;

public:
  Timing() : start(Clock::now()) {}
  Timing(size_t n) : start(Clock::now()) { entries.reserve(n); }

  void tick(const std::string &label);

  void print();

  class Aggregator {
    llvm::StringMap<std::vector<double>> samples;

  public:
    void iterate(int n, const std::function<Timing()> &f);

    void print();

    void saveToCSV(const std::string &path);
  };
};

class Kernel {
  std::string einsum;
  InputMLIR mlirSrc;
  std::vector<InputTensor> ins;

  mlir::MLIRContext mlirContext;

  std::unique_ptr<mlir::ExecutionEngine> engine;
  std::function<void(void **)> kernelFn;

public:
  Kernel(std::string einsum, InputMLIR mlirSrc, std::vector<InputTensor> ins);
  ~Kernel();

  Timing compile();
  Timing run();
};

} // namespace spruce

#endif // SPRUCE_SPRUCE_H
