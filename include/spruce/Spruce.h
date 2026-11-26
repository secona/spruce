#ifndef SPRUCE_SPRUCE_H
#define SPRUCE_SPRUCE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include <chrono>
#include <memory>
#include <optional>
#include <ratio>
#include <string>
#include <vector>

class Tensor; // Adlet tensor

namespace spruce {

struct InputTensor {
  std::string name;
  std::string path;
  std::shared_ptr<::Tensor> adletTensor;

  InputTensor(std::string name, std::string path) : name(name), path(path) {}

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
};

class Kernel {
  std::string einsum;
  std::vector<InputTensor> ins;

  mlir::MLIRContext mlirContext;
  mlir::OwningOpRef<mlir::ModuleOp> mlirModule;

  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule;

public:
  Kernel(std::string einsum, std::vector<InputTensor> ins);

  Timing compile();

  Timing run();
};

} // namespace spruce

#endif // SPRUCE_SPRUCE_H
