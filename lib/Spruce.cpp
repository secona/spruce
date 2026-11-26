#include "spruce/Spruce.h"
#include "spruce/Dialect/Spruce/IR/SpruceDialect.h"
#include "spruce/Pipelines/SprucePipeline.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <stdexcept>

namespace spruce {

namespace {

constexpr char kernelSource[] = R"mlir(
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

} // namespace

Kernel::Kernel(std::string einsum, std::vector<Tensor> ins)
    : einsum(einsum), ins(ins) {
  registry.insert<mlir::spruce::SpruceDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllGPUToLLVMIRTranslations(registry);
}

void Kernel::compile(CompileTimes &times) {
  times = {};
  const auto start = std::chrono::steady_clock::now();

  mlir::MLIRContext context(registry);

  // TODO: Create the kernel source another way.
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kernelSource, &context);
  if (!module)
    throw std::runtime_error("Failed to parse the built-in Spruce kernel");

  // Lower using -spruce-pipeline to LLVM dialecct.
  mlir::PassManager pm(&context);
  mlir::spruce::buildSprucePipeline(pm);
  if (pm.run(*module).failed())
    throw std::runtime_error("Failed to lower the built-in Spruce kernel");

  auto error = llvm::writeToOutput("kernel.mlir", [&](llvm::raw_ostream &out) {
    module->print(out);
    out << '\n';
    return llvm::Error::success();
  });

  if (error)
    throw std::runtime_error(llvm::toString(std::move(error)));

  times.total = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - start);
}

void Kernel::run(RunTimes &times) {
  times = {};
  const auto start = std::chrono::steady_clock::now();

  // TODO: Run the kernel.

  times.total = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now() - start);
}

} // namespace spruce
