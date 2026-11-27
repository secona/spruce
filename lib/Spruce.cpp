#include "spruce/Spruce.h"
#include "spruce/Dialect/Spruce/IR/SpruceDialect.h"
#include "spruce/Pipelines/SprucePipeline.h"

#include "taco/tensor.h"

#include "adlet/graph.hpp"

#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/TargetSelect.h"

#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <utility>

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

func.func @smoke() {
  %one = arith.constant 1 : index
  gpu.launch blocks(%bx, %by, %bz) in (%gx = %one, %gy = %one, %gz = %one)
             threads(%tx, %ty, %tz) in (%sx = %one, %sy = %one, %sz = %one) {
    gpu.printf "test\0A"
    gpu.terminator
  }
  return
}
)mlir";

} // namespace

// -----------------------------------------------------------------------------
// InputTensor
// -----------------------------------------------------------------------------

void InputTensor::read() {
  taco::Tensor<double> matrix(taco::read(path, taco::Sparse));
  const std::vector<int> &shape = matrix.getDimensions();

  std::vector<SparsityVector> sparsities(2);
  for (const auto &[crd, v] : matrix) {
    if (v == 0.0)
      continue;
    sparsities[0].set(crd[0]);
    sparsities[1].set(crd[1]);
  }

  adletTensor = std::make_shared<::Tensor>(shape, sparsities, name);
}

// -----------------------------------------------------------------------------
// Kernel
// -----------------------------------------------------------------------------

Kernel::Kernel(std::string einsum, std::vector<InputTensor> ins)
    : einsum(einsum), ins(ins), mlirContext([]() {
        mlir::DialectRegistry registry;
        registry.insert<mlir::spruce::SpruceDialect>();
        mlir::registerAllDialects(registry);
        mlir::registerAllExtensions(registry);
        mlir::registerAllGPUToLLVMIRTranslations(registry);
        mlir::registerAllToLLVMIRTranslations(registry);
        mlir::registerBuiltinDialectTranslation(registry);
        return mlir::MLIRContext(registry);
      }()) {}

Kernel::~Kernel() = default;

Timing Kernel::compileAndRun() {
  Timing t(5);

  // ---------------------------------------------------------------------------
  // SPA Analysis
  // ---------------------------------------------------------------------------

  std::vector<std::shared_ptr<::Tensor>> inputs(ins.size());
  std::transform(ins.begin(), ins.end(), inputs.begin(),
                 [](const InputTensor &t) { return t.adletTensor; });

  auto rows = inputs[0]->sizes[0];
  auto cols = inputs[1]->sizes[1];
  std::shared_ptr<::Tensor> out =
      std::make_shared<::Tensor>(std::vector<int>{rows, cols}, "out");

  auto op = std::make_shared<::Einsum>(inputs, out, einsum);

  auto graph = ::Graph::build_graph(inputs, out, {op});
  graph.run_propagation();

  t.tick("spa");

  // ---------------------------------------------------------------------------
  // Lowering MLIR
  // ---------------------------------------------------------------------------

  // TODO: Create the kernel source another way.
  auto module = mlir::parseSourceString<mlir::ModuleOp>(kernelSource, &mlirContext);
  if (!module)
    throw std::runtime_error("Failed to parse the built-in Spruce kernel");

  // Lower using -spruce-pipeline to LLVM dialecct.
  mlir::PassManager pm(&mlirContext);
  mlir::spruce::buildSprucePipeline(pm);
  if (pm.run(*module).failed())
    throw std::runtime_error("Failed to lower the built-in Spruce kernel");

  t.tick("mlir");

  // ---------------------------------------------------------------------------
  // Translate to LLVM
  // ---------------------------------------------------------------------------

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  auto optPipeline = mlir::makeOptimizingTransformer(
      /*optLevel=*/3, /*sizeLevel=*/0,
      /*targetMachine=*/nullptr);
  std::vector<llvm::StringRef> sharedLibs = {SPRUCE_CUDA_RUNTIME_LIBRARY};
  mlir::ExecutionEngineOptions options;
  options.transformer = optPipeline;
  options.sharedLibPaths = sharedLibs;

  auto maybeEngine = mlir::ExecutionEngine::create(*module, options);
  if (!maybeEngine)
    throw std::runtime_error(llvm::toString(maybeEngine.takeError()));
  auto engine = std::move(*maybeEngine);

  t.tick("execution-engine");

  // ---------------------------------------------------------------------------
  // Invoke
  // ---------------------------------------------------------------------------

  // std::vector<void *> args(7 * 2);
  //
  // auto denseA = (_Float16 *)std::malloc(inputs[0]->sizes[0] *
  //                                       inputs[0]->sizes[1] * sizeof(_Float16));
  // for (auto [crd, v] : *inputs[0]->data) {
  //   denseA[crd[0] * inputs[0]->sizes[1] + crd[1]] = (_Float16)v;
  // }
  //
  // args[0] = denseA;
  // args[1] = denseA;
  // args[2] = 0;
  // args[3] = reinterpret_cast<void *>(inputs[0]->sizes[0]);
  // args[4] = reinterpret_cast<void *>(inputs[0]->sizes[1]);
  // args[5] = reinterpret_cast<void *>(1);
  // args[6] = reinterpret_cast<void *>(1);
  //
  // args[7] = denseA;
  // args[8] = denseA;
  // args[9] = 0;
  // args[10] = reinterpret_cast<void *>(inputs[0]->sizes[0]);
  // args[11] = reinterpret_cast<void *>(inputs[0]->sizes[1]);
  // args[12] = reinterpret_cast<void *>(1);
  // args[13] = reinterpret_cast<void *>(1);

  if (auto error = engine->invokePacked("smoke"))
    throw std::runtime_error(llvm::toString(std::move(error)));

  t.tick("run");
  return t;
}

// -----------------------------------------------------------------------------
// Timing
// -----------------------------------------------------------------------------

void Timing::tick(const std::string &label) {
  auto now = Clock::now();
  double elapsed =
      std::chrono::duration<double, std::milli>(now - start).count();

  start = now;
  entries.push_back({label, elapsed});
}

void Timing::print() {
  for (auto &[label, time] : entries) {
    std::cout << label << ": " << time << " ms\n";
  }
}

} // namespace spruce
