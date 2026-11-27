#include "spruce/Spruce.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
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

  engine->initialize();
  auto maybeSpgemm = engine->lookupPacked("spgemm");
  if (!maybeSpgemm)
    throw std::runtime_error(llvm::toString(maybeSpgemm.takeError()));
  kernelFn = *maybeSpgemm;

  t.tick("execution-engine");

  // ---------------------------------------------------------------------------
  // Invoke
  // ---------------------------------------------------------------------------
  
  _Float16 aData[16 * 8]{}, bData[8 * 8]{};

  auto pack = [](const InputTensor &input, _Float16 *dst, int rows) {
    taco::Tensor<double> matrix(taco::read(input.path, taco::Sparse));
    for (const auto &[crd, value] : matrix) {
      if (crd[0] >= 0 && crd[0] < rows && crd[1] >= 0 && crd[1] < 8)
        dst[crd[0] * 8 + crd[1]] = static_cast<_Float16>(value);
    }
  };

  pack(ins[0], aData, 16);
  pack(ins[1], bData, 8);

  StridedMemRefType<_Float16, 2> a{aData, aData, 0, {16, 8}, {8, 1}};
  StridedMemRefType<_Float16, 2> b{bData, bData, 0, {8, 8}, {8, 1}};
  StridedMemRefType<_Float16, 2> c{};

  void *args[] = {
      &a.basePtr, &a.data, &a.offset,
      &a.sizes[0], &a.sizes[1], &a.strides[0], &a.strides[1],
      &b.basePtr, &b.data, &b.offset,
      &b.sizes[0], &b.sizes[1], &b.strides[0], &b.strides[1],
      &c,
  };

  kernelFn(args);

  t.tick("run");

  // for (int64_t row = 0; row < c.sizes[0]; ++row) {
  //   for (int64_t col = 0; col < c.sizes[1]; ++col) {
  //     auto index = c.offset + row * c.strides[0] + col * c.strides[1];
  //     std::cout << (col ? " " : "") << static_cast<float>(c.data[index]);
  //   }
  //   std::cout << '\n';
  // }

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

void Timing::Aggregator::iterate(int n, const std::function<Timing()> &f) {
  for (int i = 0; i < n; ++i) {
    Timing t = f();
    for (const auto &[label, time] : t.entries) {
      samples[label].push_back(time);
    }
  }
}

double median(std::vector<double> v) {
  const size_t n = v.size();
  const size_t mid = n / 2;

  std::nth_element(v.begin(), v.begin() + mid, v.end());

  if (n % 2 == 1) {
    return v[mid];
  }

  const double lower_mid = *std::max_element(v.begin(), v.begin() + mid);
  return lower_mid + (v[mid] - lower_mid) / 2.0;
}

void Timing::Aggregator::print() {
  std::cout << "-- Timing::Aggregator --" << "\n";
  for (auto &[key, times] : samples) {
    double m = median(times);
    std::cout << key << ": " << "median=" << m << " ms " << "n=" << times.size()
              << "\n";
  }
}

} // namespace spruce
