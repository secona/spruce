#include "spruce/Conversions/Passes.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

namespace mlir::spruce {
#define GEN_PASS_DEF_CONVERTSPRUCEGPUTONVVM
#include "spruce/Conversions/Passes.h.inc"

namespace {
struct ConvertSpruceGPUToNVVM
    : impl::ConvertSpruceGPUToNVVMBase<ConvertSpruceGPUToNVVM> {
  using Base::Base;

  void runOnOperation() override {
    // TODO: Add SpruceGPU to NVVM lowering patterns.
  }
};
} // namespace
} // namespace mlir::spruce
