#include "spruce/Conversions/Passes.h"

#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"

namespace mlir::spruce {
#define GEN_PASS_DEF_CONVERTSPRUCEGPUTONVGPU
#include "spruce/Conversions/Passes.h.inc"

namespace {
struct ConvertSpruceGPUToNVGPU
    : impl::ConvertSpruceGPUToNVGPUBase<ConvertSpruceGPUToNVGPU> {
  using Base::Base;

  void runOnOperation() override {
    // TODO: Add SpruceGPU to NVGPU lowering patterns.
  }
};
} // namespace
} // namespace mlir::spruce
