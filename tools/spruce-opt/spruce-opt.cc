// --- spruce-opt.cc: Spruce optimizer driver ----------------------------------

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "spruce/Dialect/Spruce/IR/SpruceDialect.h"
#include "spruce/Dialect/SpruceGPU/IR/SpruceGPUDialect.h"
#include "spruce/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::spruce::registerSprucePasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::spruce::SpruceDialect,
                  mlir::spruce_gpu::SpruceGPUDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllGPUToLLVMIRTranslations(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Spruce optimizer driver\n", registry));
}
