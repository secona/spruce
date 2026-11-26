// --- spruce-opt.cc: Spruce optimizer driver ----------------------------------

#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "spruce/Dialect/Spruce/IR/SpruceDialect.h"
#include "spruce/Conversions/Passes.h"
#include "spruce/Dialect/Spruce/Transforms/Passes.h"
#include "spruce/Pipelines/SprucePipeline.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::spruce::registerSpruceConversionsPasses();
  mlir::spruce::registerSpruceTransformsPasses();
  mlir::spruce::registerSprucePipeline();

  mlir::DialectRegistry registry;
  registry.insert<mlir::spruce::SpruceDialect>();
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllGPUToLLVMIRTranslations(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Spruce optimizer driver\n", registry));
}
