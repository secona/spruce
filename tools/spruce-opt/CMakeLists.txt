add_llvm_executable(spruce-opt spruce-opt.cc)
llvm_update_compile_flags(spruce-opt)

mlir_target_link_libraries(spruce-opt PRIVATE
  SpruceDialect
  SpruceGPUDialect
  SpruceTransforms
  MLIROptLib
  MLIRRegisterAllDialects
  MLIRRegisterAllExtensions
  MLIRRegisterAllPasses
  MLIRToLLVMIRTranslationRegistration
)

mlir_check_all_link_libraries(spruce-opt)
