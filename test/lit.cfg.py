import os

import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst

config.name = "SPRUCE"
config.test_format = lit.formats.ShTest(execute_external=False)
config.suffixes = [".mlir"]
config.excludes = ["Inputs"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = config.spruce_test_exec_root

llvm_config.with_system_environment(["HOME", "TMP", "TEMP", "LD_LIBRARY_PATH"])
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.use_default_substitutions()
llvm_config.add_tool_substitutions(
    [
        ToolSubst("spruce-opt", unresolved="fatal"),
        ToolSubst("mlir-runner", unresolved="fatal"),
        ToolSubst("%mlir_c_runner_utils", config.mlir_c_runner_utils),
    ],
    [config.spruce_tools_dir, config.llvm_tools_dir],
)
