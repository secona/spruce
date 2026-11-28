# Spruce

Spruce is an attempt to improve sparse tensor compilation using NVIDIA Tensor
Cores. The base of Spruce is the SparseTensor dialect in MLIR. Spruce takes a
different approach to the upstream `-sparse-gpu-codegen` pass where its getting
sparsity information from the sparse tensor expression.

The main focus of Spruce is the `f16` type with static matrix sizes and `sm_86`
compute capability, which should work for newer GPUs.

## How to Build

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR="$PWD/../llvm-project/build/lib/cmake/mlir" \
  -DLLVM_DIR="$PWD/../llvm-project/build/lib/cmake/llvm"

cmake --build build

ctest --test-dir build --output-on-failure
```

Adjust `MLIR_DIR` and `LLVM_DIR` as needed.

## Lowering Progress

Run the current lowering pipeline with:

```sh
build/bin/spruce-opt --spruce-pipeline input.mlir
```
