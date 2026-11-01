# Spruce

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
