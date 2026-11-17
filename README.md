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

## Lowering Progress

Run the current lowering pipeline with:

```sh
build/bin/spruce-opt --spruce-pipeline input.mlir
```

The pipeline runs `convert-linalg-to-spruce`, `spruce-tile-mma`,
`convert-spruce-to-vector`, `spruce-lower-block-ops`, canonicalization, and common
subexpression elimination.

The main progress test is
[spgemm-bsr-dense-dense.mlir](test/Dialect/Spruce/Transforms/SprucePipeline/spgemm-bsr-dense-dense.mlir).
It uses the same tile-aligned input as the Linalg-to-Spruce conversion test and
checks the current endpoint: SCF loops with vector contractions and the
stored-block count loaded from the BSR positions buffer. Sparse block access
still uses `spruce.block_coordinates` and `spruce.get_block`; lowering those
operations is needed before proceeding to full bufferization and GPU instruction
lowering.
