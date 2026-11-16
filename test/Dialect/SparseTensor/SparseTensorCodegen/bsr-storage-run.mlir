// RUN: spruce-opt -sparsifier="enable-runtime-library=false enable-buffer-initialization=true" %s | \
// RUN:   mlir-runner -e main -entry-point-result=void -shared-libs=%mlir_c_runner_utils | FileCheck %s

// Storage reference for the matrices in
// test/Dialect/Spruce/Transforms/SpruceLowerBlockOps/block-coordinates-run.mlir.
// Each stored 2x2 block occupies four consecutive values in row-major order.
// positions[row]:positions[row + 1] gives the row's range of block indices,
// and coordinates[p] gives block p's column. nse counts stored scalar values.
// pos[1] and crd[1] label the buffers for compressed level 1, not element 1.

#BSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (
    d0 floordiv 2 : dense,
    d1 floordiv 2 : compressed,
    d0 mod 2 : dense,
    d1 mod 2 : dense
  )
}>

// CHECK-LABEL: ---- Sparse Tensor ----
// CHECK-NEXT: nse = 8
// CHECK-NEXT: dim = ( 4, 4 )
// CHECK-NEXT: lvl = ( 2, 2, 2, 2 )
// CHECK-NEXT: pos[1] : ( 0, 1, 2 )
// CHECK-NEXT: crd[1] : ( 0, 1 )
// CHECK-NEXT: values : ( 1, 2, 3, 4, 5, 6, 7, 8 )
// CHECK-NEXT: ----
func.func @two_blocks() {
  %dense = arith.constant dense<[[1.0, 2.0, 0.0, 0.0],
                                 [3.0, 4.0, 0.0, 0.0],
                                 [0.0, 0.0, 5.0, 6.0],
                                 [0.0, 0.0, 7.0, 8.0]]> : tensor<4x4xf64>
  %sparse = sparse_tensor.convert %dense : tensor<4x4xf64> to tensor<4x4xf64, #BSR>
  sparse_tensor.print %sparse : tensor<4x4xf64, #BSR>
  bufferization.dealloc_tensor %sparse : tensor<4x4xf64, #BSR>
  return
}

// CHECK-LABEL: ---- Sparse Tensor ----
// CHECK-NEXT: nse = 12
// CHECK-NEXT: dim = ( 4, 4 )
// CHECK-NEXT: lvl = ( 2, 2, 2, 2 )
// CHECK-NEXT: pos[1] : ( 0, 2, 3 )
// CHECK-NEXT: crd[1] : ( 0, 1, 1 )
// CHECK-NEXT: values : ( 1, 2, 3, 4, 9, 10, 11, 12, 5, 6, 7, 8 )
// CHECK-NEXT: ----
func.func @three_blocks() {
  %dense = arith.constant dense<[[1.0, 2.0, 9.0, 10.0],
                                 [3.0, 4.0, 11.0, 12.0],
                                 [0.0, 0.0, 5.0, 6.0],
                                 [0.0, 0.0, 7.0, 8.0]]> : tensor<4x4xf64>
  %sparse = sparse_tensor.convert %dense : tensor<4x4xf64> to tensor<4x4xf64, #BSR>
  sparse_tensor.print %sparse : tensor<4x4xf64, #BSR>
  bufferization.dealloc_tensor %sparse : tensor<4x4xf64, #BSR>
  return
}

func.func @main() {
  func.call @two_blocks() : () -> ()
  func.call @three_blocks() : () -> ()
  return
}
