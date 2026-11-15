// RUN: spruce-opt -spruce-lower-block-ops -sparsifier="enable-runtime-library=false enable-buffer-initialization=true" %s | \
// RUN:   mlir-runner -e main -entry-point-result=void -shared-libs=%mlir_c_runner_utils | FileCheck %s

#BSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (
    d0 floordiv 2 : dense,
    d1 floordiv 2 : compressed,
    d0 mod 2 : dense,
    d1 mod 2 : dense
  )
}>

func.func @two_blocks() {
  // Only the upper-left and lower-right 2x2 blocks contain values.
  %dense = arith.constant dense<[[1.0, 2.0, 0.0, 0.0],
                                 [3.0, 4.0, 0.0, 0.0],
                                 [0.0, 0.0, 5.0, 6.0],
                                 [0.0, 0.0, 7.0, 8.0]]> : tensor<4x4xf64>
  %sparse = sparse_tensor.convert %dense : tensor<4x4xf64> to tensor<4x4xf64, #BSR>
  %count = spruce.num_blocks %sparse : tensor<4x4xf64, #BSR>

  // CHECK: {{^2$}}
  vector.print %count : index

  bufferization.dealloc_tensor %sparse : tensor<4x4xf64, #BSR>
  return
}

func.func @three_blocks() {
  // The upper-left, upper-right, and lower-right 2x2 blocks contain values.
  %dense = arith.constant dense<[[1.0, 2.0, 9.0, 10.0],
                                 [3.0, 4.0, 11.0, 12.0],
                                 [0.0, 0.0, 5.0, 6.0],
                                 [0.0, 0.0, 7.0, 8.0]]> : tensor<4x4xf64>
  %sparse = sparse_tensor.convert %dense : tensor<4x4xf64> to tensor<4x4xf64, #BSR>
  %count = spruce.num_blocks %sparse : tensor<4x4xf64, #BSR>

  // CHECK-NEXT: {{^3$}}
  vector.print %count : index

  bufferization.dealloc_tensor %sparse : tensor<4x4xf64, #BSR>
  return
}

func.func @main() {
  func.call @two_blocks() : () -> ()
  func.call @three_blocks() : () -> ()
  return
}
