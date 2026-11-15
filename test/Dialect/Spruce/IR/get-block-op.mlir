// RUN: spruce-opt -split-input-file %s | FileCheck %s

// CHECK-DAG: #[[BSR:.*]] = #sparse_tensor.encoding
#BSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (
    d0 floordiv 2 : dense,
    d1 floordiv 3 : compressed,
    d0 mod 2 : dense,
    d1 mod 3 : dense
  )
}>

// CHECK-LABEL: func.func @get_block(
// CHECK-SAME:      %[[T:.*]]: tensor<32x32xf16, #[[BSR]]>) -> tensor<3x2xf16>
// CHECK:         %[[IDX:.*]] = arith.constant 1 : index
// CHECK:         %[[RES:.*]] = spruce.get_block %[[T]][%[[IDX]]] : tensor<32x32xf16, #[[BSR]]> -> tensor<3x2xf16>
// CHECK:         return %[[RES]] : tensor<3x2xf16>

func.func @get_block(%T: tensor<32x32xf16, #BSR>) -> tensor<3x2xf16> {
  %idx = arith.constant 1 : index
  %result = spruce.get_block %T[%idx] : tensor<32x32xf16, #BSR> -> tensor<3x2xf16>
  return %result : tensor<3x2xf16>
}
