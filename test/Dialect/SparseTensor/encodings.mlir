// RUN: spruce-opt -verify-roundtrip %s | FileCheck %s

// CHECK: #[[CSR:.*]] = #sparse_tensor.encoding
#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK: #[[B3D:.*]] = #sparse_tensor.encoding
#B3D = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : compressed, d2 : compressed)
}>

// CHECK-LABEL: func.func private @main
// CHECK-SAME:    tensor<?x?xf32, #[[CSR]]>
// CHECK-SAME:    tensor<?x?x?xf32, #[[B3D]]>
func.func private @main(
  tensor<?x?xf32, #CSR>,
  tensor<?x?x?xf32, #B3D>
)
