// RUN: spruce-opt -verify-roundtrip %s | FileCheck %s

// CHECK: #[[CSR:.*]] = #sparse_tensor.encoding
#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK: #[[BSR:.*]] = #sparse_tensor.encoding
#BSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (
    d0 floordiv 2 : dense,
    d1 floordiv 3 : compressed,
    d0 mod 2 : dense,
    d1 mod 3 : dense
  )
}>

// CHECK: #[[B3D:.*]] = #sparse_tensor.encoding
#B3D = #sparse_tensor.encoding<{
  map = (d0, d1, d2) -> (d0 : dense, d1 : compressed, d2 : compressed)
}>

// CHECK-LABEL: func.func private @main
// CHECK-SAME:    tensor<?x?xf32, #[[CSR]]>
// CHECK-SAME:    tensor<?x?xf32, #[[BSR]]>
// CHECK-SAME:    tensor<?x?x?xf32, #[[B3D]]>
func.func private @main(
  tensor<?x?xf32, #CSR>,
  tensor<?x?xf32, #BSR>,
  tensor<?x?x?xf32, #B3D>
)
