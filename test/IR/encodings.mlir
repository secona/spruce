// RUN: spruce-opt -verify-roundtrip %s | FileCheck %s

// CHECK-DAG: #[[CSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 : dense, d1 : compressed) }>
// CHECK-DAG: #[[SPRUCE_CSR:.*]] = #spruce.encoding<{sparse = #[[CSR]]}>
#SpruceCSR = #spruce.encoding<{
  sparse = #sparse_tensor.encoding<{
    map = (d0, d1) -> (d0 : dense, d1 : compressed)
  }>
}>

// CHECK-DAG: #[[BSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) -> (d0 floordiv 2 : dense, d1 floordiv 3 : compressed, d0 mod 2 : dense, d1 mod 3 : dense) }>
// CHECK-DAG: #[[SPRUCE_BSR:.*]] = #spruce.encoding<{sparse = #[[BSR]]}>
#SpruceBSR = #spruce.encoding<{
  sparse = #sparse_tensor.encoding<{
    map = (d0, d1) -> (
      d0 floordiv 2 : dense,
      d1 floordiv 3 : compressed,
      d0 mod 2 : dense,
      d1 mod 3 : dense
    )
  }>
}>

// CHECK-LABEL: func.func private @main
// CHECK-SAME:    tensor<?x?xf32, #[[SPRUCE_CSR]]>
// CHECK-SAME:    tensor<?x?xf32, #[[SPRUCE_BSR]]>
func.func private @main(
  tensor<?x?xf32, #SpruceCSR>,
  tensor<?x?xf32, #SpruceBSR>
)
