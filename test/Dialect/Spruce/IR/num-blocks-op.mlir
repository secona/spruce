// RUN: spruce-opt -verify-roundtrip %s | FileCheck %s

// CHECK-DAG: #[[BSR:.*]] = #sparse_tensor.encoding
#BSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (
    d0 floordiv 2 : dense,
    d1 floordiv 3 : compressed,
    d0 mod 2 : dense,
    d1 mod 3 : dense
  )
}>

// CHECK-LABEL: func.func @num_blocks_static(
// CHECK-SAME:      %[[STATIC_T:[^:]+]]: tensor<6x12xf32, #[[BSR]]>) -> index
// CHECK:         %[[STATIC_COUNT:.*]] = spruce.num_blocks %[[STATIC_T]] : tensor<6x12xf32, #[[BSR]]>
// CHECK:         return %[[STATIC_COUNT]] : index
func.func @num_blocks_static(%T: tensor<6x12xf32, #BSR>) -> index {
  %count = spruce.num_blocks %T : tensor<6x12xf32, #BSR>
  return %count : index
}

// CHECK-LABEL: func.func @num_blocks_dynamic(
// CHECK-SAME:      %[[DYNAMIC_T:[^:]+]]: tensor<?x?xf32, #[[BSR]]>) -> index
// CHECK:         %[[DYNAMIC_COUNT:.*]] = spruce.num_blocks %[[DYNAMIC_T]] : tensor<?x?xf32, #[[BSR]]>
// CHECK:         return %[[DYNAMIC_COUNT]] : index
func.func @num_blocks_dynamic(%T: tensor<?x?xf32, #BSR>) -> index {
  %count = spruce.num_blocks %T : tensor<?x?xf32, #BSR>
  return %count : index
}
