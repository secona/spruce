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

// CHECK-LABEL: func.func @block_coordinates_static(
// CHECK-SAME: %[[STATIC_T:[^:]+]]: tensor<6x12xf32, #[[BSR]]>, %[[STATIC_P:[^:]+]]: index) -> (index, index)
// CHECK: %[[STATIC_ROW:.*]], %[[STATIC_COL:.*]] = spruce.block_coordinates %[[STATIC_T]][%[[STATIC_P]]] : tensor<6x12xf32, #[[BSR]]>
// CHECK: return %[[STATIC_ROW]], %[[STATIC_COL]] : index, index
func.func @block_coordinates_static(%T: tensor<6x12xf32, #BSR>, %p: index) -> (index, index) {
  %row, %col = spruce.block_coordinates %T[%p] : tensor<6x12xf32, #BSR>
  return %row, %col : index, index
}

// CHECK-LABEL: func.func @block_coordinates_dynamic(
// CHECK-SAME: %[[DYNAMIC_T:[^:]+]]: tensor<?x?xf32, #[[BSR]]>, %[[DYNAMIC_P:[^:]+]]: index) -> (index, index)
// CHECK: %[[DYNAMIC_ROW:.*]], %[[DYNAMIC_COL:.*]] = spruce.block_coordinates %[[DYNAMIC_T]][%[[DYNAMIC_P]]] : tensor<?x?xf32, #[[BSR]]>
// CHECK: return %[[DYNAMIC_ROW]], %[[DYNAMIC_COL]] : index, index
func.func @block_coordinates_dynamic(%T: tensor<?x?xf32, #BSR>, %p: index) -> (index, index) {
  %row, %col = spruce.block_coordinates %T[%p] : tensor<?x?xf32, #BSR>
  return %row, %col : index, index
}
