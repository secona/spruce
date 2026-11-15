// RUN: spruce-opt -spruce-lower-block-ops -canonicalize -cse -verify-roundtrip %s | FileCheck %s --implicit-check-not=spruce.num_blocks

// CHECK: #[[BSR:.*]] = #sparse_tensor.encoding
#BSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (
    d0 floordiv 16 : dense,
    d1 floordiv 8 : compressed,
    d0 mod 16 : dense,
    d1 mod 8 : dense
  )
}>

// CHECK-LABEL: func.func @num_blocks(
// CHECK-SAME:      %[[A:[^:]+]]: tensor<32x16xf32, #[[BSR]]>) -> index
// CHECK:         %[[C2:.*]] = arith.constant 2 : index
// CHECK:         %[[POS:.*]] = sparse_tensor.positions %[[A]] level = 1 : tensor<32x16xf32, #[[BSR]]> to memref<?xindex>
// CHECK:         %[[COUNT:.*]] = memref.load %[[POS]][%[[C2]]] : memref<?xindex>
// CHECK:         return %[[COUNT]] : index

func.func @num_blocks(%A: tensor<32x16xf32, #BSR>) -> index {
  %count = spruce.num_blocks %A : tensor<32x16xf32, #BSR>
  return %count : index
}
