// RUN: spruce-opt --sparsification-and-bufferization -canonicalize %s | FileCheck %s

// CHECK: #[[CSR:.*]] = #sparse_tensor.encoding
#CSR = #sparse_tensor.encoding<{
  map = (i, j) -> (i : dense, j : compressed)
}>

// CHECK-LABEL: func.func @spmv
// CHECK-SAME:      %[[ROWPTR:[^:]+]]: memref<?xindex>
// CHECK-SAME:      %[[COLIDX:[^:]+]]: memref<?xindex>
// CHECK-SAME:      %[[VALUES:[^:]+]]: memref<?xf32>
// CHECK-SAME:      %[[SPEC:[^:]+]]: !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK-SAME:      %[[X_VEC:[^:]+]]: memref<?xf32>
// CHECK-SAME:      %[[Y_VEC:[^:]+]]: memref<?xf32>
// CHECK-SAME:      -> memref<?xf32>

// CHECK:         %[[C1_IDX:.*]] = arith.constant 1 : index
// CHECK:         %[[C0_IDX:.*]] = arith.constant 0 : index

// CHECK:         %[[VAL_MEM_SZ:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] val_mem_sz : !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:         %[[VALUES_SV:.*]] = memref.subview %[[VALUES]][0] [%[[VAL_MEM_SZ]]] [1] : memref<?xf32> to memref<?xf32>
// CHECK:         %[[LVL_SZ:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] lvl_sz at 0 : !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:         %[[POS_MEM_SZ:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] pos_mem_sz at 1 : !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:         %[[ROWPTR_SV:.*]] = memref.subview %[[ROWPTR]][0] [%[[POS_MEM_SZ]]] [1] : memref<?xindex> to memref<?xindex>
// CHECK:         %[[CRD_MEM_SZ:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] crd_mem_sz at 1 : !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:         %[[COLIDX_SV:.*]] = memref.subview %[[COLIDX]][0] [%[[CRD_MEM_SZ]]] [1] : memref<?xindex> to memref<?xindex>

// CHECK:         scf.for %[[I:.*]] = %[[C0_IDX]] to %[[LVL_SZ]] step %[[C1_IDX]] {
// CHECK:           %[[Y_AT_I:.*]] = memref.load %[[Y_VEC]][%[[I]]] : memref<?xf32>
// CHECK:           %[[ROW_CURR:.*]] = memref.load %[[ROWPTR_SV]][%[[I]]] : memref<?xindex>
// CHECK:           %[[I_PLUS_1:.*]] = arith.addi %[[I]], %[[C1_IDX]] : index
// CHECK:           %[[ROW_NEXT:.*]] = memref.load %[[ROWPTR_SV]][%[[I_PLUS_1]]] : memref<?xindex>
// CHECK:           %[[Y_AT_I_NEXT:.*]] = scf.for %[[J:.*]] = %[[ROW_CURR]] to %[[ROW_NEXT]] step %[[C1_IDX]] iter_args(%[[ACC:.*]] = %[[Y_AT_I]]) -> (f32) {
// CHECK:             %[[COL_CURR:.*]] = memref.load %[[COLIDX_SV]][%[[J]]] : memref<?xindex>
// CHECK:             %[[VAL_CURR:.*]] = memref.load %[[VALUES_SV]][%[[J]]] : memref<?xf32>
// CHECK:             %[[X_AT_COL:.*]] = memref.load %[[X_VEC]][%[[COL_CURR]]] : memref<?xf32>
// CHECK:             %[[VAL_NEXT0:.*]] = arith.mulf %[[VAL_CURR]], %[[X_AT_COL]] : f32
// CHECK:             %[[VAL_NEXT:.*]] = arith.addf %[[ACC]], %[[VAL_NEXT0]] : f32
// CHECK:             scf.yield %[[VAL_NEXT]] : f32
// CHECK:           }
// CHECK:           memref.store %[[Y_AT_I_NEXT]], %[[Y_VEC]][%[[I]]] : memref<?xf32>
// CHECK:         }

// CHECK:         return %[[Y_VEC]] : memref<?xf32>


func.func @spmv(%A: tensor<?x?xf32, #CSR>,
                %x: tensor<?xf32>,
                %y: tensor<?xf32>) -> tensor<?xf32> {
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i,j) -> (i,j)>,
      affine_map<(i,j) -> (j)>,
      affine_map<(i,j) -> (i)>
    ],
    iterator_types = [
      "parallel",
      "reduction"
    ]
  } ins(%A, %x: tensor<?x?xf32, #CSR>, tensor<?xf32>)
    outs(%y: tensor<?xf32>) {
    ^bb0(%a: f32, %x_value: f32, %y_value: f32):
      %product = arith.mulf %a, %x_value : f32
      %sum = arith.addf %y_value, %product : f32
      linalg.yield %sum : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}
