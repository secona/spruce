// RUN: spruce-opt -split-input-file --sparsification-and-bufferization -canonicalize -cse %s | FileCheck %s

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

// -----

// CHECK: #[[BSR:.*]] = #sparse_tensor.encoding
#BSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (
    d0 floordiv 2 : dense,
    d1 floordiv 3 : compressed,
    d0 mod 2 : dense,
    d1 mod 3 : dense
  )
}>

// CHECK-LABEL: func.func @spmv
// CHECK-SAME:      %[[ROWPTR:[^:]+]]: memref<?xindex>
// CHECK-SAME:      %[[COLIDX:[^:]+]]: memref<?xindex>
// CHECK-SAME:      %[[VALUES:[^:]+]]: memref<?xf32>
// CHECK-SAME:      %[[SPEC:[^:]+]]: !sparse_tensor.storage_specifier<#[[BSR]]>
// CHECK-SAME:      %[[X_VEC:[^:]+]]: memref<?xf32>
// CHECK-SAME:      %[[Y_VEC:[^:]+]]: memref<?xf32>
// CHECK-SAME:      -> memref<?xf32>

// CHECK:         %[[C1_IDX:.*]] = arith.constant 1 : index
// CHECK:         %[[C3_IDX:.*]] = arith.constant 3 : index
// CHECK:         %[[C2_IDX:.*]] = arith.constant 2 : index
// CHECK:         %[[C0_IDX:.*]] = arith.constant 0 : index

// CHECK:         %[[VAL_MEM_SZ:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] val_mem_sz : !sparse_tensor.storage_specifier<#[[BSR]]>
// CHECK:         %[[VALUES_SV:.*]] = memref.subview %[[VALUES]][0] [%[[VAL_MEM_SZ]]] [1] : memref<?xf32> to memref<?xf32>
// CHECK:         %[[LVL_SZ:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] lvl_sz at 0 : !sparse_tensor.storage_specifier<#[[BSR]]>
// CHECK:         %[[POS_MEM_SZ:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] pos_mem_sz at 1 : !sparse_tensor.storage_specifier<#[[BSR]]>
// CHECK:         %[[ROWPTR_SV:.*]] = memref.subview %[[ROWPTR]][0] [%[[POS_MEM_SZ]]] [1] : memref<?xindex> to memref<?xindex>
// CHECK:         %[[CRD_MEM_SZ:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] crd_mem_sz at 1 : !sparse_tensor.storage_specifier<#[[BSR]]>
// CHECK:         %[[COLIDX_SV:.*]] = memref.subview %[[COLIDX]][0] [%[[CRD_MEM_SZ]]] [1] : memref<?xindex> to memref<?xindex>

// CHECK:         scf.for %[[I:.*]] = %c0 to %1 step %c1 {
// CHECK:           %[[ROW_CURR:.*]] = memref.load %[[ROWPTR_SV]][%[[I]]] : memref<?xindex>
// CHECK:           %[[I_PLUS_1:.*]] = arith.addi %[[I]], %[[C1_IDX]] : index
// CHECK:           %[[ROW_NEXT:.*]] = memref.load %[[ROWPTR_SV]][%[[I_PLUS_1]]] : memref<?xindex>
// CHECK:           %[[I_TIMES_2:.*]] = arith.muli %[[I]], %[[C2_IDX]] : index
// CHECK:           scf.for %arg7 = %4 to %6 step %c1 {
// CHECK:             %8 = memref.load %subview_1[%arg7] : memref<?xindex>
// CHECK:             %9 = arith.muli %arg7, %c2 : index
// CHECK:             %10 = arith.muli %8, %c3 : index
// CHECK:             scf.for %arg8 = %c0 to %c2 step %c1 {
// CHECK:               %11 = arith.addi %arg8, %9 : index
// CHECK:               %12 = arith.addi %[[I_TIMES_2]], %arg8 : index
// CHECK:               %13 = memref.load %arg5[%12] : memref<?xf32>
// CHECK:               %14 = arith.muli %11, %c3 : index
// CHECK:               %15 = scf.for %arg9 = %c0 to %c3 step %c1 iter_args(%arg10 = %13) -> (f32) {
// CHECK:                 %16 = arith.addi %arg9, %14 : index
// CHECK:                 %17 = memref.load %subview[%16] : memref<?xf32>
// CHECK:                 %18 = arith.addi %10, %arg9 : index
// CHECK:                 %19 = memref.load %arg4[%18] : memref<?xf32>
// CHECK:                 %20 = arith.mulf %17, %19 : f32
// CHECK:                 %21 = arith.addf %arg10, %20 : f32
// CHECK:                 scf.yield %21 : f32
// CHECK:               } {"Emitted from" = "linalg.generic"}
// CHECK:               memref.store %15, %arg5[%12] : memref<?xf32>
// CHECK:             } {"Emitted from" = "linalg.generic"}
// CHECK:           } {"Emitted from" = "linalg.generic"

// CHECK:         return %[[Y_VEC]] : memref<?xf32>

func.func @spmv(%A: tensor<?x?xf32, #BSR>,
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
  } ins(%A, %x: tensor<?x?xf32, #BSR>, tensor<?xf32>)
    outs(%y: tensor<?xf32>) {
    ^bb0(%a: f32, %x_value: f32, %y_value: f32):
      %product = arith.mulf %a, %x_value : f32
      %sum = arith.addf %y_value, %product : f32
      linalg.yield %sum : f32
  } -> tensor<?xf32>
  return %result : tensor<?xf32>
}
