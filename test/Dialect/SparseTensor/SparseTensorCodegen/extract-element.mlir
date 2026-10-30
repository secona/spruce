// RUN: spruce-opt -lower-sparse-foreach-to-scf -sparse-tensor-codegen -canonicalize %s | FileCheck %s

// CHECK: #[[CSR:.*]] = #sparse_tensor.encoding
#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: func.func @lookup
// CHECK-SAME:      %[[ROWPTR:[^:]+]]: memref<?xindex>
// CHECK-SAME:      %[[COLIDX:[^:]+]]: memref<?xindex>
// CHECK-SAME:      %[[VALUES:[^:]+]]: memref<?xf64>
// CHECK-SAME:      %[[SPEC:[^:]+]]: !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK-SAME:      %[[ROW:[^:]+]]: index
// CHECK-SAME:      %[[COL:[^:]+]]: index

// CHECK:         %[[C1_IDX:.*]] = arith.constant 1 : index
// CHECK:         %[[C0_IDX:.*]] = arith.constant 0 : index
// CHECK:         %[[C0_F64:.*]] = arith.constant 0.000000e+00 : f64

// CHECK:         %[[VAL_MEM_SZ:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] val_mem_sz : !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:         %[[VALUES_SV:.*]] = memref.subview %[[VALUES]][0] [%[[VAL_MEM_SZ]]] [1] : memref<?xf64> to memref<?xf64>
// CHECK:         %[[LVL_SZ:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] lvl_sz at 0 : !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:         %[[POS_MEM_SZ:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] pos_mem_sz at 1 : !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:         %[[ROWPTR_SV:.*]] = memref.subview %[[ROWPTR]][0] [%[[POS_MEM_SZ]]] [1] : memref<?xindex> to memref<?xindex>
// CHECK:         %[[CRD_MEM_SZ:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] crd_mem_sz at 1 : !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:         %[[COLIDX_SV:.*]] = memref.subview %[[COLIDX]][0] [%[[CRD_MEM_SZ]]] [1] : memref<?xindex> to memref<?xindex>

// CHECK:         %[[EL:.*]] = scf.for
// CHECK-SAME:        %[[I:.*]] = %[[C0_IDX]] to %[[LVL_SZ]]
// CHECK-SAME:        step %[[C1_IDX]]
// CHECK-SAME:        iter_args(%[[ACC:.*]] = %[[C0_F64]])
// CHECK-SAME:        -> (f64) {
// CHECK:           %[[ROW_CURR:.*]] = memref.load %[[ROWPTR_SV]][%[[I]]] : memref<?xindex>
// CHECK:           %[[I_PLUS_1:.*]] = arith.addi %[[I]], %[[C1_IDX]] : index
// CHECK:           %[[ROW_NEXT:.*]] = memref.load %[[ROWPTR_SV]][%[[I_PLUS_1]]] : memref<?xindex>
// CHECK:           %[[RESULT:.*]] = scf.for
// CHECK-SAME:          %[[J:.*]] = %[[ROW_CURR]] to %[[ROW_NEXT]]
// CHECK-SAME:          step %[[C1_IDX]]
// CHECK-SAME:          iter_args(%[[ACC_INNER:.*]] = %[[ACC]])
// CHECK-SAME:          -> (f64) {
// CHECK:             %[[COL_CURR:.*]] = memref.load %[[COLIDX_SV]][%[[J]]] : memref<?xindex>
// CHECK:             %[[VAL_CURR:.*]] = memref.load %[[VALUES_SV]][%[[J]]] : memref<?xf64>
// CHECK:             %[[SAME_ROW:.*]] = arith.cmpi eq, %[[I]], %[[ROW]] : index
// CHECK:             %[[SAME_COL:.*]] = arith.cmpi eq, %[[COL_CURR]], %[[COL]] : index
// CHECK:             %[[MATCH:.*]] = arith.andi %[[SAME_ROW]], %[[SAME_COL]] : i1
// CHECK:             %[[NEXT:.*]] = arith.select %[[MATCH]], %[[VAL_CURR]], %[[ACC_INNER]] : f64
// CHECK:             scf.yield %[[NEXT]] : f64
// CHECK:           }
// CHECK:           scf.yield %[[RESULT]] : f64
// CHECK:         }

// CHECK:         return %[[EL]] : f64

func.func @lookup(%A: tensor<?x?xf64, #CSR>,
                  %i: index, %j: index) -> f64 {
  %zero = arith.constant 0.0 : f64
  %result = sparse_tensor.foreach in %A init(%zero)
      : tensor<?x?xf64, #CSR>, f64 -> f64 do {
  ^bb0(%row: index, %col: index, %value: f64, %acc: f64):
    %sameRow = arith.cmpi eq, %row, %i : index
    %sameCol = arith.cmpi eq, %col, %j : index
    %match = arith.andi %sameRow, %sameCol : i1
    %next = arith.select %match, %value, %acc : f64
    sparse_tensor.yield %next : f64
  }
  return %result : f64
}
