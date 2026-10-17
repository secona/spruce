// RUN: spruce-opt -sparse-tensor-codegen -canonicalize %s | FileCheck %s

// CHECK: #[[CSR:.*]] = #sparse_tensor.encoding
#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: func.func private @_insert_dense_compressed_8_8_f32_0_0
// CHECK-SAME:      %[[ROWPTR:[^:]+]]: memref<?xindex>
// CHECK-SAME:      %[[COLIDX:[^:]+]]: memref<?xindex>
// CHECK-SAME:      %[[VALUES:[^:]+]]: memref<?xf32>
// CHECK-SAME:      %[[SPEC:[^:]+]]: !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK-SAME:      %[[ROW_IDX:[^:]+]]: index
// CHECK-SAME:      %[[COL_IDX:[^:]+]]: index
// CHECK-SAME:      %[[VALUE:[^:]+]]: f32

// CHECK:         %[[FALSE:.*]] = arith.constant false
// CHECK:         %[[C1_IDX:.*]] = arith.constant 1 : index

// CHECK:         %[[ROW_IDX_P1:.*]] = arith.addi %[[ROW_IDX]], %[[C1_IDX]] : index
// CHECK:         %[[POS_START:.*]] = memref.load %[[ROWPTR]][%[[ROW_IDX]]] : memref<?xindex>
// CHECK:         %[[POS_END:.*]] = memref.load %[[ROWPTR]][%[[ROW_IDX_P1]]] : memref<?xindex>
// CHECK:         %[[CRD_MEM_SIZE:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] crd_mem_sz at 1 : !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:         %[[POS_LAST:.*]] = arith.subi %[[POS_END]], %[[C1_IDX]] : index

// CHECK:         %[[ULT:.*]] = arith.cmpi ult, %[[POS_START]], %[[POS_END]] : index
// CHECK:         %[[REUSE:.*]] = scf.if %[[ULT]] -> (i1) {
// CHECK:           %[[LAST_COL:.*]] = memref.load %[[COLIDX]][%[[POS_LAST]]] : memref<?xindex>
// CHECK:           %[[EQ:.*]] = arith.cmpi eq, %[[LAST_COL]], %[[COL_IDX]] : index
// CHECK:           scf.yield %[[EQ]] : i1
// CHECK:         } else {
// CHECK:           memref.store %[[CRD_MEM_SIZE]], %[[ROWPTR]][%[[ROW_IDX]]] : memref<?xindex>
// CHECK:           scf.yield %[[FALSE]] : i1
// CHECK:         }

// CHECK:         %7:2 = scf.if %[[REUSE]] -> (memref<?xindex>, !sparse_tensor.storage_specifier<#[[CSR]]>) {
// CHECK:           scf.yield %[[COLIDX]], %[[SPEC]] : memref<?xindex>, !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:         } else {
// CHECK:           %10 = arith.addi %[[CRD_MEM_SIZE]], %[[C1_IDX]] : index
// CHECK:           memref.store %10, %[[ROWPTR]][%[[ROW_IDX_P1]]] : memref<?xindex>
// CHECK:           %11 = sparse_tensor.storage_specifier.get %[[SPEC]] crd_mem_sz at 1 : !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:           %outBuffer_0, %newSize_1 = sparse_tensor.push_back %11, %[[COLIDX]], %[[COL_IDX]] : index, memref<?xindex>, index
// CHECK:           %12 = sparse_tensor.storage_specifier.set %[[SPEC]] crd_mem_sz at 1 with %newSize_1 : !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:           scf.yield %outBuffer_0, %12 : memref<?xindex>, !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:         }

// CHECK:         %[[VAL_MEM_SZ_OLD:.*]] = sparse_tensor.storage_specifier.get %7#1 val_mem_sz : !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:         %[[VALUES_NEW:.*]], %[[NEW_SIZE:.*]] = sparse_tensor.push_back %[[VAL_MEM_SZ_OLD]], %[[VALUES]], %[[VALUE]] : index, memref<?xf32>, f32
// CHECK:         %[[VAL_MEM_SZ_NEW:.*]] = sparse_tensor.storage_specifier.set %7#1 val_mem_sz with %[[NEW_SIZE]] : !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:         return %[[ROWPTR]], %7#0, %[[VALUES_NEW]], %[[VAL_MEM_SZ_NEW]] : memref<?xindex>, memref<?xindex>, memref<?xf32>, !sparse_tensor.storage_specifier<#[[CSR]]>

// CHECK-LABEL: func.func @insert_to_csr
// CHECK-SAME:      %arg0: memref<?xindex>
// CHECK-SAME:      %arg1: memref<?xindex>
// CHECK-SAME:      %arg2: memref<?xf32>
// CHECK-SAME:      %arg3: !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK-SAME:      -> (memref<?xindex>,
// CHECK-SAME:          memref<?xindex>,
// CHECK-SAME:          memref<?xf32>,
// CHECK-SAME:          !sparse_tensor.storage_specifier<#[[CSR]]>)
// CHECK:         %[[C1_IDX:.*]] = arith.constant 1 : index
// CHECK:         %[[C1_F32:.*]] = arith.constant 1.000000e+00 : f32
// CHECK:         %0:4 = call @_insert_dense_compressed_8_8_f32_0_0(%arg0, %arg1, %arg2, %arg3, %[[C1_IDX]], %[[C1_IDX]], %[[C1_F32]])
// CHECK:              : (memref<?xindex>, memref<?xindex>, memref<?xf32>, !sparse_tensor.storage_specifier<#[[CSR]]>, index, index, f32)
// CHECK-SAME:         -> (memref<?xindex>, memref<?xindex>, memref<?xf32>, !sparse_tensor.storage_specifier<#[[CSR]]>)
// CHECK:         return %0#0, %0#1, %0#2, %0#3 : memref<?xindex>, memref<?xindex>, memref<?xf32>, !sparse_tensor.storage_specifier<#[[CSR]]>

func.func @insert_to_csr(%csr: tensor<8x8xf32, #CSR>) -> tensor<8x8xf32, #CSR> {
  %c1_idx = arith.constant 1 : index
  %c1 = arith.constant 1.0 : f32

  %csr0 = tensor.insert %c1 into %csr[%c1_idx, %c1_idx] : tensor<8x8xf32, #CSR>
  return %csr0 : tensor<8x8xf32, #CSR>
}
