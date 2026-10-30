// RUN: spruce-opt -sparse-tensor-codegen -canonicalize %s | FileCheck %s

// CHECK: #[[CSR:.*]] = #sparse_tensor.encoding
#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: func.func @assemble_csr
// CHECK-SAME:      %[[ROWPTR:.*]]: tensor<5xindex>
// CHECK-SAME:      %[[COLIDX:.*]]: tensor<?xindex>
// CHECK-SAME:      %[[VALUES:.*]]: tensor<?xf64>

// CHECK:         %[[C5:.*]] = arith.constant 5 : index
// CHECK:         %[[C6:.*]] = arith.constant 6 : index
// CHECK:         %[[C4:.*]] = arith.constant 4 : index

// CHECK:         %[[ROWPTR_STATIC:.*]] = bufferization.to_buffer %[[ROWPTR]] : tensor<5xindex> to memref<5xindex>
// CHECK:         %[[ROWPTR_BUF:.*]] = memref.cast %[[ROWPTR_STATIC]] : memref<5xindex> to memref<?xindex>
// CHECK:         %[[COLIDX_BUF:.*]] = bufferization.to_buffer %[[COLIDX]] : tensor<?xindex> to memref<?xindex>
// CHECK:         %[[VALUES_BUF:.*]] = bufferization.to_buffer %[[VALUES]] : tensor<?xf64> to memref<?xf64>

// CHECK:         %[[SPEC0:.*]] = sparse_tensor.storage_specifier.init : !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:         %[[SPEC1:.*]] = sparse_tensor.storage_specifier.set %[[SPEC0]] lvl_sz at 0 with %[[C4]] : !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:         %[[SPEC2:.*]] = sparse_tensor.storage_specifier.set %[[SPEC1]] lvl_sz at 1 with %[[C6]] : !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:         %[[SPEC3:.*]] = sparse_tensor.storage_specifier.set %[[SPEC2]] pos_mem_sz at 1 with %[[C5]] : !sparse_tensor.storage_specifier<#[[CSR]]>

// CHECK:         %[[ENTRY_COUNT:.*]] = memref.load %[[ROWPTR_STATIC]][%[[C4]]] : memref<5xindex>
// CHECK:         %[[SPEC4:.*]] = sparse_tensor.storage_specifier.set %[[SPEC3]] crd_mem_sz at 1 with %[[ENTRY_COUNT]] : !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:         %[[SPEC:.*]] = sparse_tensor.storage_specifier.set %[[SPEC4]] val_mem_sz with %[[ENTRY_COUNT]] : !sparse_tensor.storage_specifier<#[[CSR]]>

// CHECK:         return %[[ROWPTR_BUF]], %[[COLIDX_BUF]], %[[VALUES_BUF]], %[[SPEC]]

func.func @assemble_csr(
  %rowptr: tensor<5xindex>,
  %colidx: tensor<?xindex>,
  %values: tensor<?xf64>
) -> tensor<4x6xf64, #CSR> {
  %a = sparse_tensor.assemble (%rowptr, %colidx), %values
      : (tensor<5xindex>, tensor<?xindex>), tensor<?xf64> to tensor<4x6xf64, #CSR>
  return %a : tensor<4x6xf64, #CSR>
}
