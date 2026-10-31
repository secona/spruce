// RUN: spruce-opt -split-input-file -allow-unregistered-dialect -sparse-tensor-codegen -canonicalize %s | FileCheck %s

// CHECK: #[[CSR:.*]] = #sparse_tensor.encoding
#CSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (d0 : dense, d1 : compressed)
}>

// CHECK-LABEL: func.func @storage_csr
// CHECK-SAME:      %[[POS:[^:]+]]: memref<?xindex>
// CHECK-SAME:      %[[CRD:[^:]+]]: memref<?xindex>
// CHECK-SAME:      %[[VAL:[^:]+]]: memref<?xf64>
// CHECK-SAME:      %[[SPEC:[^:]+]]: !sparse_tensor.storage_specifier<#[[CSR]]>

// CHECK:         %[[VAL_MEM_SZ:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] val_mem_sz : !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:         %[[VAL_SV:.*]] = memref.subview %[[VAL]][0] [%[[VAL_MEM_SZ]]] [1] : memref<?xf64> to memref<?xf64>

// CHECK:         %[[POS_MEM_SZ:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] pos_mem_sz at 1 : !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:         %[[POS_SV:.*]] = memref.subview %[[POS]][0] [%[[POS_MEM_SZ]]] [1] : memref<?xindex> to memref<?xindex>

// CHECK:         %[[CRD_MEM_SZ:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] crd_mem_sz at 1 : !sparse_tensor.storage_specifier<#[[CSR]]>
// CHECK:         %[[CRD_SV:.*]] = memref.subview %[[CRD]][0] [%[[CRD_MEM_SZ]]] [1] : memref<?xindex> to memref<?xindex>

// CHECK:         return

func.func @storage_csr(%arg0: tensor<?x?xf64, #CSR>) {
  %c0 = arith.constant 0 : index

  %val = sparse_tensor.values %arg0 : tensor<?x?xf64, #CSR> to memref<?xf64>
  %pos = sparse_tensor.positions %arg0 level = 1 : tensor<?x?xf64, #CSR> to memref<?xindex>
  %crd = sparse_tensor.coordinates %arg0 level = 1 : tensor<?x?xf64, #CSR> to memref<?xindex>

  "test.use"(%val) : (memref<?xf64>) -> ()
  "test.use"(%pos) : (memref<?xindex>) -> ()
  "test.use"(%crd) : (memref<?xindex>) -> ()

  return
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

// CHECK-LABEL: func.func @storage_bsr
// CHECK-SAME:      %[[POS:[^:]+]]: memref<?xindex>
// CHECK-SAME:      %[[CRD:[^:]+]]: memref<?xindex>
// CHECK-SAME:      %[[VAL:[^:]+]]: memref<?xf64>
// CHECK-SAME:      %[[SPEC:[^:]+]]: !sparse_tensor.storage_specifier<#[[BSR]]>

// CHECK:         %[[VAL_MEM_SZ:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] val_mem_sz : !sparse_tensor.storage_specifier<#[[BSR]]>
// CHECK:         %[[VAL_SV:.*]] = memref.subview %[[VAL]][0] [%[[VAL_MEM_SZ]]] [1] : memref<?xf64> to memref<?xf64>

// CHECK:         %[[POS_MEM_SZ:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] pos_mem_sz at 1 : !sparse_tensor.storage_specifier<#[[BSR]]>
// CHECK:         %[[POS_SV:.*]] = memref.subview %[[POS]][0] [%[[POS_MEM_SZ]]] [1] : memref<?xindex> to memref<?xindex>

// CHECK:         %[[CRD_MEM_SZ:.*]] = sparse_tensor.storage_specifier.get %[[SPEC]] crd_mem_sz at 1 : !sparse_tensor.storage_specifier<#[[BSR]]>
// CHECK:         %[[CRD_SV:.*]] = memref.subview %[[CRD]][0] [%[[CRD_MEM_SZ]]] [1] : memref<?xindex> to memref<?xindex>

// CHECK:         return

func.func @storage_bsr(%arg0: tensor<?x?xf64, #BSR>) {
  %c0 = arith.constant 0 : index

  %val = sparse_tensor.values %arg0 : tensor<?x?xf64, #BSR> to memref<?xf64>
  %pos = sparse_tensor.positions %arg0 level = 1 : tensor<?x?xf64, #BSR> to memref<?xindex>
  %crd = sparse_tensor.coordinates %arg0 level = 1 : tensor<?x?xf64, #BSR> to memref<?xindex>

  "test.use"(%val) : (memref<?xf64>) -> ()
  "test.use"(%pos) : (memref<?xindex>) -> ()
  "test.use"(%crd) : (memref<?xindex>) -> ()

  return
}
