// RUN: spruce-opt -verify-roundtrip -split-input-file %s | FileCheck %s --implicit-check-not=^bb0

// CHECK-LABEL: func.func @kernel_mma(
// CHECK-SAME:      %[[A_H:[^:]+]]: tensor<16x8xf32>,
// CHECK-SAME:      %[[B_H:[^:]+]]: tensor<8x8xf32>,
// CHECK-SAME:      %[[C_H:[^:]+]]: tensor<16x8xf32>) -> tensor<16x8xf32> {
// CHECK:         %[[RESULT_H:.*]] = spruce.kernel(%[[A_D:[^ ]+]] = %[[A_H]] : tensor<16x8xf32>,
// CHECK-SAME:        %[[B_D:[^ ]+]] = %[[B_H]] : tensor<8x8xf32>,
// CHECK-SAME:        %[[C_D:[^ ]+]] = %[[C_H]] : tensor<16x8xf32>) {
// CHECK:           %[[PRODUCT_D:.*]] = spruce.mma %[[A_D]], %[[B_D]], %[[C_D]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
// CHECK:           spruce.yield %[[PRODUCT_D]] : tensor<16x8xf32>
// CHECK:         } -> (tensor<16x8xf32>)
// CHECK:         return %[[RESULT_H]] : tensor<16x8xf32>

func.func @kernel_mma(%A_h: tensor<16x8xf32>, %B_h: tensor<8x8xf32>, %C_h: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %result_h = spruce.kernel(%A_d = %A_h : tensor<16x8xf32>, %B_d = %B_h : tensor<8x8xf32>, %C_d = %C_h : tensor<16x8xf32>) {
    %product_d = spruce.mma %A_d, %B_d, %C_d : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
    spruce.yield %product_d : tensor<16x8xf32>
  } -> (tensor<16x8xf32>)
  return %result_h : tensor<16x8xf32>
}

// -----

// CHECK-DAG: #[[BSR:.*]] = #sparse_tensor.encoding
#BSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (
    d0 floordiv 16 : dense,
    d1 floordiv 8 : compressed,
    d0 mod 16 : dense,
    d1 mod 8 : dense
  )
}>

// CHECK-LABEL: func.func @kernel_sparse(
// CHECK-SAME:      %[[A_H:[^:]+]]: tensor<32x16xf32, #[[BSR]]>,
// CHECK-SAME:      %[[P_H:[^:]+]]: index) -> (tensor<16x8xf32>, index, index) {
// CHECK:         %[[RESULT_H:.*]]:3 = spruce.kernel(%[[A_D:[^ ]+]] = %[[A_H]] : tensor<32x16xf32, #[[BSR]]>, %[[P_D:[^ ]+]] = %[[P_H]] : index) {
// CHECK:           %[[BLOCK_D:.*]] = spruce.get_block %[[A_D]][%[[P_D]]] : tensor<32x16xf32, #[[BSR]]> -> tensor<16x8xf32>
// CHECK:           %[[ROW_D:.*]], %[[COL_D:.*]] = spruce.block_coordinates %[[A_D]][%[[P_D]]] : tensor<32x16xf32, #[[BSR]]>
// CHECK:           spruce.yield %[[BLOCK_D]], %[[ROW_D]], %[[COL_D]] : tensor<16x8xf32>, index, index
// CHECK:         } -> (tensor<16x8xf32>, index, index)
// CHECK:         return %[[RESULT_H]]#0, %[[RESULT_H]]#1, %[[RESULT_H]]#2 : tensor<16x8xf32>, index, index

func.func @kernel_sparse(%A_h: tensor<32x16xf32, #BSR>, %p_h: index) -> (tensor<16x8xf32>, index, index) {
  %block_h, %row_h, %col_h = spruce.kernel(%A_d = %A_h : tensor<32x16xf32, #BSR>, %p_d = %p_h : index) {
    %block_d = spruce.get_block %A_d[%p_d] : tensor<32x16xf32, #BSR> -> tensor<16x8xf32>
    %row_d, %col_d = spruce.block_coordinates %A_d[%p_d] : tensor<32x16xf32, #BSR>
    spruce.yield %block_d, %row_d, %col_d : tensor<16x8xf32>, index, index
  } -> (tensor<16x8xf32>, index, index)
  return %block_h, %row_h, %col_h : tensor<16x8xf32>, index, index
}

// -----

// CHECK-LABEL: func.func @kernel_buffers(
// CHECK-SAME:      %[[BUFFER_H:[^:]+]]: memref<?xf32>,
// CHECK-SAME:      %[[SIZE_H:[^:]+]]: index,
// CHECK-SAME:      %[[VALUE_H:[^:]+]]: f32) {
// CHECK:         spruce.kernel(%[[BUFFER_D:[^ ]+]] = %[[BUFFER_H]] : memref<?xf32>, %[[SIZE_D:[^ ]+]] = %[[SIZE_H]] : index, %[[VALUE_D:[^ ]+]] = %[[VALUE_H]] : f32) {
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[I:.*]] = %[[C0]] to %[[SIZE_D]] step %[[C1]] {
// CHECK:             memref.store %[[VALUE_D]], %[[BUFFER_D]][%[[I]]] : memref<?xf32>
// CHECK:           }
// CHECK:           spruce.yield
// CHECK:         }{{$}}
// CHECK:         return

func.func @kernel_buffers(%buffer_h: memref<?xf32>, %size_h: index, %value_h: f32) {
  spruce.kernel(%buffer_d = %buffer_h : memref<?xf32>, %size_d = %size_h : index, %value_d = %value_h : f32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %size_d step %c1 {
      memref.store %value_d, %buffer_d[%i] : memref<?xf32>
    }
    spruce.yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @kernel_no_inputs(
// CHECK:         %[[RESULT_H:.*]] = spruce.kernel() {
// CHECK:           %[[C1_D:.*]] = arith.constant 1 : index
// CHECK:           spruce.yield %[[C1_D]] : index
// CHECK:         } -> (index)
// CHECK:         return %[[RESULT_H]] : index

func.func @kernel_no_inputs() -> index {
  %result_h = spruce.kernel() {
    %c1_d = arith.constant 1 : index
    spruce.yield %c1_d : index
  } -> (index)
  return %result_h : index
}

// -----

// CHECK-LABEL: func.func @kernel_no_inputs_or_results(
// CHECK:         spruce.kernel() {
// CHECK:           spruce.yield
// CHECK:         }{{$}}
// CHECK:         return

func.func @kernel_no_inputs_or_results() {
  spruce.kernel() {
    spruce.yield
  }
  return
}

// -----

// CHECK-DAG: #[[TARGET:.*]] = #spruce.target<{m = 16, n = 8, k = 8}>
#TARGET = #spruce.target<{m = 16, n = 8, k = 8}>

// CHECK-LABEL: func.func @kernel_attributes(
// CHECK:         spruce.kernel() {
// CHECK:           spruce.yield
// CHECK:         } {spruce.target = #[[TARGET]]}{{$}}
// CHECK:         return

func.func @kernel_attributes() {
  spruce.kernel() {
    spruce.yield
  } {spruce.target = #TARGET}
  return
}
