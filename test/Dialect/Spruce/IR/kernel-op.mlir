// RUN: spruce-opt -verify-roundtrip -split-input-file %s | FileCheck %s

// CHECK-LABEL: func.func @kernel_mma(
// CHECK-SAME:      %[[LHS:[^:]+]]: tensor<16x8xf32>,
// CHECK-SAME:      %[[RHS:[^:]+]]: tensor<8x8xf32>,
// CHECK-SAME:      %[[ACC:[^:]+]]: tensor<16x8xf32>) -> tensor<16x8xf32> {
// CHECK:         %[[RESULT:.*]] = spruce.kernel(%[[LHS]], %[[RHS]], %[[ACC]])
// CHECK-SAME:        : (tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32>) {
// CHECK:         ^bb0(%[[A:[^:]+]]: tensor<16x8xf32>,
// CHECK-SAME:          %[[B:[^:]+]]: tensor<8x8xf32>,
// CHECK-SAME:          %[[C:[^:]+]]: tensor<16x8xf32>):
// CHECK:           %[[PRODUCT:.*]] = spruce.mma %[[A]], %[[B]], %[[C]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
// CHECK:           spruce.yield %[[PRODUCT]] : tensor<16x8xf32>
// CHECK:         } -> (tensor<16x8xf32>)
// CHECK:         return %[[RESULT]] : tensor<16x8xf32>

func.func @kernel_mma(%lhs: tensor<16x8xf32>, %rhs: tensor<8x8xf32>, %acc: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %result = spruce.kernel(%lhs, %rhs, %acc) : (tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32>) {
  ^bb0(%a: tensor<16x8xf32>, %b: tensor<8x8xf32>, %c: tensor<16x8xf32>):
    %product = spruce.mma %a, %b, %c : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
    spruce.yield %product : tensor<16x8xf32>
  } -> (tensor<16x8xf32>)
  return %result : tensor<16x8xf32>
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
// CHECK-SAME:      %[[T:[^:]+]]: tensor<32x16xf32, #[[BSR]]>,
// CHECK-SAME:      %[[P:[^:]+]]: index) -> (tensor<16x8xf32>, index, index) {
// CHECK:         %[[RESULT:.*]]:3 = spruce.kernel(%[[T]], %[[P]]) : (tensor<32x16xf32, #[[BSR]]>, index) {
// CHECK:         ^bb0(%[[A:[^:]+]]: tensor<32x16xf32, #[[BSR]]>, %[[I:[^:]+]]: index):
// CHECK:           %[[VALUES:.*]] = spruce.get_block %[[A]][%[[I]]] : tensor<32x16xf32, #[[BSR]]> -> tensor<16x8xf32>
// CHECK:           %[[R:.*]], %[[C:.*]] = spruce.block_coordinates %[[A]][%[[I]]] : tensor<32x16xf32, #[[BSR]]>
// CHECK:           spruce.yield %[[VALUES]], %[[R]], %[[C]] : tensor<16x8xf32>, index, index
// CHECK:         } -> (tensor<16x8xf32>, index, index)
// CHECK:         return %[[RESULT]]#0, %[[RESULT]]#1, %[[RESULT]]#2 : tensor<16x8xf32>, index, index

func.func @kernel_sparse(%T: tensor<32x16xf32, #BSR>, %p: index) -> (tensor<16x8xf32>, index, index) {
  %block, %row, %col = spruce.kernel(%T, %p) : (tensor<32x16xf32, #BSR>, index) {
  ^bb0(%a: tensor<32x16xf32, #BSR>, %i: index):
    %values = spruce.get_block %a[%i] : tensor<32x16xf32, #BSR> -> tensor<16x8xf32>
    %r, %c = spruce.block_coordinates %a[%i] : tensor<32x16xf32, #BSR>
    spruce.yield %values, %r, %c : tensor<16x8xf32>, index, index
  } -> (tensor<16x8xf32>, index, index)
  return %block, %row, %col : tensor<16x8xf32>, index, index
}

// -----

// CHECK-LABEL: func.func @kernel_buffers(
// CHECK-SAME:      %[[BUFFER:[^:]+]]: memref<?xf32>,
// CHECK-SAME:      %[[SIZE:[^:]+]]: index,
// CHECK-SAME:      %[[VALUE:[^:]+]]: f32) {
// CHECK:         spruce.kernel(%[[BUFFER]], %[[SIZE]], %[[VALUE]]) : (memref<?xf32>, index, f32) {
// CHECK:         ^bb0(%[[B:[^:]+]]: memref<?xf32>, %[[N:[^:]+]]: index, %[[V:[^:]+]]: f32):
// CHECK-DAG:       %[[C0:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[C1:.*]] = arith.constant 1 : index
// CHECK:           scf.for %[[I:.*]] = %[[C0]] to %[[N]] step %[[C1]] {
// CHECK:             memref.store %[[V]], %[[B]][%[[I]]] : memref<?xf32>
// CHECK:           }
// CHECK:           spruce.yield
// CHECK:         }{{$}}
// CHECK:         return

func.func @kernel_buffers(%buffer: memref<?xf32>, %size: index, %value: f32) {
  spruce.kernel(%buffer, %size, %value) : (memref<?xf32>, index, f32) {
  ^bb0(%b: memref<?xf32>, %n: index, %v: f32):
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %n step %c1 {
      memref.store %v, %b[%i] : memref<?xf32>
    }
    spruce.yield
  }
  return
}

// -----

// CHECK-LABEL: func.func @kernel_no_inputs(
// CHECK:         %[[RESULT:.*]] = spruce.kernel() {
// CHECK:           %[[C1:.*]] = arith.constant 1 : index
// CHECK:           spruce.yield %[[C1]] : index
// CHECK:         } -> (index)
// CHECK:         return %[[RESULT]] : index

func.func @kernel_no_inputs() -> index {
  %result = spruce.kernel() {
    %c1 = arith.constant 1 : index
    spruce.yield %c1 : index
  } -> (index)
  return %result : index
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
