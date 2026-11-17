// RUN: spruce-opt -convert-spruce-to-gpu -verify-roundtrip %s | FileCheck %s --implicit-check-not=spruce.kernel --implicit-check-not=spruce.yield --implicit-check-not=bufferization. --implicit-check-not='memref<' --implicit-check-not=gpu.alloc --implicit-check-not=gpu.memcpy

// This prototype outlines tensor IR without bufferization or execution mapping.
// The outlined GPU entry function has no results and runs with one thread.

// CHECK-DAG: #[[TARGET:.*]] = #spruce.target<{m = 16, n = 8, k = 8}>
// CHECK-DAG: #[[BSR:.*]] = #sparse_tensor.encoding
#TARGET = #spruce.target<{m = 16, n = 8, k = 8}>
#BSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (
    d0 floordiv 16 : dense,
    d1 floordiv 8 : compressed,
    d0 mod 16 : dense,
    d1 mod 8 : dense
  )
}>

// CHECK:       module attributes {gpu.container_module} {
// CHECK-LABEL: gpu.module @spgemm_gpu {
// CHECK-LABEL: gpu.func @spgemm_kernel0(
// CHECK-SAME:      %[[A_D:[^:]+]]: tensor<32x16xf32, #[[BSR]]>,
// CHECK-SAME:      %[[B_D:[^:]+]]: tensor<8x8xf32>,
// CHECK-SAME:      %[[C_D:[^:]+]]: tensor<16x8xf32>,
// CHECK-SAME:      %[[P_D:[^:]+]]: index) kernel attributes {spruce.target = #[[TARGET]]} {
// CHECK:         %[[BLOCK:.*]] = spruce.get_block %[[A_D]][%[[P_D]]] : tensor<32x16xf32, #[[BSR]]> -> tensor<16x8xf32>
// CHECK:         %[[PRODUCT:.*]] = spruce.mma %[[BLOCK]], %[[B_D]], %[[C_D]] : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
// CHECK:         gpu.return
// CHECK-LABEL: func.func @spgemm(
// CHECK-SAME:      %[[A_H:[^:]+]]: tensor<32x16xf32, #[[BSR]]>,
// CHECK-SAME:      %[[B_H:[^:]+]]: tensor<8x8xf32>,
// CHECK-SAME:      %[[C_H:[^:]+]]: tensor<16x8xf32>,
// CHECK-SAME:      %[[P_H:[^:]+]]: index) {
// CHECK:         %[[C1:.*]] = arith.constant 1 : index
// CHECK:         gpu.launch_func @spgemm_gpu::@spgemm_kernel0
// CHECK-SAME:        blocks in (%[[C1]], %[[C1]], %[[C1]]) threads in (%[[C1]], %[[C1]], %[[C1]])
// CHECK-SAME:        args(%[[A_H]] : tensor<32x16xf32, #[[BSR]]>, %[[B_H]] : tensor<8x8xf32>, %[[C_H]] : tensor<16x8xf32>, %[[P_H]] : index)
// CHECK:         return

func.func @spgemm(%A_h: tensor<32x16xf32, #BSR>,
                  %B_h: tensor<8x8xf32>,
                  %C_h: tensor<16x8xf32>,
                  %p_h: index) {
  spruce.kernel(%A_d = %A_h : tensor<32x16xf32, #BSR>, %B_d = %B_h : tensor<8x8xf32>, %C_d = %C_h : tensor<16x8xf32>, %p_d = %p_h : index) {
    %block = spruce.get_block %A_d[%p_d] : tensor<32x16xf32, #BSR> -> tensor<16x8xf32>
    %product = spruce.mma %block, %B_d, %C_d : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
    spruce.yield
  } {spruce.target = #TARGET}
  return
}
