// RUN: spruce-opt -convert-spruce-to-gpu -verify-roundtrip %s | FileCheck %s --implicit-check-not=spruce.kernel --implicit-check-not=spruce.yield

// CHECK:       module attributes {gpu.container_module} {
// CHECK-LABEL: gpu.module @spgemm_gpu {
// CHECK-LABEL: gpu.func @spgemm_kernel0() kernel {
// CHECK:         gpu.printf "first"
// CHECK:         gpu.return
// CHECK-LABEL: gpu.func @spgemm_kernel1() kernel {
// CHECK:         gpu.printf "second"
// CHECK:         gpu.return
// CHECK-LABEL: func.func @spgemm() {
// CHECK:         gpu.launch_func @spgemm_gpu::@spgemm_kernel0
// CHECK:         gpu.launch_func @spgemm_gpu::@spgemm_kernel1
// CHECK:         return
// CHECK-LABEL: gpu.module @spmv_gpu {
// CHECK-LABEL: gpu.func @spmv_kernel0(
// CHECK-SAME:      %[[N_D:[^:]+]]: index) kernel {
// CHECK:         gpu.printf "spmv %llu", %[[N_D]] : index
// CHECK:         gpu.return
// CHECK-LABEL: func.func @spmv(
// CHECK-SAME:      %[[N_H:[^:]+]]: index) {
// CHECK:         gpu.launch_func @spmv_gpu::@spmv_kernel0
// CHECK-SAME:        args(%[[N_H]] : index)
// CHECK:         return

func.func @spgemm() {
  spruce.kernel() {
    gpu.printf "first"
    spruce.yield
  }
  spruce.kernel() {
    gpu.printf "second"
    spruce.yield
  }
  return
}

func.func @spmv(%n_h: index) {
  spruce.kernel(%n_d = %n_h : index) {
    gpu.printf "spmv %llu", %n_d : index
    spruce.yield
  }
  return
}
