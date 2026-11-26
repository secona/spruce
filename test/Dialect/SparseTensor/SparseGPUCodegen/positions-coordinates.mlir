// RUN: spruce-opt -pre-sparsification-rewrite -sparse-reinterpret-map \
// RUN:   -sparsification="parallelization-strategy=dense-outer-loop" \
// RUN:   -sparse-gpu-codegen="num-threads=32" -verify-roundtrip %s -o %t
// RUN: FileCheck %s --input-file=%t
// RUN: FileCheck %s --check-prefix=BOUNDARY --input-file=%t

// CHECK: #[[BSR:.*]] = #sparse_tensor.encoding<{ map = (d0, d1) ->
// CHECK: #[[STORAGE:.*]] = #sparse_tensor.encoding<{ map = (d0, d1, d2, d3) ->
#BSR = #sparse_tensor.encoding<{
  map = (d0, d1) -> (
    d0 floordiv 16 : dense,
    d1 floordiv 8 : compressed,
    d0 mod 16 : dense,
    d1 mod 8 : dense
  )
}>

// CHECK-LABEL: gpu.module @sparse_kernels
// CHECK-LABEL: gpu.func @kernel0(
// CHECK-SAME:      %[[K_POS:[^:]+]]: memref<?xindex>,
// CHECK-SAME:      %[[K_CRD:[^:]+]]: memref<?xindex>,
// CHECK-SAME:      %[[K_VALUES:[^:]+]]: memref<?xf32>,
// CHECK-SAME:      %[[K_C:[^:]+]]: memref<32x8xf32>,
// CHECK-SAME:      %[[K_B:[^:]+]]: memref<16x8xf32>) kernel {
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C2:.*]] = arith.constant 2 : index
// CHECK:         gpu.thread_id x
// CHECK:         scf.for %[[ROW:.*]] = %{{.*}} to %[[C2]] step %{{.*}} {
// CHECK:           %[[START:.*]] = memref.load %[[K_POS]][%[[ROW]]] : memref<?xindex>
// CHECK:           %[[NEXT_ROW:.*]] = arith.addi %[[ROW]], %[[C1]] : index
// CHECK:           %[[END:.*]] = memref.load %[[K_POS]][%[[NEXT_ROW]]] : memref<?xindex>
// CHECK:           scf.for %[[P:.*]] = %[[START]] to %[[END]] step %[[C1]] {
// CHECK:             %[[COL:.*]] = memref.load %[[K_CRD]][%[[P]]] : memref<?xindex>
// CHECK:             memref.load %[[K_VALUES]][%{{.*}}] : memref<?xf32>
// CHECK:             memref.load %[[K_B]][%{{.*}}, %{{.*}}] : memref<16x8xf32>
// CHECK:         gpu.return

// CHECK-LABEL: func.func @spgemm(
// CHECK-SAME:      %[[A:[^:]+]]: tensor<32x16xf32, #[[BSR]]>,
// CHECK-SAME:      %[[B:[^:]+]]: tensor<16x8xf32>,
// CHECK-SAME:      %[[C:[^:]+]]: tensor<32x8xf32>) -> tensor<32x8xf32>
// CHECK:         %[[VIEW:.*]] = sparse_tensor.reinterpret_map %[[A]] : tensor<32x16xf32, #[[BSR]]> to tensor<2x2x16x8xf32, #[[STORAGE]]>
// CHECK:         %[[H_POS:.*]] = sparse_tensor.positions %[[VIEW]] {level = 1 : index} : tensor<2x2x16x8xf32, #[[STORAGE]]> to memref<?xindex>
// CHECK:         %[[H_CRD:.*]] = sparse_tensor.coordinates %[[VIEW]] {level = 1 : index} : tensor<2x2x16x8xf32, #[[STORAGE]]> to memref<?xindex>
// CHECK:         %[[POS_SIZE:.*]] = memref.dim %[[H_POS]], %{{.*}} : memref<?xindex>
// CHECK:         %[[D_POS:.*]], %[[POS_ALLOC:.*]] = gpu.alloc async [%{{.*}}] (%[[POS_SIZE]]) : memref<?xindex>
// CHECK:         gpu.memcpy async [%[[POS_ALLOC]]] %[[D_POS]], %[[H_POS]] : memref<?xindex>, memref<?xindex>
// CHECK:         %[[CRD_SIZE:.*]] = memref.dim %[[H_CRD]], %{{.*}} : memref<?xindex>
// CHECK:         %[[D_CRD:.*]], %[[CRD_ALLOC:.*]] = gpu.alloc async [%{{.*}}] (%[[CRD_SIZE]]) : memref<?xindex>
// CHECK:         gpu.memcpy async [%[[CRD_ALLOC]]] %[[D_CRD]], %[[H_CRD]] : memref<?xindex>, memref<?xindex>
// CHECK:         gpu.launch_func async @sparse_kernels::@kernel0 blocks
// CHECK-SAME:      args(%[[D_POS]] : memref<?xindex>, %[[D_CRD]] : memref<?xindex>,
// CHECK:         return

// metadata element loads in the host function.
// BOUNDARY-LABEL: gpu.func @kernel0(
// BOUNDARY-NOT:   sparse_tensor.positions
// BOUNDARY-NOT:   sparse_tensor.coordinates
// BOUNDARY:       gpu.return
// BOUNDARY-LABEL: func.func @spgemm(
// BOUNDARY-NOT:   memref.load
// BOUNDARY:       gpu.launch_func
// BOUNDARY-NOT:   memref.load
// BOUNDARY:       return

func.func @spgemm(%A: tensor<32x16xf32, #BSR>,
                  %B: tensor<16x8xf32>,
                  %C: tensor<32x8xf32>) -> tensor<32x8xf32> {
  %result = linalg.generic {
    indexing_maps = [
      affine_map<(i,k,j) -> (i,j)>,
      affine_map<(i,k,j) -> (j,k)>,
      affine_map<(i,k,j) -> (i,k)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"]
  } ins(%A, %B: tensor<32x16xf32, #BSR>, tensor<16x8xf32>)
    outs(%C: tensor<32x8xf32>) {
    ^bb0(%a: f32, %b: f32, %c: f32):
      %product = arith.mulf %a, %b : f32
      %sum = arith.addf %c, %product : f32
      linalg.yield %sum : f32
  } -> tensor<32x8xf32>
  return %result : tensor<32x8xf32>
}
