// RUN: spruce-opt -spruce-tile-mma -convert-spruce-to-vector -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -verify-roundtrip %s | FileCheck %s --check-prefix=BUFFERIZE --implicit-check-not=spruce.mma --implicit-check-not='tensor<'
// RUN: spruce-opt -spruce-tile-mma -convert-spruce-to-vector -one-shot-bufferize="bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map" -convert-vector-to-nvgpu -canonicalize -verify-roundtrip %s | FileCheck %s --check-prefix=NVGPU --implicit-check-not=spruce.mma --implicit-check-not=vector.contract --implicit-check-not='tensor<'

// BUFFERIZE-DAG: #[[TRANSPOSE:.*]] = affine_map<(d0, d1) -> (d1, d0)>
// BUFFERIZE-DAG: #[[LHS_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// BUFFERIZE-DAG: #[[RHS_MAP:.*]] = affine_map<(d0, d1, d2) -> (d1, d2)>
// BUFFERIZE-DAG: #[[ACC_MAP:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// BUFFERIZE-LABEL: func.func @supported_tile(
// BUFFERIZE-SAME:      %[[LHS:[^:]+]]: memref<16x8xf32>,
// BUFFERIZE-SAME:      %[[RHS:[^:]+]]: memref<8x8xf32>,
// BUFFERIZE-SAME:      %[[ACC:[^:]+]]: memref<16x8xf32>)
// BUFFERIZE-DAG:     %[[C0:.*]] = arith.constant 0 : index
// BUFFERIZE-DAG:     %[[ZERO:.*]] = arith.constant 0.000000e+00 : f32

// BUFFERIZE:         %[[LHS_VEC:.*]] = vector.transfer_read %[[LHS]][%[[C0]], %[[C0]]], %[[ZERO]] {in_bounds = [true, true]} : memref<16x8xf32>, vector<16x8xf32>
// BUFFERIZE:         %[[RHS_VEC:.*]] = vector.transfer_read %[[RHS]][%[[C0]], %[[C0]]], %[[ZERO]] {in_bounds = [true, true], permutation_map = #[[TRANSPOSE]]} : memref<8x8xf32>, vector<8x8xf32>
// BUFFERIZE:         %[[ACC_VEC:.*]] = vector.transfer_read %[[ACC]][%[[C0]], %[[C0]]], %[[ZERO]] {in_bounds = [true, true]} : memref<16x8xf32>, vector<16x8xf32>

// BUFFERIZE:         %[[PRODUCT:.*]] = vector.contract {indexing_maps = [#[[LHS_MAP]], #[[RHS_MAP]], #[[ACC_MAP]]], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} %[[LHS_VEC]], %[[RHS_VEC]], %[[ACC_VEC]] : vector<16x8xf32>, vector<8x8xf32> into vector<16x8xf32>
// BUFFERIZE:         vector.transfer_write %[[PRODUCT]], %[[ACC]][%[[C0]], %[[C0]]] {in_bounds = [true, true]} : vector<16x8xf32>, memref<16x8xf32>
// BUFFERIZE:         return %[[ACC]] : memref<16x8xf32>

// NVGPU-LABEL: func.func @supported_tile(
// NVGPU:         %[[LANE:.*]] = gpu.lane_id
// NVGPU:         %[[PRODUCT:.*]] = nvgpu.mma.sync(%{{.*}}, %{{.*}}, %{{.*}}) mmaShape = [16, 8, 8] : (vector<4x1xf32>, vector<2x1xf32>, vector<2x2xf32>) -> vector<2x2xf32>
// NVGPU:         vector.store
// NVGPU:         return

func.func @supported_tile(%lhs: tensor<16x8xf32>, %rhs: tensor<8x8xf32>, %acc: tensor<16x8xf32>) -> tensor<16x8xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
  return %result : tensor<16x8xf32>
}

// BUFFERIZE-LABEL: func.func @remainders(
// BUFFERIZE-SAME:      %{{[^:]+}}: memref<17x9xf32>,
// BUFFERIZE-SAME:      %{{[^:]+}}: memref<9x9xf32>,
// BUFFERIZE-SAME:      %{{[^:]+}}: memref<17x9xf32>)
// BUFFERIZE-COUNT-8: vector.contract
// BUFFERIZE:         return %{{.*}} : memref<17x9xf32>

// NVGPU-LABEL: func.func @remainders(
// NVGPU-COUNT-8: nvgpu.mma.sync
// NVGPU:         return %{{.*}} : memref<17x9xf32>

func.func @remainders(%lhs: tensor<17x9xf32>, %rhs: tensor<9x9xf32>, %acc: tensor<17x9xf32>) -> tensor<17x9xf32> {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<17x9xf32>, tensor<9x9xf32>, tensor<17x9xf32> -> tensor<17x9xf32>
  return %result : tensor<17x9xf32>
}

// BUFFERIZE-LABEL: func.func @preserve_accumulator(
// BUFFERIZE-SAME:      %[[LHS:[^:]+]]: memref<16x8xf32>,
// BUFFERIZE-SAME:      %[[RHS:[^:]+]]: memref<8x8xf32>,
// BUFFERIZE-SAME:      %[[ACC:[^:]+]]: memref<16x8xf32>)
// BUFFERIZE:         %[[ACC_VEC:.*]] = vector.transfer_read %[[ACC]]
// BUFFERIZE:         %[[PRODUCT:.*]] = vector.contract {{.*}} %{{.*}}, %{{.*}}, %[[ACC_VEC]]
// BUFFERIZE:         %[[OUTPUT:.*]] = memref.alloc(){{.*}} : memref<16x8xf32>
// BUFFERIZE:         vector.transfer_write %[[PRODUCT]], %[[OUTPUT]]
// BUFFERIZE:         return %[[OUTPUT]], %[[ACC]] : memref<16x8xf32>, memref<16x8xf32>

// NVGPU-LABEL: func.func @preserve_accumulator(
// NVGPU:         nvgpu.mma.sync
// NVGPU:         vector.store
// NVGPU:         return

func.func @preserve_accumulator(%lhs: tensor<16x8xf32>, %rhs: tensor<8x8xf32>, %acc: tensor<16x8xf32>) -> (tensor<16x8xf32>, tensor<16x8xf32>) {
  %result = spruce.mma %lhs, %rhs, %acc : tensor<16x8xf32>, tensor<8x8xf32>, tensor<16x8xf32> -> tensor<16x8xf32>
  return %result, %acc : tensor<16x8xf32>, tensor<16x8xf32>
}
