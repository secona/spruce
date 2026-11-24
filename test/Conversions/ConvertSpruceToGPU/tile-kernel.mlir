// RUN: spruce-opt -convert-spruce-to-gpu -verify-roundtrip %s | FileCheck %s --implicit-check-not=spruce.kernel --implicit-check-not=spruce.mma
// RUN: spruce-opt -convert-spruce-to-gpu -gpu-kernel-outlining -verify-roundtrip %s | FileCheck %s --check-prefix=OUTLINE

#target = #spruce.target<{m = 16, n = 8, k = 8}>
!a = memref<16x8xf16, strided<[?, ?], offset: ?>>
!b = memref<8x8xf16, strided<[?, ?], offset: ?>>

// CHECK-LABEL: func.func @tile(
// CHECK-SAME:    %[[A:[^:]+]]: {{[^%]+}}%[[B:[^:]+]]: {{[^%]+}}%[[C:[^:]+]]:
// CHECK:         %[[ONE:.*]] = arith.constant 1 : index
// CHECK:         %[[WARP:.*]] = arith.constant 32 : index
// CHECK:         gpu.launch
// CHECK-SAME:      threads({{.*}} = %[[WARP]], {{.*}} = %[[ONE]], {{.*}} = %[[ONE]])
// CHECK-SAME:      workgroup(%[[AS:[^ ]+]] : memref<16x8xf16, #gpu.address_space<workgroup>>, %[[BS:[^ ]+]] : memref<8x8xf16, #gpu.address_space<workgroup>>, %[[CS:[^ ]+]] : memref<16x8xf16, #gpu.address_space<workgroup>>)
// CHECK:           memref.load %[[A]][
// CHECK:           memref.store {{.*}}, %[[AS]][
// CHECK:           memref.load %[[B]][
// CHECK:           memref.store {{.*}}, %[[BS]][
// CHECK:           memref.load %[[C]][
// CHECK:           memref.store {{.*}}, %[[CS]][
// CHECK:           gpu.barrier
// CHECK:           %[[AF:.*]] = gpu.subgroup_mma_load_matrix %[[AS]][{{.*}}] leadDimension 8
// CHECK:           %[[BF:.*]] = gpu.subgroup_mma_load_matrix %[[BS]][{{.*}}] leadDimension 8
// CHECK:           %[[CF:.*]] = gpu.subgroup_mma_load_matrix %[[CS]][{{.*}}] leadDimension 8
// CHECK:           %[[DF:.*]] = gpu.subgroup_mma_compute %[[AF]], %[[BF]], %[[CF]]
// CHECK:           gpu.subgroup_mma_store_matrix %[[DF]], %[[CS]][
// CHECK:           gpu.barrier
// CHECK-COUNT-4:   memref.store {{.*}}, %[[C]][
// CHECK:           gpu.terminator
// CHECK:         return %[[C]]
// OUTLINE: gpu.launch_func
// OUTLINE: gpu.func
// OUTLINE-SAME: workgroup(
// OUTLINE: gpu.subgroup_mma_compute
// OUTLINE: gpu.return
func.func @tile(%a: !a, %b: !b, %c: !a) -> !a {
  %result = spruce.kernel(%a_d = %a : !a, %b_d = %b : !b, %c_d = %c : !a) {
    spruce.mma %a_d, %b_d, %c_d : !a, !b, !a
    spruce.yield %c_d : !a
  } -> (!a) {spruce.target = #target}
  return %result : !a
}
