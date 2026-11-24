// RUN: spruce-opt -convert-spruce-to-gpu -verify-roundtrip %s | FileCheck %s -DK=8 -DN=8 --implicit-check-not=spruce.kernel --implicit-check-not=spruce.mma --implicit-check-not="arith.constant 0.000000e+00 : f16"
// RUN: spruce-opt -convert-spruce-to-gpu="target=wmma" -verify-roundtrip %s | FileCheck %s -DK=16 -DN=16 --check-prefixes=CHECK,PAD
// RUN: spruce-opt -convert-spruce-to-gpu="target=mma" -gpu-kernel-outlining -verify-roundtrip %s | FileCheck %s --check-prefix=OUTLINE
// RUN: spruce-opt -pass-pipeline="builtin.module(convert-spruce-to-gpu{target=wmma},gpu-kernel-outlining,gpu.module(convert-gpu-to-nvvm))" %s | FileCheck %s --check-prefix=WMMA

#target = #spruce.target<{m = 16, n = 8, k = 8}>
!a = memref<16x8xf16, strided<[?, ?], offset: ?>>
!b = memref<8x8xf16, strided<[?, ?], offset: ?>>

// CHECK-LABEL: func.func @tile(
// CHECK-SAME:    %[[A:[^:]+]]: {{[^%]+}}%[[B:[^:]+]]: {{[^%]+}}%[[C:[^:]+]]:
// CHECK:         %[[ONE:.*]] = arith.constant 1 : index
// CHECK:         %[[WARP:.*]] = arith.constant 32 : index
// CHECK:         gpu.launch
// CHECK-SAME:      threads({{.*}} = %[[WARP]], {{.*}} = %[[ONE]], {{.*}} = %[[ONE]])
// CHECK-SAME:      workgroup(%[[AS:[^ ]+]] : memref<16x[[K]]xf16, #gpu.address_space<workgroup>>, %[[BS:[^ ]+]] : memref<[[K]]x[[N]]xf16, #gpu.address_space<workgroup>>, %[[CS:[^ ]+]] : memref<16x[[N]]xf16, #gpu.address_space<workgroup>>)
// PAD:             %[[ZERO:.*]] = arith.constant 0.000000e+00 : f16
// CHECK:           memref.load %[[A]][
// CHECK:           memref.store {{.*}}, %[[AS]][
// PAD:             memref.store %[[ZERO]], %[[AS]][
// CHECK:           memref.load %[[B]][
// CHECK:           memref.store {{.*}}, %[[BS]][
// PAD:             memref.store %[[ZERO]], %[[BS]][
// CHECK:           memref.load %[[C]][
// CHECK:           memref.store {{.*}}, %[[CS]][
// PAD:             memref.store %[[ZERO]], %[[CS]][
// CHECK:           gpu.barrier
// CHECK:           %[[AF:.*]] = gpu.subgroup_mma_load_matrix %[[AS]][{{.*}}] leadDimension [[K]]
// CHECK:           %[[BF:.*]] = gpu.subgroup_mma_load_matrix %[[BS]][{{.*}}] leadDimension [[N]]
// CHECK:           %[[CF:.*]] = gpu.subgroup_mma_load_matrix %[[CS]][{{.*}}] leadDimension [[N]]
// CHECK:           %[[DF:.*]] = gpu.subgroup_mma_compute %[[AF]], %[[BF]], %[[CF]]
// CHECK-SAME:        : !gpu.mma_matrix<16x[[K]]xf16, "AOp">, !gpu.mma_matrix<[[K]]x[[N]]xf16, "BOp"> -> !gpu.mma_matrix<16x[[N]]xf16, "COp">
// CHECK:           gpu.subgroup_mma_store_matrix %[[DF]], %[[CS]][{{.*}}] leadDimension [[N]]
// CHECK:           gpu.barrier
// CHECK-COUNT-4:   memref.store {{.*}}, %[[C]][
// CHECK:           gpu.terminator
// CHECK:         return %[[C]]
// OUTLINE: gpu.launch_func
// OUTLINE: gpu.func
// OUTLINE-SAME: workgroup(
// OUTLINE: gpu.subgroup_mma_compute
// OUTLINE-SAME: !gpu.mma_matrix<16x8xf16, "AOp">, !gpu.mma_matrix<8x8xf16, "BOp"> -> !gpu.mma_matrix<16x8xf16, "COp">
// OUTLINE: gpu.return
// WMMA-COUNT-3: nvvm.wmma.load
// WMMA: nvvm.wmma.mma
// WMMA: nvvm.wmma.store
func.func @tile(%a: !a, %b: !b, %c: !a) -> !a {
  %result = spruce.kernel(%a_d = %a : !a, %b_d = %b : !b, %c_d = %c : !a) {
    spruce.mma %a_d, %b_d, %c_d : !a, !b, !a
    spruce.yield %c_d : !a
  } -> (!a) {spruce.target = #target}
  return %result : !a
}
