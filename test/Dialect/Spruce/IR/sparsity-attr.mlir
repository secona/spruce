// RUN: spruce-opt -verify-roundtrip %s | FileCheck %s

// CHECK: #[[DIM0:.*]] = #spruce.dim_mask<2, [0, 1]>
// CHECK: #[[DIM1:.*]] = #spruce.dim_mask<2, [1, 0]>
// CHECK: #[[SPARSITY_ATTR:.*]] = #spruce.sparsity<[#[[DIM0]], #[[DIM1]]]>
// CHECK: module attributes {spruce.sparsity = #[[SPARSITY_ATTR]]}

#sparsity = #spruce.sparsity<[
  #spruce.dim_mask<2, [0, 1]>,
  #spruce.dim_mask<2, [1, 0]>
]>

module attributes {spruce.sparsity = #sparsity} {}
